"""
Bonito CRF basecalling
"""

import torch
import numpy as np
from koi.decode import beam_search, to_str

from cyclonebasecall.bonito.multiprocessing import thread_iter
from cyclonebasecall.bonito.util import chunk, stitch, batchify, unbatchify
from ctypes import cdll, c_char_p

libcudart = cdll.LoadLibrary("libcudart.so")
libcudart.cudaGetErrorString.restype = c_char_p


def cudaSetDevice(device_idx):
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        raise RuntimeError("cudaSetDevice: " + error_string)


def tensor_permute(x, input_layout, output_layout):
    if input_layout == output_layout: return x
    return x.permute(*[input_layout.index(x) for x in output_layout])


def stitch_results(results, length, size, overlap, stride, reverse=False):
    """
    Stitch results together with a given overlap.
    """
    if isinstance(results, dict):
        return {
            k: stitch_results(v, length, size, overlap, stride, reverse=reverse)
            for k, v in results.items()
        }
    if length < size:
        return results[0, :int(np.floor(length / stride))]
    return stitch(results, size, overlap, length, stride, reverse=reverse)


def compute_scores(model, batch, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, reverse=False):
    """
    Compute scores for model using koi beam_search decoding.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = model.backbone(batch.to(dtype).to(device))
        scores = model.crfencoder(x)
        # print(scores.size(), batch.size())   # (64, 800, 4096)  (64, 1, 4000)
        scores = tensor_permute(scores, 'TNC', 'NTC')
        N, T, C = scores.size()
        a = scores.reshape(N, T, 1024, 5)
        b = a[:, :, :, 1:]
        c = b.reshape(N, T, -1)
        scores = c.contiguous()

        if reverse:
            scores = model.seqdist.reverse_complement(scores)
        # koi.beam_search only accepts fp16 -- the model may run in bf16 for
        # inference, so cast here rather than forcing the whole pipeline.
        if scores.dtype != torch.float16:
            scores = scores.to(torch.float16)
        cudaSetDevice(scores.device.index)
        sequence, qstring, moves = beam_search(
            scores, beam_width=beam_width, beam_cut=beam_cut,
            scale=scale, offset=offset, blank_score=blank_score
        )
        return {
            'moves': moves,
            'qstring': qstring,
            'sequence': sequence,
        }


# ============================================================
# Viterbi guided bidirectional decode
# ============================================================

def _manual_logsumexp(x, dim=-1, keepdim=False):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    result = x_max + torch.log2(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    if not keepdim:
        result = result.squeeze(dim)
    return result

# Map viterbi path indices (0=blank,1=A,2=C,3=G,4=T) to ASCII codes
_BASE_ASCII_MAP = torch.tensor([0, 65, 67, 71, 84], dtype=torch.int8)


def compute_scores_viterbi(model, batch, use_bfloat16=True,
                           q_shift=0.0, q_scale=1.0):
    """
    Compute scores using viterbi_guided_bidirectional_reshape decoding.
    Returns dict compatible with beam_search format for stitch/fmt.

    Per-position confidence is the (normalized) forward-backward edge posterior
    along the Viterbi path:

        log p_edge(t) = alpha[t,src] + Ms[t,dst,z] + beta[t+1,dst] - log Z_t

    A block that emits a base starts a new base; subsequent stays (blank edges)
    are attributed to that base. Per-base p_correct is the mean of p_edge over
    its contributing blocks, and Phred+33 quality follows Bonito / basecall_simple:

        Q = clip(-10 * log10(1 - p_correct) * q_scale + q_shift, 1, 50)
        qstring[i] = chr(int(33.5 + Q))
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        x = model.backbone(batch.to(dtype).to(device))
        scores = model.crfencoder(x)  # (T, N, C) with blank

        seqdist = model.seqdist
        n_base = seqdist.n_base
        state_len = seqdist.state_len
        n_states = n_base ** state_len
        n_alphabet = len(seqdist.alphabet)
        idx = seqdist.idx.to(device=device, dtype=torch.long)

        # Cache idx_T on seqdist
        if not hasattr(seqdist, '_idx_T') or seqdist._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)
            seqdist._idx_T = idx_T
            seqdist._idx_T_targets = idx_T // n_alphabet
        idx_T = seqdist._idx_T
        idx_T_targets = seqdist._idx_T_targets

        T, N, _ = scores.shape
        vdtype = torch.bfloat16 if use_bfloat16 else torch.float32
        Ms = scores.transpose(1, 2).to(vdtype).reshape(T, n_states, n_alphabet, N)
        Ms_T = Ms.reshape(T, -1, N)[:, idx_T, :]
        segment_size = 8

        # Forward
        alphas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=vdtype)
        alpha = alphas_all[0]
        for t in range(T):
            alpha = torch.logsumexp(alpha[idx, :] + Ms[t], dim=1)
            if t % segment_size == 0:
                alpha = alpha - alpha.max(dim=0, keepdim=True)[0]
            alphas_all[t + 1] = alpha

        # Backward
        betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=vdtype)
        beta = betas_all[T]
        for t in range(T - 1, -1, -1):
            beta = torch.logsumexp(Ms_T[t] + beta[idx_T_targets, :], dim=1)
            if t % segment_size == 0:
                beta = beta - beta.max(dim=0, keepdim=True)[0]
            betas_all[t] = beta

        # Posterior-normalized Viterbi
        # The key missing step: compute proper posterior probabilities via softmax
        # before running Viterbi. This matches the standard pipeline:
        #   posteriors = softmax(alpha_fwd[t,src] + Ms[t] + beta[t+1,dst])
        #   viterbi on log(posteriors)
        alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=vdtype)
        alpha_max[0, :] = 0.0
        traceback = torch.zeros(T, n_states, N, dtype=torch.int8, device=device)
        for t in range(T):
            # 在线计算 guided_scores = alpha[src] + Ms + beta[dst]
            alpha_indexed_src = alphas_all[t][idx, :]
            beta_indexed_dst = betas_all[t + 1][:, None, :]
            guided_scores_t = alpha_indexed_src + Ms[t] + beta_indexed_dst

            # Viterbi forward: alpha_max[dst] = max(alpha_max[src] + guided_scores)
            alpha_indexed_max = alpha_max[idx, :]
            candidates = alpha_indexed_max + guided_scores_t
            alpha_max, best_z = candidates.max(dim=1)  # 用 float 保证精度
            traceback[t] = best_z.to(torch.int8)

            if t % 8 == 0:
                alpha_max = alpha_max - alpha_max.max(dim=0, keepdim=True)[0]

        # --- Traceback: record Viterbi edges and destination states ---
        # `paths[t, n]`       = chosen edge at block t (0=blank/stay, 1-4=ACGT).
        # `dst_states[t, n]`  = CRF state AFTER block t (needed for the k-mer
        #                       marginal below).  Loop stays tight -- int-only
        #                       gathers, no float arithmetic in the hot path.
        current_states = alpha_max.argmax(dim=0)
        paths = torch.empty(T, N, dtype=torch.long, device=device)
        dst_states = torch.empty(T, N, dtype=torch.long, device=device)
        batch_idx = torch.arange(N, device=device)

        for t in range(T - 1, -1, -1):
            dst_states[t] = current_states
            best_edges = traceback[t, current_states, batch_idx].long()
            paths[t] = best_edges
            current_states = idx[current_states, best_edges]

        # --- Build the "shifted k-mer neighbor" table used by basecall_simple ---
        # For a CRF state s (integer-encoded k-mer), basecall_simple derives Q
        # from the marginal posterior over s and its 2*n_base de-Bruijn-graph
        # neighbors, with duplicates removed:
        #     l_shift(s, b) = (s >> log2(n_base)) + msb * b       -- prepend b
        #     r_shift(s, b) = (s << log2(n_base)) mod n_states + b -- append b
        # Slot order within cand_tbl (self, 4 l-shifts, 4 r-shifts) doesn't
        # affect the final block_prob: the dedup mask keeps exactly one copy
        # of each unique state regardless of ordering.  Depends only on
        # (n_states, n_base), so cache on seqdist across batches.
        if (not hasattr(seqdist, '_qual_cand_tbl')
                or seqdist._qual_cand_tbl.device != device):
            s = torch.arange(n_states, device=device, dtype=torch.long)
            b = torch.arange(n_base, device=device, dtype=torch.long)
            msb = n_states // n_base
            cand_tbl = torch.cat([
                s.unsqueeze(1),                                    # self
                (s // n_base).unsqueeze(1) + msb * b,              # l-shifts
                ((s * n_base) % n_states).unsqueeze(1) + b,        # r-shifts
            ], dim=1)                                              # (n_states, 1+2*n_base)

            # Slot k is "unique" iff it doesn't match any earlier slot j<k.
            cand_mask = torch.ones_like(cand_tbl, dtype=torch.bool)
            for k in range(1, cand_tbl.shape[1]):
                cand_mask[:, k] = (cand_tbl[:, k:k + 1] != cand_tbl[:, :k]).all(dim=-1)

            seqdist._qual_cand_tbl = cand_tbl
            seqdist._qual_cand_mask = cand_mask
        cand_tbl = seqdist._qual_cand_tbl
        cand_mask = seqdist._qual_cand_mask

        # --- Per-block k-mer-marginal posterior (matches basecall_simple) ---
        # posts[t+1, s, n] = softmax_s(alpha + beta)[t+1, s, n]
        # block_prob[t, n] = sum over unique cand_tbl[dst_states[t, n]] of posts
        #
        # Only 1+2*n_base states per (t, n) are ever touched, so we gather
        # directly from (alpha+beta) and normalize by a per-(t, n) logZ --
        # cheaper than materializing the full (T, N, n_states) softmax.
        ab_sum = (alphas_all + betas_all).float()              # (T+1, n_states, N)
        logZ = torch.logsumexp(ab_sum[1:T + 1], dim=1)          # (T, N)

        cand_path = cand_tbl[dst_states]                       # (T, N, C)
        mask_path = cand_mask[dst_states].float()              # (T, N, C) float32
        t_plus1 = torch.arange(1, T + 1, device=device).view(T, 1, 1)
        n_idx = batch_idx.view(1, N, 1)

        # ab_cand[t, n, c] = ab_sum[t+1, cand_path[t, n, c], n]
        ab_cand = ab_sum[t_plus1, cand_path, n_idx]            # (T, N, C)

        # Fuse exp -> dedup mask -> sum -> clamp -> ^0.4 fudge factor.
        block_prob = (
            (ab_cand - logZ.unsqueeze(-1)).exp_() * mask_path
        ).sum(dim=-1).clamp_(0.0, 1.0).pow_(0.4)               # (T, N)

        # --- Aggregate per base and build Phred+33 qstring ---
        # Transpose to (N, T): downstream cumsum / scatter_add / gather all
        # operate along T, which is the inner dim -> cache-friendly on CPU.
        paths = paths.T.contiguous()                           # (N, T) long
        block_prob = block_prob.T.contiguous()                 # (N, T) float32
        moves = (paths != 0)                                   # (N, T) bool

        # A move starts a new base; subsequent stays contribute to the same
        # base (same grouping as `generate_sequence()` in basecall_simple).
        moves_long = moves.long()                              # (N, T)
        label = moves_long.cumsum(dim=1)                       # 1-indexed base pos
        valid = (label > 0).to(block_prob.dtype)               # mask leading stays
        label_idx = (label - 1).clamp_(min=0)                  # 0-indexed

        # max_bases.item() is a CPU sync but free on CPU runs; on GPU it's
        # one small sync per batch.  Needed to size the scatter buffers tight.
        max_bases = max(int(moves_long.sum(dim=1).max()), 1) if N > 0 else 1

        sum_p = torch.zeros(N, max_bases, dtype=block_prob.dtype, device=device)
        cnt = torch.zeros(N, max_bases, dtype=block_prob.dtype, device=device)
        sum_p.scatter_add_(1, label_idx, block_prob * valid)
        cnt.scatter_add_(1, label_idx, valid)
        p_correct = sum_p / cnt.clamp_(min=1e-6)               # (N, max_bases)

        # Phred + affine calibration, fused into one expression.
        p_err = (1.0 - p_correct).clamp_(min=1e-6, max=1.0)
        qscore = (-10.0 * torch.log10(p_err) * q_scale + q_shift).clamp_(1.0, 50.0)
        qbyte_base = (33.5 + qscore).to(torch.int16)           # ASCII per base

        # Scatter per-base ASCII back to (N, T) interleaved layout: ASCII at
        # move positions, 0 elsewhere (downstream `to_str` strips zeros).
        sequence = _BASE_ASCII_MAP.to(device)[paths].to(torch.int8)  # (N, T) ASCII
        qstring = (torch.gather(qbyte_base, 1, label_idx) * moves_long).to(torch.int8)
        moves = moves.to(torch.int8)                           # (N, T) for output

        return {
            'moves': moves.cpu(),
            'qstring': qstring.cpu(),
            'sequence': sequence.cpu(),
        }


def fmt(stride, attrs):
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': to_str(attrs['qstring']),
        'sequence': to_str(attrs['sequence']),
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, model_stride=5,
             reverse=False, scale=1.0, offset=0.0, decode_method='beam_search'):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

    if decode_method == 'viterbi':
        scores = thread_iter(
            (read, compute_scores_viterbi(model, batch)) for read, batch in batches
        )
    else:
        scores = thread_iter(
            (read, compute_scores(model, batch, reverse=reverse, scale=scale, offset=offset)) for read, batch in batches
        )
        

    results = thread_iter(
        (read, stitch_results(scores, end - start, chunksize, overlap, model_stride, reverse))
        for ((read, start, end), scores) in unbatchify(scores)
    )

    return thread_iter(
        (read, fmt(model_stride, attrs))
        for read, attrs in results
    )
