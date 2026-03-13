"""
Bonito CRF basecalling
"""

import torch
import numpy as np
from koi.decode import beam_search, to_str

from cyclonebasecall.bonito.multiprocessing import thread_iter
from cyclonebasecall.bonito.util import chunk, stitch, batchify, unbatchify, half_supported
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
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))
        # print(scores.size(), batch.size())   # (64, 800, 4096)  (64, 1, 4000)
        scores = tensor_permute(scores, 'TNC', 'NTC')
        N, T, C = scores.size()
        a = scores.reshape(N, T, 1024, 5)
        b = a[:, :, :, 1:]
        c = b.reshape(N, T, -1)
        scores = c.contiguous()

        if reverse:
            scores = model.seqdist.reverse_complement(scores)
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


def compute_scores_viterbi(model, batch, use_bfloat16=True):
    """
    Compute scores using viterbi_guided_bidirectional_reshape decoding.
    Returns dict compatible with beam_search format for stitch/fmt.

    Per-position confidence is computed from the forward-backward posterior
    along the Viterbi path:  confidence[t] = alpha[t,src] + Ms[t,dst,z] + beta[t+1,dst]
    and mapped to Phred+33 quality scores in the qstring.
    """
    with torch.no_grad():
        device = next(model.parameters()).device
        dtype = torch.float16 if half_supported() else torch.float32
        scores = model(batch.to(dtype).to(device))  # (T, N, C) with blank

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
            # Edge posterior in log space: alpha(t, src) + Ms(t, dst, z) + beta(t+1, dst)
            edge_post = alphas_all[t][idx, :] + Ms[t] + betas_all[t + 1][:, None, :]
            # Softmax normalize across all edges per batch element (Log.dsum = softmax)
            # Use float32 for numerical stability, then convert back
            flat = edge_post.reshape(-1, N)
            log_post = (flat - flat.max(dim=0, keepdim=True)[0]).reshape(n_states, n_alphabet, N)
            # Standard Viterbi step on normalized log-posterior scores
            alpha_max, best_z = (alpha_max[idx, :] + log_post).max(dim=1)
            traceback[t] = best_z.to(torch.int8)
            if t % segment_size == 0:
                alpha_max = alpha_max - alpha_max.max(dim=0, keepdim=True)[0]

        # Traceback with per-position confidence
        current_states = alpha_max.argmax(dim=0)
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        confidence = torch.zeros(T, N, device=device, dtype=torch.float32)
        batch_idx = torch.arange(N, device=device)

        for t in range(T - 1, -1, -1):
            dst_state = current_states
            best_edges = traceback[t, dst_state, batch_idx]
            paths[t] = best_edges
            src_state = idx[dst_state, best_edges.long()]

            # Per-position forward-backward posterior along the Viterbi path:
            #   confidence = alpha(t, src) + Ms(t, dst, edge) + beta(t+1, dst)
            confidence[t] = (
                alphas_all[t][src_state, batch_idx]
                + Ms[t][dst_state, best_edges.long(), batch_idx]
                + betas_all[t + 1][dst_state, batch_idx]
            ).float()

            current_states = src_state

        # paths: (T, N), values 0-4 (0=blank, 1-4=ACGT)
        # Convert to (N, T) beam_search-compatible format
        paths = paths.T            # (N, T)
        confidence = confidence.T  # (N, T)

        ascii_map = _BASE_ASCII_MAP.to(device)
        sequence = ascii_map[paths.long()]            # (N, T) ASCII codes
        moves = (paths != 0).to(torch.int8)           # (N, T)

        # Convert confidence to Phred+33 quality scores
        # Normalize per sample: map [min, max] -> [0, 1] -> ASCII [33, 93]
        # ASCII 33 ('!') = Q0 (lowest), ASCII 93 (']') = Q60 (highest)
        c_min = confidence.min(dim=1, keepdim=True)[0]
        c_max = confidence.max(dim=1, keepdim=True)[0]
        c_range = (c_max - c_min).clamp(min=1e-6)
        c_norm = (confidence - c_min) / c_range       # [0, 1] per sample
        qscores = (33 + c_norm * 60).to(torch.int8)   # ASCII 33~93
        qstring = moves * qscores                      # 0 where blank

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
