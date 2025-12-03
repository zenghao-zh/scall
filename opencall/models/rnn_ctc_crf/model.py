"""
opencall CTC-CRF Model.
"""

import torch
import numpy as np
from opencall.models.common.nn import (
    Module,
    Convolution,
    LinearCRFEncoder,
    Serial,
    Permute,
    layers,
    from_dict,
)
# import sys
# sys.path.append('/store/zjj/coding/tmp/opencall/opencall/libs')

from seqdist import sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring

try:
    from koi.decode import beam_search as koi_beam_search, to_str
    KOI_AVAILABLE = True
except ImportError:
    KOI_AVAILABLE = False


def get_stride(m):
    if hasattr(m, "stride"):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1


class CTC_CRF(SequenceDist):
    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat(
            [
                torch.arange(self.n_base ** (self.state_len))[:, None],
                torch.arange(self.n_base ** (self.state_len))
                .repeat_interleave(self.n_base)
                .reshape(self.n_base, -1)
                .T,
            ],
            dim=1,
        ).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base ** (self.state_len)

    def logZ(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return scores - self.logZ(scores)[:, None] / len(scores)

    def forward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (
            scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None]
        )
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat(
            [
                log_trans_probs[:, :, :, [0]],
                log_trans_probs[:, :, :, 1:]
                .transpose(3, 2)
                .reshape(T, N, -1, self.n_base),
            ],
            dim=-1,
        )
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        # convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = (
            T,
            N,
            *(self.n_base for _ in range(self.state_len)),
            self.n_base + 1,
        )
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(
            scores[..., 0]
            .permute(0, 1, *range(self.state_len + 1, 1, -1))
            .reshape(T, N, -1, 1),
            [0, 2],
        )
        emissions = torch.flip(
            scores[..., 1:]
            .permute(
                0,
                1,
                *range(self.state_len, 1, -1),
                self.state_len + 2,
                self.state_len + 1
            )
            .reshape(T, N, -1, self.n_base),
            [0, 2, 3],
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def viterbi_fused(self, scores):
        """
        融合的 Viterbi 解码：一次 forward 遍历 + traceback
        Args:
            scores: (T, N, C) 原始分数张量
        Returns:
            paths: (N, T) 解码路径，值为 0=blank, 1-4=ACGT
        """
        T, N, _ = scores.shape
        n_states = self.n_base ** self.state_len
        n_alphabet = len(self.alphabet)  # NZ = n_base + 1
        
        # Ms[t, n, c, z] = 在时间 t，从状态 idx[c,z] 转移到状态 c 的分数
        Ms = scores.reshape(T, N, n_states, n_alphabet)
        device = scores.device
        idx = self.idx.to(device=device, dtype=torch.long)  # (C, NZ)
        
        # ========== Forward Pass ==========
        # alpha[n, c] = 到达状态 c 的最优路径分数
        alpha = torch.zeros(N, n_states, device=device, dtype=scores.dtype)
        # traceback[t, n, c] = 在时间 t 到达状态 c 的最优边索引 z
        traceback = torch.zeros(T, N, n_states, dtype=torch.long, device=device)
        
        for t in range(T):
            # 获取所有入边的来源状态分数
            # alpha[:, idx] 形状: (N, C, NZ)
            prev_scores = alpha[:, idx]  # (N, C, NZ)
            # 加上边分数: candidates[n, c, z] = alpha[n, idx[c,z]] + Ms[t, n, c, z]
            candidates = prev_scores + Ms[t]  # (N, C, NZ)
            # 对每个目标状态，选择最优的入边
            alpha, traceback[t] = candidates.max(dim=-1)  # 都是 (N, C)
        # ========== Backward Traceback ==========
        # 找到终止时刻的最优状态
        current_states = alpha.argmax(dim=-1)  # (N,)
        # 回溯路径
        paths = torch.zeros(T, N, dtype=torch.long, device=device)
        batch_idx = torch.arange(N, device=device)
        for t in range(T - 1, -1, -1):
            # 获取到达 current_states 的最优边
            best_edges = traceback[t, batch_idx, current_states]  # (N,)
            # 边索引就是 action: 0=blank, 1=A, 2=C, 3=G, 4=T
            paths[t] = best_edges
            # 更新 current_states 为来源状态
            current_states = idx[current_states, best_edges]
        
        return paths.T  # (N, T)

    def path_to_str(self, path):
        alphabet = np.frombuffer("".join(self.alphabet).encode(), dtype="u1")
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i : n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)  # indices 用于在self.idx.flatten()里找kmer的下标
        move_indices = stay_indices[:, 1:] + targets[:, : n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(
        self,
        scores,
        targets,
        target_lengths,
        loss_clip=None,
        reduction="mean",
        normalise_scores=True,
    ):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = -(logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == "mean":
            return loss.mean()
        elif reduction in ("none", None):
            return loss
        else:
            raise ValueError("Unknown reduction type {}".format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(
            stay_scores, move_scores, target_lengths + 1 - self.state_len
        )


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(
        c_in,
        c_out,
        ks,
        stride=stride,
        padding=ks // 2,
        bias=bias,
        activation=activation,
    )


def rnn_encoder(
    n_base,
    state_len,
    insize=1,
    stride=5,
    winlen=19,
    activation="swish",
    rnn_type="lstm",
    features=768,
    scale=5.0,
    dropout=0.0,
    blank_score=None,
    expand_blanks=True,
    num_layers=5,
):
    rnn = layers[rnn_type]
    return Serial(
        [
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(
                16, features, ks=winlen, stride=stride, bias=True, activation=activation
            ),
            Permute([2, 0, 1]),
            *(
                rnn(features, features, reverse=(num_layers - i) % 2, dropout=dropout)
                for i in range(num_layers)
            ),
            LinearCRFEncoder(
                features,
                n_base,
                state_len,
                activation="tanh",
                scale=scale,
                blank_score=blank_score,
                expand_blanks=expand_blanks,
            ),
        ]
    )


class seqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        return self.encoder(x)

    def decode_batch(self, x, beam_width=1, beam_cut=100, scale=1.0, offset=0.0, blank_score=2.0, use_koi=False):
        """
        Decode a batch of scores using either koi beam_search or viterbi decoding.
        
        Args:
            x: scores tensor of shape (T, N, C)
            beam_width: beam width for beam search
            beam_cut: beam cut threshold for beam search
            scale: scale factor for beam search
            offset: offset for beam search
            blank_score: blank score for beam search
            use_koi: whether to use koi beam_search (if available)
            
        Returns:
            list of decoded sequences
        """
        if use_koi and KOI_AVAILABLE:
            # Use koi beam_search
            # Permute from (T, N, C) to (N, T, C)
            T, N, C = x.shape
            scores = x.permute(1, 0, 2).contiguous()
            
            # Reshape to remove blank: (N, T, n_states, n_alphabet) -> (N, T, n_states, n_bases)
            # C = n_states * n_alphabet, where n_alphabet = n_base + 1 (including blank)
            n_states = self.seqdist.n_base ** self.seqdist.state_len
            n_alphabet = len(self.seqdist.alphabet)  # n_base + 1 (including blank)
            
            # Reshape: (N, T, C) -> (N, T, n_states, n_alphabet)
            scores = scores.reshape(N, T, n_states, n_alphabet)
            # Remove blank (first element): keep [:, :, :, 1:]
            scores = scores[:, :, :, 1:]
            # Reshape back: (N, T, n_states, n_base) -> (N, T, n_states * n_base)
            scores = scores.reshape(N, T, -1).contiguous()
            
            # Convert to fp16 (required by koi)
            if scores.dtype != torch.float16:
                scores = scores.to(torch.float16)
            
            # Call koi beam_search
            sequence, qstring, moves = koi_beam_search(
                scores,
                beam_width=beam_width,
                beam_cut=beam_cut,
                scale=scale,
                offset=offset,
                blank_score=blank_score,
            )
            
            # Convert each sequence to string using to_str
            results = []
            for i in range(N):
                seq_str = to_str(sequence[i])
                results.append(seq_str)
            
            return results
        else:           # Fallback to viterbi decoding
            # scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
            # tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
            # return [self.seqdist.path_to_str(path) for path in tracebacks.cpu().numpy()]

            # Fallback to viterbi_fused (much faster than posteriors×2)
            with torch.no_grad():
                paths = self.seqdist.viterbi_fused(x.to(torch.float32))
            return [self.seqdist.path_to_str(path) for path in paths.cpu().numpy()]

    def get_path(self, x):
        with torch.no_grad():
            paths = self.seqdist.viterbi_fused(x.to(torch.float32))
        return paths.cpu().numpy()[0]

    def decode(self, x, beam_width=5, beam_cut=1e-3, scale=1.0, offset=0.0, blank_score=2.0, use_koi=True):
        """
        Decode a single sample.
        
        Args:
            x: scores tensor of shape (T, C) or (T, 1, C)
            beam_width: beam width for beam search
            beam_cut: beam cut threshold for beam search
            scale: scale factor for beam search
            offset: offset for beam search
            blank_score: blank score for beam search
            use_koi: whether to use koi beam_search (if available)
            
        Returns:
            decoded sequence string
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (T, C) -> (T, 1, C)
        return self.decode_batch(
            x, 
            beam_width=beam_width, 
            beam_cut=beam_cut, 
            scale=scale,
            offset=offset,
            blank_score=blank_score,
            use_koi=use_koi
        )[0]

    def greedy_decode(self, x):
        """
        Greedy decoding guided by backward scores (equivalent to beam_search with k=1).
        
        This uses forward-backward algorithm:
        1. Compute backward scores for guidance
        2. At each time step, select transition with highest (edge_score + backward_score)
        3. Follow the greedy path through the lattice
        
        Args:
            x: scores tensor of shape (T, C) or (T, N, C)
            
        Returns:
            decoded sequence string (if T, C) or list of strings (if T, N, C)
        """
        # Handle single sample or batch
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (T, C) -> (T, 1, C)
            single_sample = True
        else:
            single_sample = False
        
        T, N, C = x.shape
        x = x.to(torch.float32)
        
        # Get model parameters
        n_states = self.seqdist.n_base ** self.seqdist.state_len
        n_alphabet = len(self.seqdist.alphabet)  # n_base + 1 (including blank)
        
        # Reshape scores: (T, N, C) -> (T, N, n_states, n_alphabet)
        scores = x.reshape(T, N, n_states, n_alphabet)
        
        # Compute backward scores for guidance: (T+1, N, n_states)
        betas = self.seqdist.backward_scores(x)
        
        # Add backward scores to edge scores
        # For transition (old_state, action) -> new_state at time t:
        # guided_score = edge_score[t, old_state, action] + beta[t+1, new_state]
        idx_long = self.seqdist.idx.long().to(betas.device)  # (n_states, n_alphabet)
        betas_for_transitions = torch.gather(
            betas[1:].unsqueeze(2).expand(T, N, n_states, n_states),  # (T, N, n_states, n_states)
            dim=3,
            index=idx_long.unsqueeze(0).unsqueeze(0).expand(T, N, -1, -1)  # (T, N, n_states, n_alphabet)
        )  # Result: (T, N, n_states, n_alphabet)
        guided_scores = scores + betas_for_transitions
        
        # Initialize: all samples start from state 0 (e.g., "AAAAA" for 5-mer)
        current_states = torch.zeros(N, dtype=torch.long, device=x.device)
        paths = []
        
        # Greedy forward pass using guided scores
        for t in range(T):
            batch_idx = torch.arange(N, device=x.device, dtype=torch.long)
            state_scores = guided_scores[t, batch_idx, current_states, :]  # (N, n_alphabet)
            
            # Greedy: select action with highest guided score
            best_actions = state_scores.argmax(dim=1)  # (N,)
            
            # Transition to next states
            next_states = self.seqdist.idx[current_states, best_actions].long()  # (N,)
            
            paths.append(best_actions.cpu())
            current_states = next_states
        
        # Convert to sequences
        paths = torch.stack(paths, dim=0).T.numpy()  # (N, T)
        results = [self.seqdist.path_to_str(paths[n]) for n in range(N)]
        
        return results[0] if single_sample else results
    
    def greedy_decode_batch(self, x):
        """
        Batch version of greedy decode.
        
        Args:
            x: scores tensor of shape (T, N, C)
            
        Returns:
            list of decoded sequences
        """
        return self.greedy_decode(x)

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(
            scores.to(torch.float32), targets, target_lengths, **kwargs
        )


class Model(seqdistModel):
    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config["global_norm"]["state_len"],
            alphabet=config["labels"]["labels"],
        )
        if "type" in config["encoder"]:  # new-style config
            encoder = from_dict(config["encoder"])
        else:  # old-style
            encoder = rnn_encoder(
                seqdist.n_base,
                seqdist.state_len,
                insize=config["input"]["features"],
                **config["encoder"]
            )
        super().__init__(encoder, seqdist)
        self.config = config
        self.tokenization = "kmer"
