"""
opencall CTC-CRF Model.
"""

import torch
import numpy as np
from opencall.models.common.nn import (
    Module,
    Convolution,
    Serial,
    Permute,
    layers,
    from_dict,
)
import sys
import os
# Dynamically find the opencall/libs directory relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_dir = os.path.join(current_dir, '..', '..', 'libs')
sys.path.append(os.path.abspath(libs_dir))
from seqdist import sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring


class LinearClassifier(Module):
    def __init__(
        self,
        insize,
        num_classes=5,
        bias=True,
        activation=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.linear = torch.nn.Linear(insize, num_classes, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.linear.in_features,
            "num_classes": self.num_classes,
            "bias": self.linear.bias is not None,
            "activation": self.activation.name if self.activation else None,
        }
        if include_weights:
            res["params"] = {
                "W": self.linear.weight,
                "b": self.linear.bias if self.linear.bias is not None else [],
            }
        return res


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
    insize=1,
    stride=5,
    winlen=19,
    activation="swish",
    rnn_type="lstm",
    features=768,
    dropout=0.0,
    num_layers=5,
    num_classes=5,
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
            LinearClassifier(
                features,
                num_classes=num_classes,
                activation="tanh",
            ),
        ]
    )


class ClassificationModel(Module):
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.num_classes = num_classes
        self.alphabet = "ACGT"

    def forward(self, x):
        return self.encoder(x)

    def loss(self, scores, targets):
        # Reshape scores and targets for cross entropy
        # scores: (T, N, num_classes) -> (T*N, num_classes)
        # targets: (N, L) -> (T*N,)
        T, N, C = scores.shape
        
        # Zero pad targets to T length if needed
        if targets.size(1) != T:
            if targets.size(1) < T:
                # Zero pad to T length
                padding_size = T - targets.size(1)
                targets_padded = torch.cat([
                    targets, 
                    torch.zeros(N, padding_size, dtype=targets.dtype, device=targets.device)
                ], dim=1)
            else:
                # Truncate to T length
                targets_padded = targets[:, :T]
        else:
            targets_padded = targets
            
        scores_reshaped =  scores.permute(1,2,0)
        targets_reshaped = targets_padded  # (N,T), no need to reshape
        targets_reshaped = targets_reshaped.to(torch.int64)
        # Use cross entropy loss
        return torch.nn.functional.cross_entropy(scores_reshaped, targets_reshaped)

    def predict(self, x):
        scores = self.forward(x)
        return torch.argmax(scores, dim=-1)

    def decode(self, scores):

        predictions = torch.argmax(scores, dim=-1)  # (T, N)
        
        # Convert to numpy for easier processing
        predictions_np = predictions.cpu().numpy()
        
        # Use numpy to find non-zero values and convert directly to string
        # This is much faster than dictionary mapping + string joining
        # Only include predictions 1,2,3,4 (exclude 0 and values > 4)
        valid_mask = (predictions_np >= 1) & (predictions_np <= 4)
        if np.any(valid_mask):
            # Filter valid values (1-4) and convert to bytes then decode
            valid_values = predictions_np[valid_mask]
            # We need to map: 1->65(A), 2->67(C), 3->71(G), 4->84(T)
            ascii_mapping = np.array([0, 65, 67, 71, 84], dtype=np.uint8)  # 0->0, 1->A, 2->C, 3->G, 4->T
            ascii_values = ascii_mapping[valid_values]
            decoded_sequence = ascii_values.tobytes().decode('ascii')
            return decoded_sequence
        else:
            return ''

    def decode_batch(self, scores_batch):
        """
        Decode a batch of scores efficiently using CUDA when available.
        
        Args:
            scores_batch: Tensor of shape (T, N, C) where T is sequence length,
                         N is batch size, C is number of classes
        
        Returns:
            List of decoded sequences
        """
        predictions = torch.argmax(scores_batch, dim=-1)  # (T, N)
        
        if scores_batch.is_cuda:
            # CUDA version - process on GPU for speed
            return self._decode_batch_cuda(predictions)
        else:
            # CPU version - use multiprocessing
            return self._decode_batch_cpu(predictions)
    
    def _decode_batch_cuda(self, predictions):
        """CUDA-optimized batch decoding"""
        T, N = predictions.shape
        
        # Create valid mask for predictions 1,2,3,4
        valid_mask = (predictions >= 1) & (predictions <= 4)  # (T, N)
        
        # Create ASCII mapping tensor on GPU
        ascii_mapping = torch.tensor([0, 65, 67, 71, 84], dtype=torch.uint8, device=predictions.device)
        
        decoded_sequences = []
        
        # Process each sequence in the batch
        for seq_idx in range(N):
            seq_predictions = predictions[:, seq_idx]  # (T,)
            seq_valid_mask = valid_mask[:, seq_idx]    # (T,)
            
            if seq_valid_mask.any():
                # Get valid values
                valid_values = seq_predictions[seq_valid_mask]  # (valid_length,)
                
                # Map to ASCII values using GPU
                ascii_values = ascii_mapping[valid_values]  # (valid_length,)
                
                # Convert to CPU and decode
                decoded_sequence = ascii_values.cpu().numpy().tobytes().decode('ascii')
                decoded_sequences.append(decoded_sequence)
            else:
                decoded_sequences.append('')
        
        return decoded_sequences
    
    def _decode_batch_cpu(self, predictions):
        """CPU multiprocessing version for non-CUDA tensors"""
        # Convert to numpy for multiprocessing
        predictions_np = predictions.cpu().numpy()
        
        # Use multiprocessing for parallel decoding
        from multiprocessing import Pool, cpu_count
        
        # Determine number of processes
        num_sequences = predictions_np.shape[1]
        num_processes = min(16, num_sequences)
        
        if num_processes > 1 and num_sequences > 1:
            # Use multiprocessing for parallel decoding
            with Pool(processes=num_processes) as pool:
                # Prepare data for multiprocessing: each sequence as a separate array
                sequence_data = [predictions_np[:, seq_idx] for seq_idx in range(num_sequences)]
                decoded_sequences = pool.map(_decode_single_sequence, sequence_data)
        else:
            # Fallback to sequential processing for small batches or single CPU
            decoded_sequences = []
            for seq_idx in range(num_sequences):
                seq_predictions = predictions_np[:, seq_idx]
                decoded_sequences.append(_decode_single_sequence(seq_predictions))
        
        return decoded_sequences


def _decode_single_sequence(seq_predictions):
    """Helper function for multiprocessing - must be at module level to be picklable"""
    # Only include predictions 1,2,3,4 (exclude 0 and values > 4)
    valid_mask = (seq_predictions >= 1) & (seq_predictions <= 4)
    if np.any(valid_mask):
        # Filter valid values (1-4) and convert to bytes then decode
        valid_values = seq_predictions[valid_mask]
        # We need to map: 1->65(A), 2->67(C), 3->71(G), 4->84(T)
        ascii_mapping = np.array([0, 65, 67, 71, 84], dtype=np.uint8)  # 0->0, 1->A, 2->C, 3->G, 4->T
        ascii_values = ascii_mapping[valid_values]
        decoded_sequence = ascii_values.tobytes().decode('ascii')
        return decoded_sequence
    else:
        return ''


class Model(ClassificationModel):
    def __init__(self, config):
        if "type" in config["encoder"]:  # new-style config
            encoder = from_dict(config["encoder"])
        else:  # old-style
            encoder = rnn_encoder(
                insize=config["input"]["features"],
                num_classes=9,  # Fixed to 9 classes (N,A,T,C,G,A_,T_,C_,G_)# 改这里
                **config["encoder"]
            )
        super().__init__(encoder, num_classes=5)
        self.config = config
        self.tokenization = "classification"
