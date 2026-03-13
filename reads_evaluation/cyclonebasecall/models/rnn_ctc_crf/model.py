"""
opencall CTC-CRF Model.
"""

import torch
import numpy as np
from cyclonebasecall.models.common.nn import Module, Convolution, LinearCRFEncoder, Serial, Permute, layers, from_dict

from seqdist import sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring


def get_stride(m):
    if hasattr(m, 'stride'):
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
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return sparse.bwd_scores_cupy(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        #convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation)


def build_backbone(insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, num_layers=5):
    """CNN + LSTM backbone, output shape (T, N, features)."""
    rnn = layers[rnn_type]
    return Serial([
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
            Permute([2, 0, 1]),
            *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
    ])


def rnn_encoder(n_base, state_len, insize=1, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5):
    rnn = layers[rnn_type]
    return Serial([
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
            Permute([2, 0, 1]),
            *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
            LinearCRFEncoder(
                features, n_base, state_len, activation='tanh', scale=scale,
                blank_score=blank_score, expand_blanks=expand_blanks
            )
    ])


class seqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        return self.encoder(x)

    def decode_batch(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()]

    def get_path(self, x):
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return tracebacks.cpu().numpy()[0]

    def decode(self, x):
        return self.decode_batch(x.unsqueeze(1))[0]

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs)

class Model(seqdistModel):

    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: #new-style config
            encoder = from_dict(config['encoder'])
            super().__init__(encoder, seqdist)
        else: #old-style
            # Build backbone (CNN+LSTM) and crfencoder (LinearCRFEncoder) separately
            enc = config['encoder']
            backbone = build_backbone(
                insize=config['input']['features'],
                stride=enc.get('stride', 5), winlen=enc.get('winlen', 19),
                activation=enc.get('activation', 'swish'), rnn_type=enc.get('rnn_type', 'lstm'),
                features=enc.get('features', 768), num_layers=enc.get('num_layers', 5),
            )
            # Pass backbone as encoder to seqdistModel (self.encoder = backbone)
            super().__init__(backbone, seqdist)
            self.crfencoder = LinearCRFEncoder(
                insize=enc.get('features', 768), n_base=seqdist.n_base, state_len=seqdist.state_len,
                activation='tanh', scale=enc.get('scale', 5.0),
                blank_score=enc.get('blank_score'), expand_blanks=True,
            )
        self.config = config
        self.tokenization = "kmer"

    @property
    def backbone(self):
        """Alias for self.encoder (CNN+LSTM backbone)."""
        return self.encoder

    def forward(self, x):
        x = self.encoder(x)
        if hasattr(self, 'crfencoder'):
            x = self.crfencoder(x)
        return x
