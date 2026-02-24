"""
Self-contained Viterbi evaluation script.

Usage:
    python /workspace/huada/scall/viterbi_0211.py \
    --model_dir /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214 \
    --data_dir /workspace/huada/moffett_data/250F600274011_train_data/val \
    --device cuda:0 \
    --val_batch_size 64 \
    --seed 25
"""

import os
import sys
import re
import time
import argparse
import random
from collections import defaultdict, OrderedDict

import toml
import torch
from torch.nn import Module
from torch.nn.init import orthogonal_
from torch.utils.data import Dataset, DataLoader
from sklearn import utils
import numpy as np
import parasail

pro_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, pro_dir)


# ============================================================
# NN Building Blocks
# ============================================================
layers = {}

def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer

register(torch.nn.Tanh)

@register
class Swish(torch.nn.SiLU):
    pass

@register
class Serial(torch.nn.Sequential):
    def __init__(self, sublayers):
        super().__init__(*sublayers)

@register
class Convolution(Module):
    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)

@register
class Permute(Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(size + (num_resample,), dtype=torch.float32, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)

def manual_logsumexp(x, dim=-1, keepdim=False):
    """
    手动实现 logsumexp，数值稳定版本
    
    等价于: torch.logsumexp(x, dim=dim, keepdim=keepdim)
    """
    # 步骤1: 找到最大值（防止 exp 溢出）
    x_max = x.max(dim=dim, keepdim=True)[0]
    
    # 步骤2: 减去最大值
    x_shifted = x - x_max
    
    # 步骤3: exp -> sum -> log
    result = x_max + torch.log2(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    
    # 步骤4: 处理 keepdim
    if not keepdim:
        result = result.squeeze(dim)
    
    return result

class RNNWrapper(Module):
    def __init__(self, rnn_type, *args, reverse=False, orthogonal_weight_init=True,
                 disable_state_bias=True, bidirectional=False, **kwargs):
        super().__init__()
        self.reverse = reverse
        self.rnn = rnn_type(*args, bidirectional=bidirectional, **kwargs)
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias:
            self.disable_state_bias()

    def forward(self, x):
        if self.reverse:
            x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse:
            y = y.flip(0)
        return y

    def init_biases(self, types=("bias_ih",)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    param.set_(0.5 * truncated_normal(param.shape, dtype=param.dtype, device=param.device))

    def init_orthogonal(self, types=True):
        if not types:
            return
        if types is True:
            types = ("weight_ih", "weight_hh")
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i : i + self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if "bias_hh" in name:
                x.requires_grad = False
                x.zero_()

@register
class LSTM(RNNWrapper):
    def __init__(self, size, insize, bias=True, reverse=False, dropout=0.0):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=reverse, dropout=dropout)

@register
class LinearCRFEncoder(Module):
    def __init__(self, insize, n_base, state_len, bias=True, scale=None,
                 activation=None, blank_score=None, expand_blanks=True):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = ((n_base + 1) * n_base ** state_len if blank_score is None
                else n_base ** (state_len + 1))
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score,
            ).view(T, N, -1)
        return scores

# ============================================================
# Fake Quantization
# ============================================================

class FakeQuant(torch.nn.Module):
    def __init__(self, num_bits, max_val):
        super(FakeQuant, self).__init__()
        self.scale = max_val / (2**(num_bits - 1) - 1)
        self.num_bits = num_bits

    def forward(self, x):
        scale = self.scale.to(dtype=x.dtype, device=x.device)
        x = torch.clamp(x / scale, -2**(self.num_bits - 1), 2**(self.num_bits - 1) - 1)
        x = torch.round(x)
        x = x * scale
        return x


def insert_fakequant_backbone(model, act_scales, bitwidth, device):
    """Insert FakeQuant after LSTM layers in backbone (skip the last LSTM)."""
    num_layers = len(model.backbone._modules)
    for i, (name, module) in enumerate(model.backbone._modules.items()):
        if isinstance(module, LSTM):
            if i != num_layers - 1:  # skip last LSTM
                scale_key = f'encoder.{name}'
                model.backbone._modules[name] = torch.nn.Sequential(
                    module,
                    FakeQuant(bitwidth, act_scales[scale_key]["output"].to(device))
                )
    return model


# ============================================================
# Helper utilities
# ============================================================

def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks // 2, bias=bias, activation=activation)

def get_stride(m):
    if hasattr(m, "stride"):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1

def match_names(state_dict, model):
    """Match weight names between checkpoint and model by shape sorting."""
    keys_and_shapes = lambda sd: zip(
        *[(k, s) for s, i, k in sorted(
            [(v.shape, i, k) for i, (k, v) in enumerate(sd.items())]
        )]
    )
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2, "Model architecture does not match checkpoint weights!"
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

def parasail_to_sam(result, seq):
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)
    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query
    clip = "" if cliplen == 0 else "{}S".format(cliplen)
    if first_op == "I":
        pre = "{}S".format(int(first_count) + cliplen)
    elif first_op == "D":
        pre = clip
        rstart = int(first_count)
    else:
        pre = "{}{}".format(clip, prefix)
    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = "{}S".format(end_clip) if end_clip > 0 else ""
    return rstart, "".join((pre, mid, suf))

def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)
    if len(alignment.traceback.ref) / len(ref) < min_coverage:
        return 0.0
    _, cigar = parasail_to_sam(alignment, seq)
    for count, op in re.findall(split_cigar, cigar):
        counts[op] += int(count)
    if balanced:
        acc = (counts["="] - counts["I"]) / (counts["="] + counts["X"] + counts["D"])
    else:
        acc = counts["="] / (counts["="] + counts["I"] + counts["X"] + counts["D"])
    return acc * 100

ascii_mapping_tensor = torch.tensor([0, 65, 67, 71, 84], dtype=torch.uint8)

def decode_ref(encoded, labels):
    valid_mask = (encoded >= 1) & (encoded <= 4)
    valid_values = encoded[valid_mask].to(torch.int64)
    return ascii_mapping_tensor[valid_values].cpu().numpy().tobytes().decode('ascii')

# ============================================================
# CTC_CRF (Viterbi decoding)
# ============================================================

class CTC_CRF:
    def __init__(self, state_len, alphabet):
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet) - 1
        self.idx = torch.cat([
            torch.arange(self.n_base ** self.state_len)[:, None],
            torch.arange(self.n_base ** self.state_len)
            .repeat_interleave(self.n_base)
            .reshape(self.n_base, -1).T,
        ], dim=1).to(torch.int32)

    def viterbi_guided_bidirectional_reshape(self, scores, use_bfloat16=True):
        T, N, _ = scores.shape
        n_states = self.n_base ** self.state_len
        n_alphabet = len(self.alphabet)
        device = scores.device
        idx = self.idx.to(device=device, dtype=torch.long)

        if not hasattr(self, '_idx_T') or self._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)
            self._idx_T = idx_T
            self._idx_T_targets = idx_T // n_alphabet

        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        Ms = scores.transpose(1, 2).to(dtype).reshape(T, n_states, n_alphabet, N)
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        Ms_T = Ms.reshape(T, -1, N)[:, idx_T, :]
        segment_size = 8

        # Forward
        alphas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
        alpha = alphas_all[0]
        for t in range(T):
            alpha = manual_logsumexp(alpha[idx, :] + Ms[t], dim=1)
            if t % segment_size == 0:
                alpha = alpha - alpha.min(dim=0, keepdim=True)[0]
            alphas_all[t + 1] = alpha

        # Backward
        betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
        beta = betas_all[T]
        for t in range(T - 1, -1, -1):
            beta = manual_logsumexp(Ms_T[t] + beta[idx_T_targets, :], dim=1)
            if t % segment_size == 0:
                beta = beta - beta.min(dim=0, keepdim=True)[0]
            betas_all[t] = beta

        # Guided Viterbi
        alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=dtype)
        alpha_max[0, :] = 0.0
        traceback = torch.zeros(T, n_states, N, dtype=torch.int8, device=device)
        for t in range(T):
            guided = alphas_all[t][idx, :] + Ms[t] + betas_all[t + 1][:, None, :]
            alpha_max, best_z = (alpha_max[idx, :] + guided).max(dim=1)
            traceback[t] = best_z.to(torch.int8)
            if t % segment_size == 0:
                alpha_max = alpha_max - alpha_max.max(dim=0, keepdim=True)[0]

        # Traceback
        current_states = alpha_max.argmax(dim=0)
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        batch_idx = torch.arange(N, device=device)
        for t in range(T - 1, -1, -1):
            best_edges = traceback[t, current_states, batch_idx]
            paths[t] = best_edges
            current_states = idx[current_states, best_edges.long()]
        return paths.T.to(torch.long)

    def path_to_str(self, path):
        alphabet = np.frombuffer("".join(self.alphabet).encode(), dtype="u1")
        return alphabet[path[path != 0]].tobytes().decode()

# ============================================================
# Model: backbone (CNN+LSTM) + crfencoder (LinearCRFEncoder)
# ============================================================

def build_backbone(insize=1, stride=5, winlen=19, activation="swish",
                   rnn_type="lstm", features=768, dropout=0.0, num_layers=5):
    """CNN + LSTM backbone, output shape (T, N, features)."""
    rnn = layers[rnn_type]
    return Serial([
        conv(insize, 4, ks=5, bias=True, activation=activation),
        conv(4, 16, ks=5, bias=True, activation=activation),
        conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation),
        Permute([2, 0, 1]),
        *(rnn(features, features, reverse=(num_layers - i) % 2, dropout=dropout)
          for i in range(num_layers)),
    ])


class Model(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        state_len = config["global_norm"]["state_len"]
        alphabet = config["labels"]["labels"]
        n_base = len(alphabet) - 1
        enc = config["encoder"]

        # CNN + LSTM backbone
        self.backbone = build_backbone(
            insize=config["input"]["features"],
            stride=enc["stride"], winlen=enc["winlen"],
            activation=enc["activation"], rnn_type=enc["rnn_type"],
            features=enc["features"], num_layers=enc["num_layers"],
        )
        self.stride = get_stride(self.backbone)

        # LinearCRFEncoder head
        self.crfencoder = LinearCRFEncoder(
            insize=enc["features"], n_base=n_base, state_len=state_len,
            activation="tanh", scale=enc.get("scale", 5.0),
            blank_score=enc.get("blank_score"), expand_blanks=True,
        )

        # Viterbi decoder
        self.seqdist = CTC_CRF(state_len=state_len, alphabet=alphabet)
        self.alphabet = alphabet

    def forward(self, x):
        x = self.backbone(x)
        x = self.crfencoder(x)
        return x

    def decode_batch(self, scores, use_bfloat16=True):
        with torch.no_grad():
            dtype = torch.bfloat16 if use_bfloat16 else torch.float32
            paths = self.seqdist.viterbi_guided_bidirectional_reshape(
                scores.to(dtype), use_bfloat16=use_bfloat16
            )
        return [self.seqdist.path_to_str(path) for path in paths.cpu().numpy()]


# ============================================================
# Model loading
# ============================================================

def load_model(config, weight_path, device, half=False):
    device = torch.device(device)
    model = Model(config)
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
    model.load_state_dict(state_dict)
    if half:
        model = model.half()
    model.eval()
    model.to(device)
    return model


def load_quant_model(config, io_quant_path, act_scales_path, device, half=False, bitwidth=8):
    """Load a model with FakeQuant layers inserted (produced by lstmIOquant_jjy.py)."""
    device = torch.device(device)
    model = Model(config)

    # Insert FakeQuant layers using act_scales
    act_scales = torch.load(act_scales_path, map_location=device)
    insert_fakequant_backbone(model, act_scales, bitwidth, device)

    # Load quantized weights
    state_dict = torch.load(io_quant_path, map_location=device)
    state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
    model.load_state_dict(state_dict)

    if half:
        model = model.half()
    model.eval()
    model.to(device)
    return model


# ============================================================
# Data loading (inline from opencall)
# ============================================================
import glob
import h5py
from sklearn import utils as sk_utils

class TrainingDataSet3(Dataset):
    def __init__(self, data_dir, tokenization):
        self.tokenization = tokenization
        self._load_hd5_npy(data_dir)

    def _load_hd5_npy(self, data_dir):
        # npy_path = '/workspace/basecall_data/train_data/wt_hac_r2.1.1-20240325'
        self.maxlen = 0
        hd5_dir = glob.glob(f'{data_dir}/*.hd5')
        self.hd5_list = []
        npy_list = []
        hd5_num = 0
        for i in range(len(hd5_dir)):
            try:
                hd5_file = h5py.File(hd5_dir[i], 'r')
            except Exception as e:
                continue
            self.hd5_list.append(hd5_file)
            dat_npy = np.load(f"{os.path.dirname(hd5_dir[i])}/{os.path.basename(hd5_dir[i]).split('.')[0]}.npy")
            if dat_npy.shape[1] == 5:
                dat_npy = np.column_stack((dat_npy, np.array([0]*dat_npy.shape[0])))
            if dat_npy[-1, 0] > self.maxlen:
                self.maxlen = int(dat_npy[-1, 0])
            dat_npy = np.column_stack((dat_npy, np.array([hd5_num]*dat_npy.shape[0])))
            npy_list.append(dat_npy[0:-1, :])
            hd5_num += 1
        dat_np = np.concatenate(npy_list, axis = 0).astype(int)
        self.region_np = utils.shuffle(dat_np, random_state=0)

    def _load_hd5(self, read_index, cur_start, cur_end, ref_start, ref_end, hd5_index):
        hd5_file = self.hd5_list[hd5_index]
        per_num = hd5_file.attrs['batch_size']
        batch_num = int(read_index // per_num)
        read = hd5_file['batch_{}'.format(batch_num)][str(read_index)]
        cur = self._get_current(read, (cur_start, cur_end), standardize=True)
        ref_start2 = ref_start + 0
        ref_end2 = ref_end - 0
        refs = read['Seq'][ref_start2:ref_end2]
        return cur, refs
    
    def _get_signal(self, read, region=None):
        if region is None:
            return read['Signal']
        a, b = region
        return read['Signal'][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read.attrs['offset']) * read.attrs['range'] / read.attrs['digitisation']
        if standardize:
            current = (current - read.attrs['shift_frompA']) / read.attrs['scale_frompA']
        return current

    def __getitem__(self, index):
        read_index, cur_start, cur_end, ref_start, ref_end, is_first_chunk, hd5_index = self.region_np[index, :].tolist()
        cur, refs = self._load_hd5(read_index, cur_start, cur_end, ref_start, ref_end, hd5_index)

        if self.tokenization == "flipflop":
            seqs_orig = flipflopfings.flipflop_code(refs, 4)
            indata = cur.astype(np.float32)
            seqs = np.full((self.maxlen,), -1)
        elif self.tokenization == "kmer":
            seqs_orig = refs + 1
            indata = cur
            seqs = np.full((self.maxlen,), 0)
        else:
            seqs_orig = refs + 1
            indata = cur
            seqs = np.full((self.maxlen,), 0)

        seqs[:len(seqs_orig)] = seqs_orig
        seqlen = len(seqs_orig)
        indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT

        return indata.astype(np.float32), seqs, seqlen

    def __len__(self):
        return self.region_np.shape[0]


# ============================================================
# Evaluation
# ============================================================

def model_eval(dataloader, model, is_half, device):
    targets, seqs = [], []
    t0 = time.perf_counter()
    total_samples = 0
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=0.95)

    with torch.no_grad():
        for data, target, *_ in dataloader:
            targets.extend(torch.unbind(target, 0))
            data = data.type(torch.float16).to(device) if is_half else data.to(device)
            total_samples += data.shape[0] * data.shape[2]

            # backbone (CNN + LSTM) → crfencoder (LinearCRFEncoder) → viterbi decode
            x = model.backbone(data)  # 墨芯卡
            scores = model.crfencoder(x)
            seqs.extend(model.decode_batch(scores))

    duration = time.perf_counter() - t0
    refs = [decode_ref(t, model.alphabet) for t in targets]
    accuracies = [accuracy_with_cov(r, s) if len(s) else 0.0 for r, s in zip(refs, seqs)]

    res_mean = np.mean(accuracies)
    res_median = np.median(accuracies)
    res_speed = total_samples / duration
    res_base_speed = np.sum([len(s) for s in seqs]) / duration

    print(f"* mean      {res_mean:.2f}%")
    print(f"* median    {res_median:.2f}%")
    print(f"* time      {duration:.2f}")
    print(f"* samples/s {res_speed:.2E}")
    print(f"* bases/s   {res_base_speed:.2E}")
    print(f"* chunks    {len(refs)}")
    return res_mean, res_median, duration, res_speed, res_base_speed, len(refs)


# ============================================================
# CLI
# ============================================================

def get_parser():
    parser = argparse.ArgumentParser(description='Viterbi Evaluation Script')
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing config.toml, io_quant.pth, act_scales.pth")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data dir (should end with 'train', val is auto-detected)")
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_half", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=25)
    return parser


def main():
    args = get_parser().parse_args()

    MODEL_DIR = args.model_dir
    CONFIG_PATH = os.path.join(MODEL_DIR, "config.toml")
    IO_QUANT_PATH = os.path.join(MODEL_DIR, "layer_9_6x_io_quant.pth")
    ACT_SCALES_PATH = os.path.join(MODEL_DIR, "layer_9_6x_act_scales.pth")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.device.startswith('cuda'):
        device_id = int(args.device.split(':')[1]) if ':' in args.device else 0
        torch.cuda.set_device(device_id)

    assert os.path.exists(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
    assert os.path.exists(IO_QUANT_PATH), f"Weight file not found: {IO_QUANT_PATH}"
    assert os.path.exists(ACT_SCALES_PATH), f"Act scales file not found: {ACT_SCALES_PATH}"

    config = toml.load(CONFIG_PATH)

    print("[Loading quantized model]")
    model = load_quant_model(
        config=config,
        io_quant_path=IO_QUANT_PATH,
        act_scales_path=ACT_SCALES_PATH,
        device=args.device,
        half=args.use_half,
    )
    print(f"Quantized model loaded: {type(model).__name__}")

    print("[Loading validation data]")
    val_dir = args.data_dir
    test_dataset = TrainingDataSet3(val_dir, tokenization="kmer")
    valid_loader = DataLoader(test_dataset, batch_size=args.val_batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    print(f"Validation: {len(test_dataset)} samples, {len(valid_loader)} batches")

    print("=" * 60)
    print(f"Model: {MODEL_DIR}")
    print(f"Weights: {IO_QUANT_PATH}")
    print(f"Act scales: {ACT_SCALES_PATH}")
    print(f"Device: {args.device}, Half: {args.use_half}")
    print("=" * 60)

    res = model_eval(valid_loader, model, args.use_half, args.device)

    print(f"\n{'='*60}")
    print(f"Mean: {res[0]:.2f}%  Median: {res[1]:.2f}%  Time: {res[2]:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
