import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(__file__))))
from collections import OrderedDict
import opencall_cli.hdf5_utils as cu # cu stands for cyf_utils
from opencall_cli.hdf5_utils import concat, Event, Cyf
import numpy as np
import torch
from contextlib import contextmanager
# from seqdist import sparse
# from seqdist.core import SequenceDist, Max, Log, semiring
from moffett.MHDConformer.infer import *
import moffett.MHDConformer.infer as infer
from tqdm import tqdm
MHD_PATH = os.path.dirname(infer.__file__)
os.chdir(MHD_PATH)


alphabet_list = ["N", "A", "C", "G", "T"]
state_length = 5


def med_mad_norm(x, dtype='f4'):
    med, mad = med_mad(x)
    normed_x = (x - med) / mad
    return normed_x.astype(dtype)


def transitions_into_base(b_idx, trans_num, device):
    a = torch.arange(0, 1024, dtype=torch.long, device=device)
    b = 5*a + b_idx
    return b


def chunk_read(signal, chunk_size, overlap):
    if len(signal) < chunk_size:
        return signal[:, None, None], np.array([0]), np.array([len(signal)])
    chunk_ends = np.arange(chunk_size, len(
        signal), chunk_size - overlap, dtype=int)
    chunk_ends = np.concatenate([chunk_ends, [len(signal)]], 0)
    chunk_starts = chunk_ends - chunk_size
    nchunks = len(chunk_ends)

    chunks = np.empty((chunk_size, nchunks, 1), dtype='f4')
    for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
        chunks[:, i, 0] = signal[start:end]
    return chunks, chunk_starts, chunk_ends


def stitch_chunks(out, chunk_starts, chunk_ends, stride, log_len, overlap):
    chunk_base_len = log_len
    nchunks = len(chunk_starts)
    stitched_out = []
    if nchunks == 1:
        return out[:, 0]
    elif nchunks == 2:
        overlap_base_len = (chunk_ends[-2] - chunk_starts[-1]) // stride
        half_overlap_base_len = overlap_base_len // 2
        end = chunk_base_len - half_overlap_base_len
        stitched_out.append(out[0:end, 0])
        # print(0, end)
        start = half_overlap_base_len
        stitched_out.append(out[start:chunk_base_len, 1])
        res = np.concatenate(stitched_out, 0)
        return res
    else:
        overlap_base_len = overlap
        half_overlap_base_len = overlap_base_len // 2
        for i in range(nchunks - 2):
            if i == 0:
                stitched_out.append(out[0:-1*half_overlap_base_len, 0])
                # print(i, 0, 950)
            else:
                stitched_out.append(out[half_overlap_base_len:-1*half_overlap_base_len, i])
                # print(i, 50, 950)
        # last two

        end = chunk_base_len - half_overlap_base_len
        stitched_out.append(out[half_overlap_base_len:end, 0])
        # print(50, end)
        start = half_overlap_base_len
        stitched_out.append(out[start:chunk_base_len, 1])
        # print(start, chunk_base_len)
        res = np.concatenate(stitched_out, 0)
        return res


def med_mad(data, factor=None, axis=None, keepdims=False):
    if factor is None:
        factor = 1.4826
    dmed = np.median(data, axis=axis, keepdims=True)
    dmad = factor * np.median(abs(data - dmed), axis=axis, keepdims=True)
    if axis is None:
        dmed = dmed.flatten()[0]
        dmad = dmad.flatten()[0]
    elif not keepdims:
        dmed = dmed.squeeze(axis)
        dmad = dmad.squeeze(axis)
    return dmed, dmad


def stitch(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap
    """
    if chunks.shape[0] == 1: return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

    if reverse:
        chunks = list(chunks)
        return concat([
            chunks[-1][:-start], *(x[-end:-start] for x in reversed(chunks[1:-1])), chunks[0][-first_chunk_end:]
        ])
    else:
        return concat([
            chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]
        ])


def path_to_str(valid_path, alphabet):
    alphabet = np.frombuffer(''.join(alphabet).encode(), dtype='u1')
    seq = alphabet[valid_path]
    return seq.tobytes().decode()


def one_read_basecall(params, basecall_model):
    """
    update qvalue
    :param params:
    :param basecall_model:
    :return:
    """
    signal_origin = params["data"]
    stride = params["stride"]
    chunk_size = params["chunk_size"] * stride
    overlap = params["overlap"] * stride

    normed_signal = med_mad_norm(signal_origin)
    chunks, chunk_starts, chunk_ends = chunk_read(normed_signal, chunk_size, overlap)
    
    chunks = torch.tensor(chunks)
    chunks_ = chunks.permute([1, 2, 0])

    seqs, sequences, errors, moves = basecall_model.inference([chunks_])
    
    chunk_best_paths_np = np.transpose(np.array(sequences), (1, 0))
    path_after_stitch = stitch_chunks(chunk_best_paths_np, chunk_starts, chunk_ends, stride=stride, log_len=params["chunk_size"], overlap=params["overlap"])

    valid_mask = path_after_stitch != 0

    valid_path = path_after_stitch[valid_mask]

    if len(valid_path) > 0:
        basecall_str = path_to_str(valid_path, alphabet=alphabet_list)
        return basecall_str
    else:
        return ""

def get_data(data_name, fast5_path):
    min_read_len = 10010
    max_read_len = 10000000
    adaptor_duration = 5000
    h5 = Cyf(fast5_path, "a")
    print(f"===============Now start processing {fast5_path}===============")
    for read_id in h5.data.keys():
        orig_signal = h5.data[read_id]["Raw"]["Signal"]
        if min_read_len < len(orig_signal) < max_read_len:
            signal = orig_signal[adaptor_duration:-10]
            read_info = "{}_{}_{}".format(data_name, os.path.basename(fast5_path).split(".")[0], read_id)
            params = {
                "stride": 5,
                "chunk_size": 256, # logits, 1280 / 5
                "overlap": 70,  # logits, 350 / 5
                "fastq": True,
                "retry_num": 0,
                "data": signal,
                "read_id": read_info
            }
            yield params


if __name__ == '__main__':
    test_data_name = "Whatever_you_like(space_not_allowed)"
    fast5_dir = "/workspace/OpenCall/data/fast5/bacillus"
    fastq_path = "/workspace/OpenCall/data/results/bacillus.fastq"
    if os.path.exists(fastq_path):
        os.remove(fastq_path)

    yaml_path = os.path.join (MHD_PATH, "config.yaml")
    conformer_lib_path = MHD_PATH + "/../install/ubuntu18.04-gcc7.5.0-x86_64/lib/libmfmodeltesterv2.so"
    model = MHDConformerMultiprocessRuntime(
        mfmodeltesterv2_lib_path=conformer_lib_path, model_path=yaml_path, device_id=0, num_of_subprocess=32)

    for file in os.listdir(fast5_dir):
        basecall_res = list()
        data_all = [data for data in get_data(test_data_name, os.path.join(fast5_dir,file))]
        data_num = len(data_all)
        with tqdm(total=data_num) as pbar:
            for num, params_data in enumerate(data_all):
                out = one_read_basecall(params_data, model)
                if len(out[0]) > 0:
                    basecall_res.append((out, "?"*len(out), params_data["read_id"]))
                    # print(num, out[0])
                    # print(num, out[1])
                pbar.update(1)

        #  save fastq
        f = open(fastq_path, "a")
        for basecall_str, qstring, read_id in basecall_res:
            f.write("{}{}\n{}\n".format("@", read_id, basecall_str))
            f.write("+\n{}\n".format(qstring))
        f.close()
    del model



