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
    Compute scores for model.
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


def fmt(stride, attrs):
    return {
        'stride': stride,
        'moves': attrs['moves'].numpy(),
        'qstring': to_str(attrs['qstring']),
        'sequence': to_str(attrs['sequence']),
    }


def basecall(model, reads, chunksize=4000, overlap=100, batchsize=32, model_stride=5, reverse=False, scale=1.0, offset=0.0):
    """
    Basecalls a set of reads.
    """
    chunks = thread_iter(
        ((read, 0, len(read.signal)), chunk(torch.from_numpy(read.signal), chunksize, overlap))
        for read in reads
    )

    batches = thread_iter(batchify(chunks, batchsize=batchsize))

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
