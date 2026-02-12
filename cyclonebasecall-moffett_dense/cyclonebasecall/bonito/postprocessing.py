# -*- coding:utf-8 -*-
from __future__ import annotations

import multiprocessing
import os
import re
import threading
import time
from dataclasses import dataclass
from Bio.SeqIO.QualityIO import FastqGeneralIterator
import itertools

from tqdm import tqdm
import concurrent.futures
import multiprocessing as mp


@dataclass
class Poly:
    number: int = 0
    rate: float = 0.0


class RMPoly:
    @staticmethod
    def remove_header_poly(seq: str, qual: str, st_range: int = 3, poly_limit: int = 2) -> tuple[str, str]:
        seq, qual = str(seq), str(qual)
        if len(seq) != len(qual):
            raise ValueError('seq and qual must have the same length')
        # for st in range(st_range):
        #     poly_ext = len(list(next(itertools.groupby(seq[st:]))[1]))
        #     if poly_ext >= poly_limit:
        #         st += poly_ext
        #         return seq[st:], qual[st:]
        st = 0
        poly_ext = len(list(next(itertools.groupby(seq[st:]))[1]))
        if poly_ext > poly_limit:
            st += poly_ext
        return seq[st:], qual[st:]

    @staticmethod
    def remove_tail_poly(seq: str, qual: str, poly_limit: int = 5) -> tuple[str, str]:
        seq_rev, qual_rev = str(seq)[::-1], str(qual)[::-1]
        if len(seq) != len(qual):
            raise ValueError('seq and qual must have the same length')
        if not seq:
            return seq, qual
        st = 0
        poly_ext = len(list(next(itertools.groupby(seq[::-1]))[1]))
        if poly_ext > poly_limit:
            st = poly_ext
        return seq_rev[st:][::-1], qual_rev[st:][::-1]

    @staticmethod
    def remove_mid_poly(seq: str, qual: str, poly_limit: int = 5):
        seq, qual = str(seq), str(qual)
        if len(seq) != len(qual):
            raise ValueError('seq and qual must have the same length')
        if not seq:
            return seq, qual
        tmp_lst = [0] * (len(seq) * 2)
        tmp_lst[::2] = list(seq)
        tmp_lst[1::2] = list(qual)
        tmp_seq = ''.join(tmp_lst)
        pattern = r"(([AGCT]).)(?:\1){" + str(poly_limit - 1) + r",}"
        tmp_seq = re.sub(pattern, '', tmp_seq)
        new_seq = tmp_seq[::2]
        new_qual = tmp_seq[1::2]
        return new_seq, new_qual


def homopolymer_counting(basecall_str: str, poly_limit: int = 5) -> Poly:
    basecall_str = str(basecall_str)
    poly = Poly()
    sum_length = 0
    for k, g in itertools.groupby(basecall_str):
        length = len(list(g))
        if length >= poly_limit:
            poly.number += 1
            sum_length += length
    poly.rate = sum_length / len(basecall_str)
    return poly


def remove_seq_poly(seq: str, qual: str, poly_limit: int = 5):
    seq, qual = str(seq), str(qual)
    if len(seq) != len(qual):
        raise ValueError('seq and qual must have the same length')
    remove_poly_str = ''
    remove_qual_str = ''
    p = 0
    for k, g in itertools.groupby(seq):
        repeat_str = ''.join(list(g))
        if len(repeat_str) < poly_limit:
            remove_poly_str += repeat_str
            remove_qual_str += qual[p:p + len(repeat_str)]
        p += len(repeat_str)
    return remove_poly_str, remove_qual_str


def remove_fastq_poly(fastq_path: str, out_path: str):
    with open(fastq_path, 'r') as in_handle, open(out_path, 'w') as out_handle:
        for title, seq, qual in tqdm(FastqGeneralIterator(in_handle)):
            if len(seq) <= 5:
                continue
            seq, qual = RMPoly.remove_header_poly(seq, qual, poly_limit=3)
            seq, qual = RMPoly.remove_tail_poly(seq, qual, poly_limit=3)
            # seq, qual = RMPoly.remove_mid_poly(seq, qual, poly_limit=50)
            if not seq:
                continue
            out_handle.write("@%s\n%s\n+\n%s\n" % (title, seq, qual))


if __name__ == "__main__":
    s = 'AAAAAAAAAGGAAAATTTTTTTTTTTTT'
    q = 'BBBBBBBBBIIBBBBGGGGGGGGGGGGG'
    poly = homopolymer_counting(s)
    print(f'poly数量：{poly.number}')
    print(f'poly比例：{poly.rate}')
    # print(remove_seq_poly(s, q))
    s = 'AAAGGGGGT'
    q = 'BBBIIIIIB'
    # '/home/wtbeta_shanzhu/caojie/pycharmproject/cycloneutil'
    # s, q = '', ''
    RMPoly.remove_header_poly(s, q)
    RMPoly.remove_tail_poly(s, q)
    # RMPoly.remove_mid_poly(s, q, poly_limit=5)

    fastq_path = '/store/caojie-data/raw.fastq'
    new_fastq_path = '/store/caojie-data/postproc.fastq'
    n2 = '/mnt/seqdata/basecall_data/bgi_reads/output_tmp/real_data_2/data_for_analysing/cj_rd3.fastq'
    n3 = '/mnt/seqdata/basecall_data/bgi_reads/output_tmp/real_data_2/data_for_analysing/cj_rd4.fastq'
    remove_fastq_poly(fastq_path, new_fastq_path)
