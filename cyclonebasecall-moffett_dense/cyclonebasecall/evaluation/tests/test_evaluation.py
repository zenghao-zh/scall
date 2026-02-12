# -*- coding:utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import unittest
from mapping_and_parsing import get_n_reads_results, basecall, get_one_reads_results
import numpy as np
import warnings
# warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestEvalutation(unittest.TestCase):

    def test_get_n_reads_results(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        fastq_path = "/workspace/OpenCall/data/test.fastq"
        ref_path = "/workspace/OpenCall/data/ecoli.fasta"
        #这里的文库为Bacillus_subtilis，使用绝对路径
        # change_mode(fastq_name)
        res = get_n_reads_results(fastq_path, ref_path)
        print(res)

    def test_basecall(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        arr = np.load( '/workspace/OpenCall/data/test.npy', allow_pickle=True)
        seq, qstring = basecall(arr.tolist(), 'test', '256_1g')
        print(f'seq:{seq}')
        print(f'qstring:{qstring}')

    def test_get_one_reads_results(self):
        # warnings.filterwarnings("ignore", category=DeprecationWarning)
        arr = np.load( '/workspace/OpenCall/data/test.npy', allow_pickle=True)
        seq, qstring = basecall(arr.tolist(), 'test', '256_1g')
        # 这里的文库为ecoli，使用名称即可
        ref_path = "/workspace/OpenCall/data/ecoli.fasta"
        res = get_one_reads_results(seq, ref_path)
        print(res)
if __name__ == '__main__':
    unittest.main()