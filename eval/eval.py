#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2024/01/09 09:58:38
@Author  :   Junjie Zhen
@Email   :   zhenjunjie@genomics.cn
'''

import warnings
import sys
import os.path as osp
sys.path.append(osp.dirname(__file__))
from cycloneeval.mapping_and_parsing import get_n_reads_results
warnings.filterwarnings("ignore", category=DeprecationWarning)
fastq_path = "/workspace/huada/scall/1.fastq"
ref_path = "/workspace/huada/scall/ecoli.fasta"
# change_mode(fastq_name)
res = get_n_reads_results(fastq_path, ref_path)
print(res)
