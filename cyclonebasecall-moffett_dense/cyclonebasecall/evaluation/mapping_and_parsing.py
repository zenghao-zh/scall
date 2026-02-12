# -*- coding:utf-8 -*-
from .mapping_str import single_basecall_str_mapping, parsing_mapping_res, cigar_analysing
from typing import Dict, List, Tuple
import requests
import pysam
import warnings
import numpy as np
import pandas as pd
import json
import sys
import os

# import pyximport
# pyximport.install(setup_args={"include_dirs":np.get_include()},
#                   reload_support=True)

from cyclonebasecall.evaluation.mapping_optim_v1 import exceptions
from cyclonebasecall.evaluation.mapping_optim_v1.cam_v1 import Cam


# from mapping_optim_v1.cam import Cam

warnings.filterwarnings("ignore", category=Warning)


def identity_rate_modal(df):
    df_mapped = df[["identity_rate"]]
    # df_mapped = self.df_detail[["identity_rate"]]
    a = df_mapped["identity_rate"]
    try:
        # n_bins = int(round(np.max(a)-0.6,3)/0.005)
        # counts, edges = np.histogram(a=a,range=(0.6,np.max(a)), bins=n_bins)
        counts, edges = np.histogram(a=a, range=(0.6, 1.0), bins=80)
    except:
        n_bins = 100
        counts, edges = np.histogram(a=a, range=(0, 1), bins=n_bins)
    list_a = counts.tolist()
    modal_index_in_counts = list_a.index(max(list_a))  # 返回最大值的索引
    modal_identity_rate = (
        edges[modal_index_in_counts] + edges[modal_index_in_counts + 1]) / 2
    return modal_identity_rate

# sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def get_n_reads_results(fastq_path: str, ref_path: str, mapping_tool: str = "minimap2") -> Dict:
    '''
    输入fastq文件路径，返回测序的评估结果, 并打印在屏幕上
    :param fastq_path: fastq文件路径
    :param ref_path: 文库的文件路径
    :param mapping_tool: mapping工具
    :return: summary字典
    '''
    cam = Cam(ref_file=ref_path, query_file=fastq_path,
              mapping_tool=mapping_tool)
    cam.mapping()
    # columns = ['folder_name', 'channel_num', 'start_time', 'end_time', 'start_index',
    #      'end_index', 'ref_name', 'ref_start_pos', 'query_name', 'query_length',
    #      'aligned_length', 'is_unmapped', 'flag', 'mapq', 'coverage', 'total_ref_base',
    #      'total_read_base', 'total_clipped', 'total_base_del', 'total_base_ins',
    #      'total_match', 'total_mismatch', 'identity_rate', 'mismatch_rate',
    #      'insertion_rate', 'deletion_rate']
    df_detail = cam.df_detail
    # save_path = f'./data/{fastq_path.split(".")[0]}/mapping_detail.csv'
    # os.makedirs(save_path, exist_ok=True)
    # cam.df_summary.to_csv(save_path, index=False)
    df_summary = cam.df_summary
    res = dict(zip(list(df_summary.columns), list(df_summary.values[0])))
    Modal_Identity_Rate = identity_rate_modal(df_detail)
    res["Modal_Identity_Rate"] = Modal_Identity_Rate
    res["Median_Identity_Rate"] = df_detail['identity_rate'].median()
    return res


def basecall(signal: List, signal_id: str, model_name: str) -> Tuple:
    """
    碱基判读，根据电信号序列返回对应的碱基序列
    :param signal:列表结构的电信号序列
    :param signal_id:序列名称
    :param model_name: 模型名称
    :return:返回碱基序列以及相应的q值字符串,如果根据电信号无法获取basecall结果，返回(None,None)
    """
    url = "http://192.168.0.98:10080/predict"
    params = {
        "chunk_size": 1000,
        "overlap": 100,
        "stride": 5,
        "max_concurrent_chunks": 128,
        "qscore_scale": 1.0,
        "qscore_offset": 0.,
        "data": signal,
        "read_id": signal_id,
        "basecall_model": model_name,
    }
    r = requests.post(url, json=params)

    if r.status_code == 200:
        response = json.loads(r.content)
        basecall_res, qstring_res = response[0], response[1]
        return basecall_res, qstring_res
    return None, None


def get_one_reads_results(query_seq: str, ref_path: str) -> Dict:
    """
    将query序列比对到参考基因组相应位置
    :param query_seq: 需要比对的序列
    :param ref_name: 参考基因组名称
    :return: 结果字典
        status: 2,代表无法mapping到基因组上，一般是因为匹配度太低了；3，代表mapping成功
        statistics: 匹配的各种指标
        reference: 参考序列，-代表插入
        query: 比对的序列，-代表插入
        desc：结果状态描述
    """
    idx_path = os.path.splitext(ref_path)[0] + ".idx"
    if not os.path.exists(idx_path):
        generate_idx = f"/usr/local/minimap2/minimap2 -d {idx_path} {ref_path}"
        os.system(generate_idx)
    sam_status = single_basecall_str_mapping(query_seq, idx_path)
    if not sam_status:
        return {"status": 2, 'statistics': {}, 'reference': '', 'query': '', 'desc': 'mapping with no result'}
    if sam_status:
        print('right')
    statistics, compare_seq = parsing_mapping_res(**sam_status)
    total_match = statistics['total_match']
    total_mismatch = statistics['total_mismatch']
    total_base_ins = statistics['total_base_ins']
    total_ref_base = statistics['total_ref_base']
    total_base_del = statistics['total_base_del']
    mismatch_rate = total_mismatch / (total_ref_base + total_base_ins)
    insertion_rate = total_base_ins / (total_ref_base + total_base_ins)
    identity_rate = total_match / (total_ref_base + total_base_ins)
    deletion_rate = total_base_del / (total_ref_base + total_base_ins)
    statistics['mismatch_rate'] = mismatch_rate
    statistics['insertion_rate'] = insertion_rate
    statistics['identity_rate'] = identity_rate
    statistics['deletion_rate'] = deletion_rate
    ref_res_seq, qry_res_seq = cigar_analysing(**compare_seq)
    res = {"status": 3, 'statistics': statistics,
           'reference': ref_res_seq, 'query': qry_res_seq, 'desc': 'success'}
    return res


if __name__ == "__main__":
    fastq_files = [
        "/workspace/OpenCall/opencall/evaluation/data/test.fastq",
        '/workspace/OpenCall/opencall/evaluation/data/20220727172915_LAB256V1_5K_PC28_10_B0_HD53_J4_0_AD1_Ecoli_gTube_Zhengrongrong_Mux.fastq'
        # '/mnt/seqdata/output_data/20220127155506_LAB256V2_5K_PC28_28_B16_H49-5c20-J4-F_AD3_Ecoli_gTube_HuangPing_Mux/data_for_analysing/20220127155506_LAB256V2_5K_PC28_28_B16_H49-5c20-J4-F_AD3_Ecoli_gTube_HuangPing_Mux.fastq'
    ]
    ref = "/workspace/OpenCall/opencall/evaluation/data/ecoli.fasta"
    res=get_n_reads_results(fastq_files[1], ref, 'minimap2')
    print(res)
    # arr = np.load(
    #     '/workspace/OpenCall/opencall/evaluation/data//test.npy', allow_pickle=True)
    # seq, qstring = basecall(arr.tolist(), 'test', '256_1g')
    # print(f'seq:{seq}')
    # print(f'qstring:{qstring}')
    # res = get_one_reads_results(seq, ref)
    # print(res)
    # arr = np.array(pd.read_csv('./data/test1.csv')['current(pA)']).tolist()
    # seq, _ = basecall(arr, 'test1', '256_1g')
    # get_one_reads_results(seq, 'ecoli')
    # arr = np.load('./data/test.npy', allow_pickle=True)
    # s,q=basecall(arr.tolist(),'test','256_1g')
    # get_one_reads_results(s,'ecoli')
    # fastq_name = '20220310054227_WTseqV1_5K_PC28_30_Z3_HD53_J4_A_WT04_AD3_Bacillus_subtilis_MoChenjie'
    # fastq_path = f'/mnt/seqdata/output_data/{fastq_name}/data_for_analysing/{fastq_name}.fastq'
    # ref_path = f'/mnt/seqdata/data/refs/Bacillus_subtilis.fasta'
    # change_mode(fastq_name)
    # get_n_reads_results(fastq_path, ref_path)
    # from Bio.SeqRecord import SeqRecord
    # from Bio.Seq import Seq
    # from Bio import SeqIO

    # i = 0
    # for aligned_read in pysam.Samfile('./data/test.sam'):
    #     if i == 0:
    #         i += 1
    #         continue
    #     # print(aligned_read)
    #     # seq = Seq(aligned_read.query_sequence)[:-118010]
    #     seq = aligned_read.query_sequence
    #     # first = SeqRecord(seq, id=aligned_read.query_name + '_1', description='')
    #     # SeqIO.write(first, f"./data/m1.fasta", "fasta")
    #     sam_status1 = single_basecall_str_mapping(seq, './data/m1.fasta')
    #     status1, _ = parsing_mapping_res(sam_status1['query_read'], sam_status1['ref_read'],
    #                                      sam_status1['cigar_string'])
    #     ...
