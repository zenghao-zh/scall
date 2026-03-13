# -*- coding:utf-8 -*-
import collections
import re
from typing import List

import mappy as mp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from camstat import cam
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
# from cytoolz import pluck
from collections import deque


class Cigar:
    """ A parser of CIGAR string.

        Learn/Copy from: https://gitlab.genomics.cn/cyclone/qvalue_statis/-/blob/master/file_base.py::CIGAR
    """

    def __init__(self, cigar_string: str, ref_read: str, query_read: str):
        if cigar_string is None:
            cigar_string = ""

        if query_read is None:
            query_read = ""

        self.cigar = cigar_string
        self.ref_read = ref_read
        self.query_read = query_read

        self.ref_operations = ["M", "D", "N", "=", "X"]
        self.query_operations = ["M", "I", "S", "=", "X"]
        self.pattern = re.compile("([0-9]+[a-zA-Z=])")
        self.compare_seq = {}
        self.cigar_ext = ""
        self.ref_cigar = ""
        self.query_cigar = ""

        self.ref_match_indices = []  # np.array, the same below
        self.query_match_indices = []

        self.ref_mismatch_indices = []
        self.query_mismatch_indices = []

        self.query_insert_indices = []
        self.ref_delete_indices = []

        self.query_clip_indices = []

        self.stats = dict()  # dict

    def __call__(self):
        self.parse()
        self.match()
        self.index()
        self.count()

    def parse(self):
        elems = self.pattern.findall(self.cigar)
        for elem in elems:
            times = int(elem[:-1])
            op = elem[-1]
            if op in self.query_operations:
                self.query_cigar += op * times
            if op in self.ref_operations:
                self.ref_cigar += op * times
            self.cigar_ext += op * times

        if self.cigar is not None and self.cigar != "":
            assert len(self.ref_cigar) == len(self.ref_read)
            assert len(self.query_cigar) == len(self.query_read)
            self.compare_seq["ref_cigar"] = self.ref_cigar
            self.compare_seq["qry_cigar"] = self.query_cigar
            self.compare_seq['ref_seq'] = self.ref_read
            self.compare_seq['qry_seq'] = self.query_read

    def match(self):
        ref_indices = self._find_indices(string=self.ref_cigar, operations=["M"])
        qry_indices = self._find_indices(string=self.query_cigar, operations=["M"])
        ref_cigars = list(self.ref_cigar)
        query_cigars = list(self.query_cigar)
        assert len(ref_indices) == len(qry_indices)
        for i in range(len(ref_indices)):
            r = ref_indices[i]
            q = qry_indices[i]
            if self.ref_read[r] == self.query_read[q]:
                ref_cigars[r] = query_cigars[q] = "="
            else:
                ref_cigars[r] = query_cigars[q] = "X"
        self.ref_cigar = "".join(ref_cigars)
        self.query_cigar = "".join(query_cigars)

    def index(self):
        self.ref_match_indices = self._find_indices(string=self.ref_cigar, operations=["="])
        self.query_match_indices = self._find_indices(string=self.query_cigar, operations=["="])

        self.ref_mismatch_indices = self._find_indices(string=self.ref_cigar, operations=["X"])
        self.query_mismatch_indices = self._find_indices(string=self.query_cigar, operations=["X"])

        self.ref_delete_indices = self._find_indices(string=self.ref_cigar, operations=["D"])
        self.query_insert_indices = self._find_indices(string=self.query_cigar, operations=["I"])

        self.query_clip_indices = self._find_indices(string=self.cigar_ext, operations=["H", "P", "S"])

    def _find_indices(self, string: str, operations: List[str]):
        indices = np.array([i for i, s in enumerate(string) if s in operations])
        return indices

    def count(self):
        if len(self.query_mismatch_indices) == 0:
            total_read_base = len(self.query_read)
        else:
            total_read_base = len(self.query_insert_indices) + \
                              len(self.query_match_indices) + \
                              len(self.query_mismatch_indices)
        self.stats = dict(
            total_ref_base=len(self.ref_read),
            # total_read_base=len(self.query_read)-len(self.query_clip_indices),
            total_read_base=total_read_base,
            total_clipped=len(self.query_clip_indices),
            total_base_del=len(self.ref_delete_indices),
            total_base_ins=len(self.query_insert_indices),
            total_match=len(self.ref_match_indices),
            total_mismatch=len(self.ref_mismatch_indices)
        )


def single_basecall_str_mapping(seq, ref_path):
    ref = mp.Aligner(ref_path, best_n=1, preset='map-ont')
    sam_dict = {}
    for hit in ref.map(str(seq.upper())):  # traverse alignments
        sam_dict['ref_read'] = ref.seq(hit.ctg, hit.r_st, hit.r_en)
        sam_dict['query_read'] = seq if hit.strand == 1 else str(Seq(seq).reverse_complement())
        if hit.strand == 1:
            soft_pre = hit.q_st
            cigar_string = ''
            if soft_pre > 0:
                cigar_string += f'{soft_pre}S'
            cigar_string += hit.cigar_str
            soft_after = len(seq) - len(seq[hit.q_st:hit.q_en]) - soft_pre
            if soft_after > 0:
                cigar_string += f'{soft_after}S'
        else:
            soft_after = hit.q_st
            cigar_string = hit.cigar_str
            if soft_after > 0:
                cigar_string += f'{soft_after}S'
            soft_pre = len(seq) - len(seq[hit.q_st:hit.q_en]) - soft_after
            if soft_pre>0:
                cigar_string = f'{soft_pre}S' + cigar_string
        sam_dict['cigar_string'] = cigar_string
        return sam_dict

def single_basecall_str_mapping2(seq, ref_path, compare_txt):
    ref = mp.Aligner(ref_path, best_n=1, preset='map-ont')
    sam_dict = {}
    for hit in ref.map(str(seq.upper())):  # traverse alignments
        tmpfile = open(f'./data/{compare_txt}', 'a')
        print(hit.r_st, hit.r_en, file=tmpfile)
        tmpfile.close()
        sam_dict['ref_read'] = ref.seq(hit.ctg, hit.r_st, hit.r_en)
        sam_dict['query_read'] = seq
        soft_pre = hit.q_st
        cigar_string = ''
        if soft_pre > 0:
            cigar_string += f'{soft_pre}S'
        cigar_string += hit.cigar_str
        soft_after = len(seq) - len(seq[hit.q_st:hit.q_en]) - soft_pre
        if soft_after > 0:
            cigar_string += f'{soft_after}S'
        sam_dict['cigar_string'] = cigar_string
        return sam_dict


def parsing_mapping_res(query_read, ref_read, cigar_string):
    cigar = Cigar(
        cigar_string=cigar_string,
        ref_read=str(ref_read.upper()),
        query_read=str(query_read.upper())
    )
    cigar()
    return cigar.stats, cigar.compare_seq


def cigar_analysing(ref_cigar, qry_cigar, ref_seq, qry_seq):
    ref_cigar_dq = deque(ref_cigar)
    qry_cigar_dq = deque(qry_cigar)
    ref_seq = deque(ref_seq)
    qry_seq = deque(qry_seq)
    while qry_cigar_dq and qry_cigar_dq[0] == 'S':
        qry_seq.popleft()
        qry_cigar_dq.popleft()
    res_ref = []
    res_qry = []
    while ref_cigar_dq and qry_cigar_dq:
        while qry_cigar_dq and qry_cigar_dq[0] == 'I':
            res_qry.append(qry_seq.popleft())
            qry_cigar_dq.popleft()
            res_ref.append('-')
        while ref_cigar_dq and ref_cigar_dq[0] in ('D', 'N'):
            res_ref.append(ref_seq.popleft())
            ref_cigar_dq.popleft()
            res_qry.append('-')
        if ref_cigar_dq and qry_cigar_dq:
            res_qry.append(qry_seq.popleft())
            res_ref.append(ref_seq.popleft())
            qry_cigar_dq.popleft()
            ref_cigar_dq.popleft()
    ref_res_seq = ''.join(res_ref)
    qry_res_seq = ''.join(res_qry)
    return ref_res_seq, qry_res_seq
    # with open('./data/cigar_res.txt', 'w') as f:
    #     f.write(f'{ref_res_seq}\n{qry_res_seq}')


def cigar_analysing(ref_cigar, qry_cigar, ref_seq, qry_seq):
    ref_cigar_dq = deque(ref_cigar)
    qry_cigar_dq = deque(qry_cigar)
    ref_seq = deque(ref_seq)
    qry_seq = deque(qry_seq)
    while qry_cigar_dq and qry_cigar_dq[0] == 'S':
        qry_seq.popleft()
        qry_cigar_dq.popleft()
    res_ref = []
    res_qry = []
    while ref_cigar_dq and qry_cigar_dq:
        while qry_cigar_dq and qry_cigar_dq[0] == 'I':
            res_qry.append(qry_seq.popleft())
            qry_cigar_dq.popleft()
            res_ref.append('-')
        while ref_cigar_dq and ref_cigar_dq[0] in ('D', 'N'):
            res_ref.append(ref_seq.popleft())
            ref_cigar_dq.popleft()
            res_qry.append('-')
        if ref_cigar_dq and qry_cigar_dq:
            res_qry.append(qry_seq.popleft())
            res_ref.append(ref_seq.popleft())
            qry_cigar_dq.popleft()
            ref_cigar_dq.popleft()
    ref_res_seq = ''.join(res_ref)
    qry_res_seq = ''.join(res_qry)
    return ref_res_seq, qry_res_seq


def diff_abundance_plot(fasta_path, tsv_id_path, tsv_abundance_path, real_taxonomy, out_path):
    df_id = pd.read_csv(tsv_id_path, sep='\t')
    df_abundance = pd.read_csv(tsv_abundance_path, sep='\t', header=None)
    df_abundance.columns = df_abundance.iloc[-1, :]
    df_abundance.drop([len(df_abundance) - 1], inplace=True)
    genus_dict = collections.defaultdict(list)
    id_map = collections.defaultdict(lambda: '')
    for real_type in real_taxonomy:
        genus = real_type.split()[0]
        df_abundance_sub = df_abundance[df_abundance['genus'] == genus]
        ser_main = df_id[df_id['species'] == real_type]
        id_main, species_main = ser_main['tax_id'].values[0], ser_main['species'].values[0]
        abundance_main = df_abundance_sub[df_abundance_sub['species'] == real_type]['abundance'].values[0]
        id_map[str(id_main)]
        for i in range(len(df_abundance_sub)):
            ser = df_abundance_sub.iloc[i]
            species_other = ser['species']
            if species_other == species_main:
                continue
            abundance_other = ser['abundance']
            id_other = df_id[df_id['species'] == species_other]['tax_id'].values[0]
            genus_dict[(id_main, abundance_main, species_main)].append((id_other, abundance_other, species_other))
            id_map[str(id_other)]
    # for read in SeqIO.parse(fasta_path, 'fasta'):
    #     # c=mp.Aligner.map('AGCT','AGLError  ')
    #     print(read.id)
    #     print(read.seq)
    #     print(read.description)
    for read in mp.fastx_read(fasta_path, read_comment=True):
        read_id = read[0].split(':')[0]
        if read_id in set(id_map.keys()):
            id_map[read_id] = read[0]
    reads_IO = SeqIO.index(fasta_path, 'fasta')
    lst = []
    cnt = 0
    sum = 0
    length = []
    res = {}
    for id_main, abundance_main, species_main in genus_dict:
        id_main_integrity = id_map[str(id_main)]
        seq_detail = reads_IO[id_main_integrity]
        seq_record = SeqRecord(seq_detail.seq, id=id_main_integrity, description=seq_detail.description)
        length.append(len(seq_detail.seq))
        SeqIO.write(seq_record, "./data/my_ref.fasta", "fasta")
        if len(genus_dict[id_main, abundance_main, species_main]) <= 0:
            continue
        for id_other, abundance_other, species_other in genus_dict[id_main, abundance_main, species_main]:
            sum += 1
            id_other_integrity = id_map[str(id_other)]
            seq_other = reads_IO[id_other_integrity].seq
            sam_status = single_basecall_str_mapping(seq_other, "./data/my_ref.fasta")
            # alignment = pw2.align.globalxx(seq_detail.seq.upper(), seq_other.upper())[0]
            # diff = 1 - alignment.score / (alignment.end - alignment.start)
            abundance_fold = float(abundance_main) / float(abundance_other)
            # lst.append((diff, abundance_fold))
            if not sam_status:
                continue
            status = parsing_mapping_res(sam_status['query_read'], sam_status['ref_read'], sam_status['cigar_string'])
            diff2 = 1 - status['total_match'] / status['total_ref_base']
            lst.append((diff2, abundance_fold))
            res[id_main, id_other] = [df_id[df_id['tax_id'] == id_main].genus.values[0],
                                      df_id[df_id['tax_id'] == id_main].species.values[0],
                                      df_id[df_id['tax_id'] == id_other].species.values[0],
                                      diff2, abundance_fold]
            # except:
            #     cnt += 1
            #     single_basecall_str_mapping(seq_other, "./data/my_ref.fasta")
            #     continue
    # print(cnt)
    # print(length)
    # print(sum)
    df = pd.DataFrame(res, columns=['genus', 'main_species', 'other_species', 'diff', 'abundance_fold'])
    df.to_csv('./data/diff_abundance.csv', sep='\t', index=True)
    lst.sort()
    x = [*pluck(0, lst)]
    y = [*pluck(1, lst)]
    plt.close()
    plt.plot(x, y)
    plt.xlabel('Difference of genome')
    plt.ylabel('Abundance fold')
    plt.show()


if __name__ == "__main__":
    # fasta_path = './data/ysy106_1_consensus.fasta'
    # ref_path = './data/synVII_yeast_marker_BY4741.idx'
    # for read in SeqIO.parse(fasta_path, 'fasta'):
    #     print(read.seq[:10])
    #     a = single_basecall_str_mapping(read.seq, ref_path)
    #     if a:
    #         parsing_mapping_res(**a)
    fasta_path = '/home/syh/emu/database_emu/species_taxid.fasta'
    real_taxonomy = ['Limosilactobacillus fermentum', 'Bacillus subtilis', 'Staphylococcus aureus',
                     'Listeria monocytogenes', 'Salmonella enterica', 'Escherichia coli', 'Enterococcus faecalis',
                     'Pseudomonas aeruginosa']
    out_path = '/home/caojie/tmp/data/boinfo/syh_t1/'
    tsv_id_path = '/home/syh/emu/database_emu//taxonomy.tsv'
    tsv_abundance_path = '/home/syh/emu/output_emu_119/119_rel-abundance.sort.tsv'
    diff_abundance_plot(fasta_path, tsv_id_path, tsv_abundance_path, real_taxonomy, out_path)
