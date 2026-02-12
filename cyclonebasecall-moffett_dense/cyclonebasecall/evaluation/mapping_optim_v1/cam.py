# -*- coding: utf-8 -*-
import asyncio
import multiprocessing
from typing import List
import array
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, as_completed, ALL_COMPLETED
from Bio import SeqIO
import h5py
import numpy as np
import pandas as pd
import pysam
import requests
from exceptions import (CamError, PathNotFoundError,
                        InvalidMappingToolError, CamFileAccessModeError)
import json
from multiprocessing import Manager, Pool, cpu_count

CPU_COUNT = cpu_count()

ref_seqs = None
li = []


def f5(query_read):
    if query_read.reference_name is None:
        assert query_read.flag == 4
        ref_seq = ""
    else:
        ref_seq = ""
    aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
    if aligned_read.flag in (0, 4, 16):
        return aligned_read.to_dict()


def f6(query_read):
    if query_read.reference_name is None:
        assert query_read.flag == 4
        ref_seq = ""
    else:
        ref_seq = ref_seqs[query_read.reference_name].seq
    aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
    if aligned_read.flag in (0, 4, 16):
        li.append(aligned_read.to_dict())


class Cam:
    """ Cyclone Alignment/Mapping. """

    def __init__(self, ref_file: str, query_file: str, mapping_tool: str = "minimap2"):
        if not os.path.isfile(ref_file):
            raise PathNotFoundError(
                f"Reference sequence file {ref_file} not found")

        if not os.path.isfile(query_file):
            raise PathNotFoundError(
                f"Query sequence file {query_file} not found")

        self.ref_file = ref_file
        self.query_file = query_file
        self.mapping_tool = mapping_tool

        self.df_detail = None
        self.df_summary = None

    def __call__(self):
        try:
            self.mapping()
            # self.store()
        except Exception as e:
            err_msg = e.args[0]
            raise CamError(err_msg)

    def mapping(self):
        tmp_root_path = os.path.dirname(self.query_file)
        self.folder_name = os.path.splitext(
            os.path.basename(self.query_file))[0]
        # tmp_folder = f'{tmp_root_path}/{self.folder_name}/data_for_analysing/'
        tmp_folder = tmp_root_path
        os.makedirs(tmp_folder, exist_ok=True)

        ref_idx = os.path.splitext(self.ref_file)[0] + ".idx"
        mapping = Mapping(
            ref_file=self.ref_file,
            index_file=ref_idx,
            query_file=self.query_file,
            sam_file=f'{tmp_folder}/{self.folder_name}.sam',
            bam_file=f'{tmp_folder}/{self.folder_name}.bam',
            sorted_bam_file=f'{tmp_folder}/{self.folder_name}.sorted.bam',
            bam_file_index=f'{tmp_folder}/{self.folder_name}.sorted.bam.bai'
        )

        mapping(mapping_tool=self.mapping_tool)
        self.index_file = mapping.idx_file
        self.sam_file = mapping.sam_file

        sam = Sam(sam_file=self.sam_file, ref_file=self.ref_file, query_file=self.query_file,
                  mapping_tool=self.mapping_tool)

        self.df_detail = sam.df_detail
        self.df_summary = sam.df_summary
        # self.ref_qvalues_matrix = sam.ref_qvalues_matrix
        self.query_q40 = sam.query_q40

    def store(self):
        cam_file_path = os.path.splitext(self.query_file)[0] + ".cam"
        cam_file_proxy = CamFile(cam_file_path=cam_file_path, mode="w")
        cam_file_proxy.write(
            df_summary=self.df_summary,
            df_detail=self.df_detail,
            ref_qvalues_matrix=self.ref_qvalues_matrix,
            query_q40_match=np.array(self.query_q40["query_q40_match"]),
            query_q40_mismatch=np.array(self.query_q40["query_q40_mismatch"]),
            query_q40_ins=np.array(self.query_q40["query_q40_ins"]),
        )


class Mapping:

    def __init__(self, ref_file: str, query_file: str, index_file: str = None,
                 sam_file: str = None, bam_file: str = None, sorted_bam_file: str = None, bam_file_index: str = None):
        if not os.path.isfile(ref_file) or not os.path.isfile(query_file):
            raise FileNotFoundError(
                f"Reference file {ref_file} or query file {query_file} not found")

        self.default_ref_name = os.path.splitext(os.path.basename(ref_file))[0]
        self.default_query_name = os.path.splitext(
            os.path.basename(query_file))[0]
        self.ref_file = ref_file
        self.query_file = query_file

        if not index_file or not os.path.isfile(index_file):
            self.idx_file = os.path.join(os.path.dirname(
                self.ref_file), self.default_ref_name + ".idx")
        else:
            self.idx_file = index_file

        if sam_file is None:
            self.sam_file = os.path.join(os.path.dirname(
                self.query_file), self.default_query_name + ".sam")
        else:
            self.sam_file = sam_file

        if bam_file is None:
            self.bam_file = os.path.join(os.path.dirname(
                self.query_file), self.default_query_name + ".bam")
        else:
            self.bam_file = bam_file

        if sorted_bam_file is None:
            self.sorted_bam_file = os.path.join(os.path.dirname(self.query_file),
                                                self.default_query_name + ".sorted.bam")
        else:
            self.sorted_bam_file = sorted_bam_file

        if bam_file_index is None:
            self.bam_file_index = os.path.join(os.path.dirname(self.query_file),
                                               self.default_query_name + ".sorted.bam.bai")
        else:
            self.bam_file_index = bam_file_index

    def __call__(self, mapping_tool: str = "minimap2"):
        self.index()
        self.align(mapping_tool=mapping_tool)

    def index(self):
        if not os.path.isfile(self.idx_file):
            cmd = f"/workspace/minimap2-2.24_x64-linux/minimap2 -d {self.idx_file} {self.ref_file}"
            po = subprocess.Popen(
                cmd.split(), bufsize=-1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            po.wait()

    def align(self, mapping_tool: str = "minimap2"):
        # self.index()
        # cmd = "/workspace/minimap2-2.24_x64-linux/minimap2 -ax map-ont --secondary=no {idx_file} {qry_file} -o {sam_file}".format(
        if mapping_tool == "minimap2":
            cmd = f"/workspace/minimap2-2.24_x64-linux/minimap2 -ax map-ont --secondary=no {self.idx_file} {self.query_file} -o {self.sam_file} >/dev/null"
        elif mapping_tool == "graphmap":
            cmd = f"/usr/local/graphmap/bin/Linux-x64/graphmap align --error-rate 0.99 --preset sensitive --mapq 0.9 --max-error 0.99 --evalue 1e100 -r {self.ref_file} -d {self.query_file} -o {self.sam_file}"
        else:
            err_msg = f"Unknown mapping tool {mapping_tool}"
            raise ValueError(err_msg)
        po = subprocess.Popen(
            cmd.split(), bufsize=-1, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        po.wait()


class Sam:
    """ A class represents the SAM/BAM file. """

    def __init__(self, sam_file: str, ref_file: str, query_file: str, mapping_tool: str = None,
                 mode: str = "rb", check_header: bool = True, check_sq: bool = True,
                 index_file: str = None, require_index: bool = True, thread_num: int = 1,
                 **kwargs):
        if not os.path.isfile(ref_file) or not os.path.isfile(query_file):
            raise FileNotFoundError(
                "Reference file '%s' or query file '%s' not found" % (ref_file, query_file))

        if not os.path.isfile(sam_file):
            raise FileNotFoundError("Bam file '%s' not found" % (sam_file))

        self._df_detail = None
        self._df_summary = None
        # a 2d numpy array, it's row number is 5, and the column number is the length of reference sequence
        self._ref_qvalues = None
        self._query_q40 = dict(q40_match=[], q40_mismatch=[], q40_ins=[])

        self.ref_file = ref_file
        self.query_file = query_file
        self.ref_seqs = dict()
        self.mapping_tool = mapping_tool

        if index_file is None or not os.path.isfile(index_file):
            index_file = os.path.splitext(sam_file)[0] + ".sorted.bam.bai"
        # self.sam_obj = pysam.AlignmentFile(
        self.sam_obj = pysam.Samfile(
            filepath_or_object=sam_file,
            mode=mode,
            check_header=check_header,
            check_sq=check_sq,
            reference_filename=ref_file,
            index_filename=index_file,
            require_index=require_index,
            threads=thread_num,
            **kwargs
        )
        self.aligned_reads = []
        self.aligned_stat = dict()

        self.ref_counter = dict()  # key is reference name, values are numpy matrix
        self.open_ref_file()
        global ref_seqs
        ref_seqs = self.ref_seqs
        self.query_counter = dict(
            query_match_qvalues=[],
            query_mismatch_qvalues=[],
            query_ins_qvalues=[]
        )

    def open_ref_file(self):
        fd = open(self.ref_file, "r")
        format = os.path.splitext(self.ref_file)[-1].lstrip(".")
        fast_file = SeqIO.parse(fd, format=format)
        for ref_seq in fast_file:
            # self.ref_seq_lens[ref_seq.id] = len(ref_seq)
            self.ref_seqs[ref_seq.id] = ref_seq
            # ref_len = len(ref_seq)
            # self.ref_counter[ref_seq.id] = np.zeros((5, ref_len))
        fd.close()

    @property
    def reads_count(self):
        count = 0
        f = open(self.query_file, "r")
        format = os.path.splitext(self.query_file)[-1].lstrip(".")
        if format.lower() == "fq":
            format = "fastq"
        for read in SeqIO.parse(f, format=format):
            count += 1
        return count

    @property
    def df_detail(self):
        if self._df_detail is None:
            start = time.time()
            self.count()
            print('self.count,Time-consuming:', time.time() - start)
            # self.aligned_reads= self.li
            self._df_detail = pd.DataFrame(data=self.aligned_reads)
            self._df_detail.eval(
                "identity_rate = total_match / (total_ref_base + total_base_ins)", inplace=True)
            self._df_detail.eval(
                "mismatch_rate = total_mismatch / (total_ref_base + total_base_ins)", inplace=True)
            self._df_detail.eval(
                "insertion_rate = total_base_ins / (total_ref_base + total_base_ins)", inplace=True)
            self._df_detail.eval(
                "deletion_rate = total_base_del / (total_ref_base + total_base_ins)", inplace=True)
        return self._df_detail

    @property
    def df_summary(self):
        if self._df_summary is None:
            self.collect_query_qvalues()
            self.summary()
            self._df_summary = pd.DataFrame(data=[self.aligned_stat])
        return self._df_summary

    @property
    def ref_qvalues_matrix(self):
        if self._ref_qvalues is None:
            # if len(self._ref_qvalues) > 0:
            self.collect_ref_qvalues()
        return self._ref_qvalues

    @property
    def query_q40(self):
        # if self._query_q40 is None:
        if not all([v for v in self._query_q40.values()]):
            self.collect_q40()
        return self._query_q40

    def count(self):
        for aligned_read in self.fetch_aligned_read():
            d = aligned_read.to_dict()
            self.aligned_reads.append(d)

    def fetch_aligned_read(self):
        for query_read in self.sam_obj:
            if query_read.reference_name is None:
                assert query_read.flag == 4
                ref_seq = ""
            else:
                ref_seq = self.ref_seqs[query_read.reference_name].seq
            aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
            if aligned_read.flag in {0, 4, 16}:
                yield aligned_read

    def count2(self):
        with ThreadPoolExecutor(max_workers=70) as t:
            all_task = [t.submit(f6, query_read, )
                        for query_read in self.sam_obj]
            wait(all_task, return_when=ALL_COMPLETED)
            # for task in as_completed(all_task):
            #     d = task.result()
            #     self.aligned_reads.append(d)
            print('finished')

    def fetch_aligned_read2(self, query_read):
        if query_read.reference_name is None:
            assert query_read.flag == 4
            ref_seq = ""
        else:
            ref_seq = self.ref_seqs[query_read.reference_name].seq
        aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
        if aligned_read.flag in (0, 4, 16):
            self.aligned_reads.append(aligned_read.to_dict())

    def count3(self):
        loop = asyncio.get_event_loop()
        tasks = [self.fetch_aligned_read3(query) for query in self.sam_obj]
        loop.run_until_complete(asyncio.gather(*tasks))
        loop.close()
        print('finish')

    async def fetch_aligned_read3(self, query_read):
        if query_read.reference_name is None:
            assert query_read.flag == 4
            ref_seq = ""
        else:
            ref_seq = self.ref_seqs[query_read.reference_name].seq
        aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
        if aligned_read.flag in (0, 4, 16):
            self.aligned_reads.append(aligned_read.to_dict())

    def count4(self):
        # self.li = Manager().list()
        # with Pool(processes=10) as pool:
        pool = Pool(processes=10)
        for read in self.sam_obj:
            pool.apply_async(self.fetch_aligned_read4, args=(read,))
        # pool.starmap_async(self.fetch_aligned_read4, self.sam_obj)
        pool.close()
        pool.join()

    def fetch_aligned_read4(self, query_read):
        if query_read.reference_name is None:
            assert query_read.flag == 4
            ref_seq = ""
        else:
            ref_seq = self.ref_seqs[query_read.reference_name].seq
        aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
        if aligned_read.flag in (0, 4, 16):
            d = aligned_read.to_dict()
            # self.li.append(d['channel_num'])
            with open("j1.txt", "a") as fp:
                fp.write(d['query_name'] + "\n")

    def count5(self):
        with Pool(processes=CPU_COUNT) as pool:
            for read in self.fetch_aligned_read5():
                d = pool.apply_async(self.f5, args=(read,))
                self.aligned_reads.append(d.get())
        pass

    def fetch_aligned_read5(self):
        for query_read in self.sam_obj:
            if query_read.reference_name is None:
                assert query_read.flag == 4
                ref_seq = ""
            else:
                ref_seq = self.ref_seqs[query_read.reference_name].seq
            aligned_read = AlignedRead(query_read=query_read, ref_seq=ref_seq)
            if aligned_read.flag in {0, 4, 16}:
                a = aligned_read.to_dict()
                yield a['channel_num']

    def summary(self):
        total_read_base = self.df_detail["total_read_base"].sum()
        total_ref_base = self.df_detail["total_ref_base"].sum()
        total_base_del = self.df_detail["total_base_del"].sum()
        total_base_ins = self.df_detail["total_base_ins"].sum()
        total_match = self.df_detail["total_match"].sum()
        total_mismatch = self.df_detail["total_mismatch"].sum()

        total_reads = len(self.df_detail)
        unaligned_reads = len(self.df_detail.query("is_unmapped == True"))

        identity_rate = total_match / (total_ref_base + total_base_ins)
        mismatch_rate = total_mismatch / (total_ref_base + total_base_ins)
        insertion_rate = total_base_ins / (total_ref_base + total_base_ins)
        deletion_rate = total_base_del / (total_ref_base + total_base_ins)

        mapping_rate = 1 - (unaligned_reads / total_reads)

        query_match_qvalues = np.array(
            self.query_counter["query_match_qvalues"])
        query_mismatch_qvalues = np.array(
            self.query_counter["query_mismatch_qvalues"])
        query_ins_qvalues = np.array(self.query_counter["query_ins_qvalues"])

        d = dict(
            mapping_tool=self.mapping_tool,
            total_read_base=total_read_base,
            total_ref_base=total_ref_base,

            # total_num_del=total_num_del,
            total_base_del=total_base_del,
            # total_num_ins=total_num_ins,
            total_base_ins=total_base_ins,

            total_match=total_match,
            total_mismatch=total_mismatch,

            total_reads=total_reads,
            unaligned_reads=unaligned_reads,

            identity_rate=identity_rate,
            mismatch_rate=mismatch_rate,
            insertion_rate=insertion_rate,
            deletion_rate=deletion_rate,

            mapping_rate=mapping_rate,

            query_match_qvalues=query_match_qvalues,
            query_mismatch_qvalues=query_mismatch_qvalues,
            query_ins_qvalues=query_ins_qvalues
        )
        self.aligned_stat.update(d)

    def collect_ref_qvalues(self):
        for _, row in self.df_detail.iterrows():
            flag = row["flag"]
            ref_name = row["ref_name"]
            if flag == 4 or ref_name is None:
                continue
            start = row["ref_start_pos"]
            query_qvales = np.array(row["qvalues"])

            ref_match_indices = row["ref_match_indices"] + start
            query_match_indices = row["query_match_indices"]

            ref_mismatch_indices = row["ref_mismatch_indices"] + start
            query_mismatch_indices = row["query_mismatch_indices"]

            ref_del_indices = row["ref_del_indices"] + start

            if len(ref_match_indices) > 0:
                self.ref_counter[ref_name][0, ref_match_indices] += 1
            if len(ref_mismatch_indices) > 0:
                self.ref_counter[ref_name][1, ref_mismatch_indices] += 1
            if len(ref_del_indices) > 0:
                self.ref_counter[ref_name][2, ref_del_indices] += 1
            if len(ref_match_indices) > 0 and len(query_match_indices) > 0:
                self.ref_counter[ref_name][3,
                                           ref_match_indices] += query_qvales[query_match_indices]
            if len(ref_mismatch_indices) > 0 and len(query_mismatch_indices) > 0:
                self.ref_counter[ref_name][4,
                                           ref_mismatch_indices] += query_qvales[query_mismatch_indices]
        self._ref_qvalues = np.hstack([v for v in self.ref_counter.values()])

    def collect_query_qvalues(self):
        for _, row in self.df_detail.iterrows():
            if row["flag"] == 4:
                continue
            qvalue_seq = row["qvalues"]
            match_indices = row["query_match_indices"]
            mismatch_indices = row["query_mismatch_indices"]
            ins_indices = row["query_ins_indices"]
            read_qvalue = Quality(
                qvalue_seq=qvalue_seq,
                match_indices=match_indices,
                mismatch_indices=mismatch_indices,
                ins_indices=ins_indices
            )
            read_qvalue.group()
            self.query_counter["query_match_qvalues"].extend(
                read_qvalue.match_qvalues)
            self.query_counter["query_mismatch_qvalues"].extend(
                read_qvalue.mismatch_qvalues)
            self.query_counter["query_ins_qvalues"].extend(
                read_qvalue.ins_qvalues)

    def collect_q40(self):
        if not all([v for v in self.query_counter.values()]):
            self.collect_query_qvalues()
        query_match_qvalues = self.query_counter["query_match_qvalues"]
        query_mismatch_qvalues = self.query_counter["query_mismatch_qvalues"]
        query_ins_qvalues = self.query_counter["query_ins_qvalues"]
        try:
            max_qvalue = max(40, np.amax(query_match_qvalues), np.amax(query_mismatch_qvalues),
                             np.amax(query_ins_qvalues), )
        except ValueError:
            max_qvalue = 40
        query_q40_match = self._group_q40(
            qvalues=query_match_qvalues, max_qvalue=max_qvalue)
        query_q40_mismatch = self._group_q40(
            qvalues=query_mismatch_qvalues, max_qvalue=max_qvalue)
        query_q40_ins = self._group_q40(
            qvalues=query_ins_qvalues, max_qvalue=max_qvalue)
        self._query_q40["query_q40_match"] = query_q40_match
        self._query_q40["query_q40_mismatch"] = query_q40_mismatch
        self._query_q40["query_q40_ins"] = query_q40_ins

    def _group_q40(self, qvalues: list, max_qvalue: int):
        uni_qvalues = set(qvalues)
        q40_buckets = np.zeros(max_qvalue + 1)
        qvalue_array = np.array(qvalues)
        for uq in uni_qvalues:
            q40_buckets[uq] += np.sum(qvalue_array == uq)
        return q40_buckets


class AlignedRead:
    """ Aligned read from SAM/BAM file. """

    def __init__(self, ref_seq: str, query_read: pysam.AlignedSegment):
        self._cigar = None
        self.init_attributes(ref_seq=ref_seq, sam_read=query_read)

    def init_attributes(self, ref_seq: str, sam_read: pysam.AlignedSegment):
        # self.ref_seq = ref_seq
        self._aligned_read = sam_read
        self.ref_name = sam_read.reference_name if sam_read.reference_name is not None else ""
        self.ref_read_start_pos = sam_read.reference_start if sam_read.reference_start is not None else 0
        self.ref_read = self.set_ref_read(ref_seq=ref_seq, query_read=sam_read)
        self.query_name = sam_read.query_name
        self.query_read = sam_read.seq
        self.query_length = sam_read.query_length
        self.query_alignment_length = sam_read.query_alignment_length
        self.is_unmapped = sam_read.is_unmapped
        self.flag = sam_read.flag
        self.mapq = sam_read.mapping_quality
        self.cigar_string = sam_read.cigarstring if sam_read.cigarstring is not None else ""
        self.qvalues_string = sam_read.qual
        # self.qvalues = np.array(sam_read.query_qualities) if len(sam_read.query_qualities) > 0 or sam_read is None else np.array([])
        self.qvalues = np.array(
            sam_read.query_qualities) if sam_read.query_qualities else np.array([])

        self.folder_name = None
        self.channel_num = None
        self.read_start_time = None
        self.read_end_time = None
        self.read_start_index = None
        self.read_end_index = None
        self.split_query_name()

    def set_ref_read(self, ref_seq: str, query_read: pysam.AlignedSegment):
        if query_read.flag == 4:
            return ""
        ref_start = query_read.reference_start
        ref_end = query_read.reference_end
        # start = max(0, ref_start - self.padding)
        # end = min(len(ref_seq), ref_end + self.padding)
        return ref_seq[ref_start:ref_end].upper()

    def split_query_name(self):
        query_name_splited = self.query_name.rsplit("_", maxsplit=5)
        self.folder_name = query_name_splited[0]
        self.channel_num = query_name_splited[1]

        time_range = query_name_splited[-2].split("~")
        self.read_start_time = float(time_range[0])
        self.read_end_time = float(time_range[1])

        index_range = query_name_splited[-1].split("-")
        self.read_start_index = int(float(index_range[0]))
        self.read_end_index = int(float(index_range[1]))

    @property
    def coverage(self):
        try:
            _coverage = self.query_alignment_length / self.query_length
        except ZeroDivisionError:
            _coverage = 0.0
        return _coverage

    @property
    def cigar(self):
        if self._cigar is None:
            self._cigar = Cigar(
                cigar_string=self.cigar_string,
                ref_read=self.ref_read,
                query_read=self.query_read
            )
            self._cigar()
            # print("cigar index:", self.query_name, ":", end="\t")
            # print("match:", len(self._cigar.query_match_indices), end="\t")
            # print("mismatch:", len(self._cigar.query_mismatch_indices), end="\t")
            # print("insertion:", len(self._cigar.query_insert_indices))
        return self._cigar

    def to_dict(self):
        dict_ = dict(
            folder_name=self.folder_name,
            channel_num=self.channel_num,
            start_time=self.read_start_time,
            end_time=self.read_end_time,
            start_index=self.read_start_index,
            end_index=self.read_end_index,

            ref_name=self.ref_name,
            ref_start_pos=self.ref_read_start_pos,
            query_name=self.query_name,
            query_length=self.query_length,
            aligned_length=self.query_alignment_length,
            is_unmapped=self.is_unmapped,
            flag=self.flag,
            mapq=self.mapq,
            cigar_string=self.cigar_string,
            qvalues_string=self.qvalues_string,

            coverage=self.coverage,
            qvalues=self.qvalues,

            ref_match_indices=self.cigar.ref_match_indices,
            ref_mismatch_indices=self.cigar.ref_mismatch_indices,
            ref_del_indices=self.cigar.ref_delete_indices,

            query_match_indices=self.cigar.query_match_indices,
            query_mismatch_indices=self.cigar.query_mismatch_indices,
            query_ins_indices=self.cigar.query_insert_indices,
        )
        dict_.update(self.cigar.stats)
        return dict_


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

    def match(self):
        ref_indices = self._find_indices(
            string=self.ref_cigar, operations=["M"])
        qry_indices = self._find_indices(
            string=self.query_cigar, operations=["M"])
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
        self.ref_match_indices = self._find_indices(
            string=self.ref_cigar, operations=["="])
        self.query_match_indices = self._find_indices(
            string=self.query_cigar, operations=["="])

        self.ref_mismatch_indices = self._find_indices(
            string=self.ref_cigar, operations=["X"])
        self.query_mismatch_indices = self._find_indices(
            string=self.query_cigar, operations=["X"])

        self.ref_delete_indices = self._find_indices(
            string=self.ref_cigar, operations=["D"])
        self.query_insert_indices = self._find_indices(
            string=self.query_cigar, operations=["I"])

        self.query_clip_indices = self._find_indices(
            string=self.cigar_ext, operations=["H", "P", "S"])

    def _find_indices(self, string: str, operations: List[str]):
        indices = np.array(
            [i for i, s in enumerate(string) if s in operations])
        return indices

    def count(self):
        if len(self.query_mismatch_indices) == 0:
            total_read_base = len(self.query_read)
        else:
            total_read_base = len(self.query_insert_indices) + len(self.query_match_indices) + len(
                self.query_mismatch_indices)
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


class Quality:

    def __init__(self, qvalue_seq: array, match_indices: list, mismatch_indices: list, ins_indices: list):
        if qvalue_seq is None:
            qvalue_seq = []
        self.qvalue_seq = qvalue_seq
        self.match_indices = match_indices
        self.mismatch_indices = mismatch_indices
        self.ins_indices = ins_indices

        self.match_qvalues = np.array([])
        self.mismatch_qvalues = np.array([])
        self.ins_qvalues = np.array([])

    def group(self):
        # qvalues = np.array([ord(q) - 33 for q in self.qvalue_seq], dtype=int)
        qvalues = np.array([q for q in self.qvalue_seq])

        # if len(self.match_indices) > 0 and all(self.match_indices):
        if len(self.match_indices) > 0:
            self.match_qvalues = qvalues[self.match_indices]

        # if len(self.mismatch_indices) > 0 and all(self.mismatch_indices):
        if len(self.mismatch_indices) > 0:
            self.mismatch_qvalues = qvalues[self.mismatch_indices]

        # if len(self.ins_indices) > 0 and all(self.ins_indices):
        if len(self.ins_indices) > 0:
            self.ins_qvalues = qvalues[self.ins_indices]


class CamFile:

    def __init__(self, cam_file_path: str, mode: str):
        if mode not in {"r", "w"}:
            err_msg = "You can only open cam file to read or write"
            raise CamFileAccessModeError(err_msg)

        self.cam_file_name = cam_file_path
        self.mode = mode

    # def read(self):
    #     if self.mode == "r":
    #         reader = CamFileReader()

    def write(self, df_summary: pd.DataFrame, df_detail: pd.DataFrame,
              ref_qvalues_matrix: np.ndarray, query_q40_match: np.ndarray,
              query_q40_mismatch: np.ndarray, query_q40_ins: np.ndarray):
        if self.mode == "w":
            writer = CamFileWriter(cam_file_path=self.cam_file_name)
            writer.write(
                df_summary=df_summary,
                df_detail=df_detail,
                ref_qvalues_matrix=ref_qvalues_matrix,
                query_q40_match=query_q40_match,
                query_q40_mismatch=query_q40_mismatch,
                query_q40_ins=query_q40_ins
            )


class CamFileReader:

    def __init__(self, cam_file_path: str):
        self.cam_file = h5py.File(name=cam_file_path, mode="r")
        self.folder_name = os.path.splitext(os.path.basename(cam_file_path))[0]

        self.df_summary = None
        self.df_detail = None
        self.ref_qvalues_matrix = None
        self.query_q40_match = None
        self.query_q40_mismatch = None
        self.query_q40_ins = None

    def read(self):
        self.read_summary()
        self.read_detail()
        self.read_qvalues()
        self.cam_file.close()

    def read_summary(self):
        record = dict()
        summary = self.cam_file["summary"]
        for attr in summary.attrs:
            record[attr] = summary.attrs[attr]
        for dataset_name in summary:
            record[dataset_name] = summary[dataset_name][:]
        self.df_summary = pd.DataFrame(data=[record])

    def read_detail(self):
        records = []
        details = self.cam_file["detail"]
        for query_name in details:
            detail = details[query_name]
            record = dict()
            record.update(detail.attrs)
            for dataset_name in detail:
                if dataset_name in {"cigar_string", "qvalues_string"}:
                    record[dataset_name] = detail[dataset_name][0].decode(
                        "utf-8")
                else:
                    record[dataset_name] = detail[dataset_name][:]
            records.append(record)
        self.df_detail = pd.DataFrame(data=records)

    def read_qvalues(self):
        qvalues = self.cam_file["qvalues"]
        self.ref_qvalues_matrix = qvalues["ref_qvalues_matrix"][:]
        self.query_q40_ins = qvalues["query_q40_ins"][:]
        self.query_q40_match = qvalues["query_q40_match"][:]
        self.query_q40_mismatch = qvalues["query_q40_mismatch"][:]


class CamFileWriter:

    def __init__(self, cam_file_path: str):
        self.cam_file = h5py.File(name=cam_file_path, mode="w")
        self.folder_name = os.path.splitext(os.path.basename(cam_file_path))[0]

        self.df_summary = None
        self.df_detail = None
        self.ref_qvalues_matrix = None
        self.query_q40_match = None
        self.query_q40_mismatch = None
        self.query_q40_ins = None

    def write(self, df_summary: pd.DataFrame, df_detail: pd.DataFrame, ref_qvalues_matrix: np.ndarray,
              query_q40_match: np.ndarray, query_q40_mismatch: np.ndarray, query_q40_ins: np.ndarray):
        self.df_summary = df_summary
        self.df_detail = df_detail
        self.ref_qvalues_matrix = ref_qvalues_matrix
        self.query_q40_match = query_q40_match
        self.query_q40_mismatch = query_q40_mismatch
        self.query_q40_ins = query_q40_ins

        self.write_summary()
        self.write_detail()
        self.write_qvalues()
        self.cam_file.close()

    def write_summary(self):
        attrs = {
            "total_read_base": "int",
            "total_ref_base": "int",
            "total_base_del": "int",
            "total_base_ins": "int",
            "total_match": "int",
            "total_mismatch": "int",
            "total_reads": "int",
            "unaligned_reads": "int",
            "identity_rate": "float16",
            "mismatch_rate": "float16",
            "insertion_rate": "float16",
            "deletion_rate": "float16",
            "mapping_rate": "float16"
        }
        datasets = [
            "query_match_qvalues",
            "query_mismatch_qvalues",
            "query_ins_qvalues"
        ]
        summary = self.cam_file.create_group(name="summary")
        for attr in attrs:
            summary.attrs.create(
                name=attr, dtype=attrs[attr], data=self.df_summary.at[0, attr])
        for name in datasets:
            summary.create_dataset(
                name=name,
                dtype="int",
                shape=self.df_summary.at[0, name].shape,
                data=self.df_summary.at[0, name]
            )
        self.cam_file.flush()

    def write_detail(self):
        attrs = {
            "folder_name": h5py.special_dtype(vlen=str),
            "channel_num": h5py.special_dtype(vlen=str),
            "start_time": "float16",
            "end_time": "float16",
            "start_index": "int",
            "end_index": "int",
            "ref_name": h5py.special_dtype(vlen=str),
            "ref_start_pos": "int",
            "query_length": "int",
            "aligned_length": "int",
            "is_unmapped": np.bool,
            "flag": "int",
            "mapq": "float16",
            "coverage": "float16",

            "total_ref_base": "float16",
            "total_read_base": "float16",
            "total_clipped": "float16",
            "total_base_del": "float16",
            "total_base_ins": "float16",
            "total_match": "float16",
            "total_mismatch": "float16",

            "identity_rate": "float16",
            "mismatch_rate": "float16",
            "insertion_rate": "float16",
            "deletion_rate": "float16"
        }
        datasets = [
            "cigar_string",
            "qvalues_string",
            "qvalues",
            "ref_match_indices",
            "ref_mismatch_indices",
            "ref_del_indices",
            "query_match_indices",
            "query_mismatch_indices",
            "query_ins_indices"
        ]
        details = self.cam_file.create_group(name="detail")
        for _, row in self.df_detail.iterrows():
            detail = details.create_group(name=row["query_name"])
            for attr in attrs:
                detail.attrs.create(
                    name=attr, dtype=attrs[attr], data=row[attr])
            for name in datasets:
                if name in {"cigar_string", "qvalues_string"}:
                    detail.create_dataset(name=name, dtype=h5py.special_dtype(
                        vlen=str), shape=(1,), data=row[name])
                    # print(detail[name][0])
                else:
                    """
                    the rest fields are:
                    qvalues
                    ref_match_indices
                    ref_mismatch_indices
                    ref_del_indices
                    query_match_indices
                    query_mismatch_indices
                    query_ins_indices
                    """
                    detail.create_dataset(
                        name=name, dtype="int", shape=row[name].shape, data=row[name])
        self.cam_file.flush()

    def write_qvalues(self):
        qvalues = self.cam_file.create_group(name="qvalues")
        # ref_qvalues_matrix
        qvalues.create_dataset(
            name="ref_qvalues_matrix",
            dtype="float16",
            shape=self.ref_qvalues_matrix.shape,
            data=self.ref_qvalues_matrix
        )
        # query_q40_match
        qvalues.create_dataset(
            name="query_q40_match",
            dtype="int",
            shape=self.query_q40_match.shape,
            data=self.query_q40_match
        )
        # query_q40_mismatch
        qvalues.create_dataset(
            name="query_q40_mismatch",
            dtype="int",
            shape=self.query_q40_mismatch.shape,
            data=self.query_q40_mismatch
        )
        # query_q40_ins
        qvalues.create_dataset(
            name="query_q40_ins",
            dtype="int",
            shape=self.query_q40_ins.shape,
            data=self.query_q40_ins
        )
        self.cam_file.flush()


if __name__ == "__main__":
    # from line_profiler import LineProfiler
    from cam_v1 import Cam as Cam2

    # li = multiprocessing.Manager().list()
    ref = "/home/czx/ws/basecall_visualization/ref_seqs/ecoli.fasta"
    fastq_files = [
        "/home/czx/ws/camstat/data/test.fastq",
        '/mnt/seqdata/output_data/20220127155506_LAB256V2_5K_PC28_28_B16_H49-5c20-J4-F_AD3_Ecoli_gTube_HuangPing_Mux/data_for_analysing/20220127155506_LAB256V2_5K_PC28_28_B16_H49-5c20-J4-F_AD3_Ecoli_gTube_HuangPing_Mux.fastq'
    ]
    sam_file = '/mnt/seqdata/output_data/test/data_for_analysing/test.sam'
    change_mode('test')
    # lp = LineProfiler()
    for fastq_file in fastq_files[:1]:
        start_first = time.time()
        cam = Cam(ref_file=ref, query_file=fastq_file, mapping_tool="minimap2")
        cam.mapping()
        print('cam:', time.time() - start_first)
        start_second = time.time()
        optimized_cam = Cam2(
            ref_file=ref, query_file=fastq_file, mapping_tool='minimap2')
        optimized_cam.mapping()
        print('cam:', time.time() - start_second)
        a = cam.df_detail.sort_values(by=['channel_num', 'start_time'])
        b = optimized_cam.df_detail.sort_values(
            by=['channel_num', 'start_time'])
        cam.df_detail.sort_values(by=['channel_num', 'start_time']).to_csv(
            'origin1.csv', index=False)
        optimized_cam.df_detail.sort_values(
            by=['channel_num', 'start_time']).to_csv('test1.csv', index=False)
        pass
        # lp_wrapper = lp(cam.mapping)
        # lp_wrapper()
        # lp.print_stats()
        # for query_name in cam.df_detail.loc[cam.df_detail["query_name"].duplicated().index]["query_name"].sort_values():
        #     print(query_name, )
        # df = cam.df_detail.loc[cam.df_detail["query_name"].duplicated().index]
        # df.sort_values(by="query_name", inplace=True)
        # for _, row in df.iterrows():
        #     print(row.query_name, row.flag)
    # print(cam.df_detail["total_match"])
    # assert all(cam.df_detail[cam.df_detail.flag == 4])
    # cam_file_path = "/home/czx/ws/basecall_visualization/flask_app/tests/data/20201212_LAB256_20K_MPC29_AD3_HD42_ECOLI_GENOME_WPR_1/data_for_analysing/20201212_LAB256_20K_MPC29_AD3_HD42_ECOLI_GENOME_WPR_1.cam"
    # reader = CamFileReader(cam_file_path=cam_file_path)
    # reader.read()
