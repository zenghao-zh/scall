"""
Bonito Input/Output
"""

import os
import sys
import csv
import pandas as pd
from threading import Thread
from logging import getLogger
from collections import namedtuple
from contextlib import contextmanager
from os.path import realpath, splitext, dirname, basename
import mappy
import math
import numpy as np
from pysam import AlignmentFile, AlignmentHeader, AlignedSegment
from cycloneio.bgi_file import BGIFile
from cyclonebasecall.bonito.convert import typical_indices
from cyclonebasecall.bonito.util import mean_qscore_from_qstring
import h5py


logger = getLogger('bonito')
Format = namedtuple("Format", "aligned name mode")

__ont_bam_spec__ = "0.0.2"


def biofmt(aligned=False):
    """
    Select the output format.
    """
    mode, name = ('w', 'sam') if aligned else ('wfq', 'fastq')
    aligned = "aligned" if aligned else "unaligned"
    stdout = realpath('/dev/fd/1')
    if sys.stdout.isatty() or stdout.startswith('/proc'):
        return Format(aligned, name, mode)
    ext = stdout.split(os.extsep)[-1]
    if ext in ['fq', 'fastq']:
        return Format(aligned, 'fastq', 'wfq')
    elif ext == "bam":
        return Format(aligned, 'bam', 'wb')
    elif ext == "cram":
        return Format(aligned, 'cram', 'wc')
    elif ext == "sam":
        return Format(aligned, 'sam', 'w')
    else:
        return Format(aligned, name, mode)


def encode_moves(moves, stride, sep=','):
    """
    Encode a numpy array of integers into a comma seperated string
    starting with `stride`. For efficiency, this method is only
    valid for +ve single digit values in `moves`.

    >>> encode_moves(np.array([0, 1, 0, 1, 1], dtype=np.int8), 5)
    '5,0,1,0,1,1'
    """
    separators = np.full(2 * moves.size, ord(sep), dtype=np.dtype('B'))
    # convert moves to ascii and interleave with separators
    #  ~3 orders faster than `sep.join(np.char.mod("%d", moves))`
    separators[1::2] = moves + ord('0')
    return f"{stride}{separators.tobytes().decode('ascii')}"


@contextmanager
def devnull(*args, **kwds):
    """
    A context manager that sends all out stdout & stderr to devnull.
    """
    save_fds = [os.dup(1), os.dup(2)]
    null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)
    try:
        yield
    finally:
        os.dup2(save_fds[0], 1)
        os.dup2(save_fds[1], 2)
        for fd in null_fds + save_fds: os.close(fd)


def write_fasta(header, sequence, fd=sys.stdout):
    """
    Write a fasta record to a file descriptor.
    """
    fd.write(f">{header}\n{sequence}\n")


def write_fastq(header, sequence, qstring, fd=sys.stdout, tags=None, sep="\t"):
    """
    Write a fastq record to a file descriptor.
    """
    if tags is not None:
        fd.write(f"@{header} {sep.join(tags)}\n")
    else:
        fd.write(f"@{header}\n")
    fd.write(f"{sequence}\n+\n{qstring}\n")


def sam_header(groups, sep='\t'):
    """
    Format a string sam header.
    """
    HD = sep.join([
        '@HD',
        'VN:1.5',
        'SO:unknown',
        'ob:%s' % __ont_bam_spec__,
    ])
    PG1 = sep.join([
        '@PG',
        'ID:basecaller',
        'PN:bonito',
        'VN: no',
        'CL:bonito %s' % ' '.join(sys.argv[1:]),
    ])
    PG2 = sep.join([
        '@PG',
        'ID:aligner',
        'PN:minimap2',
        'VN:%s' % mappy.__version__,
        'DS:mappy',
    ])
    return '%s\n' % os.linesep.join([HD, PG1, PG2, *groups])


def sam_record(read_id, sequence, qstring, mapping, tags=None, sep='\t'):
    """
    Format a string sam record.
    """
    if mapping:
        softclip = [
            '%sS' % mapping.q_st if mapping.q_st else '',
            mapping.cigar_str,
            '%sS' % (len(sequence) - mapping.q_en) if len(sequence) - mapping.q_en else ''
        ]
        record = [
            read_id,
            0 if mapping.strand == +1 else 16,
            mapping.ctg,
            mapping.r_st + 1,
            mapping.mapq,
            ''.join(softclip if mapping.strand == +1 else softclip[::-1]),
            '*', 0, 0,
            sequence if mapping.strand == +1 else mappy.revcomp(sequence),
            qstring,
            'NM:i:%s' % mapping.NM,
            'MD:Z:%s' % mapping.MD,
        ]
    else:
        record = [
            read_id, 4, '*', 0, 0, '*', '*', 0, 0, sequence, qstring, 'NM:i:0'
        ]

    if tags is not None:
        record.extend(tags)

    return sep.join(map(str, record))


def summary_file():
    """
    Return the filename to use for the summary tsv.
    """
    stdout = realpath('/dev/fd/1')
    if sys.stdout.isatty() or stdout.startswith('/proc'):
        return 'summary.tsv'
    return '%s_summary.tsv' % splitext(stdout)[0]


summary_field_names = [
    'filename',
    'read_id',
    'run_id',
    'channel',
    'mux',
    'start_time',
    'duration',
    'template_start',
    'template_duration',
    'sequence_length_template',
    'mean_qscore_template',
    #if alignment
    'alignment_genome',
    'alignment_genome_start',
    'alignment_genome_end',
    'alignment_strand_start',
    'alignment_strand_end',
    'alignment_direction',
    'alignment_length',
    'alignment_num_aligned',
    'alignment_num_correct',
    'alignment_num_insertions',
    'alignment_num_deletions',
    'alignment_num_substitutions',
    'alignment_mapq',
    'alignment_strand_coverage',
    'alignment_identity',
    'alignment_accuracy',
]


def summary_row(read, seqlen, qscore, alignment=False):
    """
    Summary tsv row.
    """
    fields = [
        read.filename,
        read.read_id,
        read.run_id,
        read.channel,
        read.mux,
        read.start,
        read.duration,
        read.template_start,
        read.template_duration,
        seqlen,
        qscore,
    ]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            alignment.mapq,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ])

    elif alignment is None:
        fields.extend(
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        )

    return dict(zip(summary_field_names, fields))


duplex_summary_field_names = [
    'filename_template',
    'read_id_template',
    'filename_complement',
    'read_id_complement',
    'run_id',
    'channel_template',
    'mux_template',
    'channel_complement',
    'mux_complement',
    'sequence_length_duplex',
    'mean_qscore_duplex',
    #if alignment
    'alignment_genome',
    'alignment_genome_start',
    'alignment_genome_end',
    'alignment_strand_start',
    'alignment_strand_end',
    'alignment_direction',
    'alignment_length',
    'alignment_num_aligned',
    'alignment_num_correct',
    'alignment_num_insertions',
    'alignment_num_deletions',
    'alignment_num_substitutions',
    'alignment_mapq',
    'alignment_strand_coverage',
    'alignment_identity',
    'alignment_accuracy',
]


def duplex_summary_row(read_temp, comp_read, seqlen, qscore, alignment=False):
    """
    Duplex summary tsv row.
    """
    fields = [
        read_temp.filename,
        read_temp.read_id,
        comp_read.filename,
        comp_read.read_id,
        read_temp.run_id,
        read_temp.channel,
        read_temp.mux,
        comp_read.channel,
        comp_read.mux,
        seqlen,
        qscore,
    ]

    if alignment:

        ins = sum(count for count, op in alignment.cigar if op == 1)
        dels = sum(count for count, op in alignment.cigar if op == 2)
        subs = alignment.NM - ins - dels
        length = alignment.blen
        matches = length - ins - dels
        correct = alignment.mlen

        fields.extend([
            alignment.ctg,
            alignment.r_st,
            alignment.r_en,
            alignment.q_st if alignment.strand == +1 else seqlen - alignment.q_en,
            alignment.q_en if alignment.strand == +1 else seqlen - alignment.q_st,
            '+' if alignment.strand == +1 else '-',
            length, matches, correct,
            ins, dels, subs,
            alignment.mapq,
            (alignment.q_en - alignment.q_st) / seqlen,
            correct / matches,
            correct / length,
        ])

    elif alignment is None:
        fields.extend(
            ['*', -1, -1, -1, -1, '*', 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0]
        )

    return dict(zip(duplex_summary_field_names, fields))


def moves_to_seq(seq_str, moves):
    """
    将碱基字符串seq_str（如'ACGT'）和moves（0/1数组）转换为Seq数组：
    A=1, C=2, G=3, T=4, 其他=0。moves==1时依次用seq数字填充，moves==0时填0。
    """
    # 检查moves中1的个数和seq_str长度一致
    moves_ones = sum(moves)
    assert moves_ones == len(seq_str), f"moves中1的个数({moves_ones})与seq_str长度({len(seq_str)})不一致"
    
    base2num = {'A': 1, 'C': 2, 'G': 3, 'T': 4}
    seq_nums = [base2num.get(b, 0) for b in seq_str]
    seq_array = []
    seq_idx = 0
    for m in moves:
        if m == 1:
            if seq_idx < len(seq_nums):
                seq_array.append(seq_nums[seq_idx])
                seq_idx += 1
            else:
                seq_array.append(0)
        else:
            seq_array.append(0)
    return np.array(seq_array, dtype=np.int8)


class CSVLogger:
    def __init__(self, filename, sep=','):
        self.filename = str(filename)
        if os.path.exists(self.filename):
            with open(self.filename) as f:
                self.columns = csv.DictReader(f).fieldnames
        else:
            self.columns = None
        self.fh = open(self.filename, 'a', newline='')
        self.csvwriter = csv.writer(self.fh, delimiter=sep)
        self.count = 0

    def set_columns(self, columns):
        if self.columns:
            raise Exception('Columns already set')
        self.columns = list(columns)
        self.csvwriter.writerow(self.columns)

    def append(self, row):
        if self.columns is None:
            self.set_columns(row.keys())
        self.csvwriter.writerow([row.get(k, '-') for k in self.columns])
        self.count += 1
        if self.count > 100:
            self.count = 0
            self.fh.flush()

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class NullWriter(Thread):

    def __init__(self, mode, iterator, duplex=False, **kwargs):
        super().__init__()
        self.log = []
        self.duplex = duplex
        self.iterator = iterator

    def run(self):

        for read, res in self.iterator:
            if self.duplex:
                samples = len(read[0].signal) + len(read[1].signal)
                read_id = '%s;%s' % (read[0].read_id, read[1].read_id)
            else:
                samples = len(read.signal)
                read_id = read.read_id
            self.log.append((read_id, samples))


class Writer(Thread):

    def __init__(self, mode, iterator, aligner, fd=sys.stdout, fastq_path=None, duplex=False, ref_fn=None, groups=None, group_key=None, verbose=False):
        super().__init__()
        self.fd = fd
        self.log = []
        self.mode = mode
        self.duplex = duplex
        self.aligner = aligner
        self.iterator = iterator
        self.fastq_path = fastq_path
        self.group_key = group_key
        self.verbose = verbose

    def run(self):
        # with CSVLogger(summary_file(), sep='\t') as summary:
        # if os.path.exists(self.fastq_path):
        #     os.remove(self.fastq_path)
        f = open(self.fastq_path, "a")
        if self.verbose:
            self.h5_path = self.fastq_path.replace('.fastq', '.h5')
            h5file = h5py.File(self.h5_path, "a")
            
        for read, res in self.iterator:
            seq = res['sequence']
            qstring = res.get('qstring', '*')
                
            samples = len(read.signal)
            read_id = read.read_id
            if 'channel' in read_id:
                pass
            else:
                channel = splitext(basename(read.file_path))[0]
                read_id = channel + '_' + read.read_id
            # calculate q value
            # qvalues = [ord(x) - 33 for x in qstring]
            # read_length = len(qvalues)
            # if read_length > 0:
            #     err_rate = sum([math.pow(10,-x/10) for x in qvalues])/read_length
            # else:
            #     err_rate = 0
            # # print(err_rate)
            # if err_rate == 0:
            #     q_read = 0
            # else:
            #     q_read = -10 * math.log10(err_rate)            
            # total_read_base = len(seq)

            if len(seq):
                # write_fastq(read_id, seq, qstring, fd=self.fd, tags=tags)
                f.write("{}{}\n{}\n".format("@", read_id, seq))
                f.write("+\n{}\n".format(qstring))
                self.log.append((read_id, samples))
                if self.verbose:
                    grp = h5file.require_group(read_id)
                    # 生成Seq并写入
                    # seq_array = moves_to_seq(res['sequence'], res['moves'])
                    # if "Seq" in grp:
                    #     del grp["Seq"]
                    # grp.create_dataset("Seq", data=seq_array, compression="gzip")
                    if "mv" in grp:
                        del grp["mv"]
                    grp.create_dataset("mv", data=res['moves'], compression="gzip")

            else:
                logger.warn("> skipping empty sequence %s", read_id)
        f.close()
        if self.verbose:
            h5file.close()


class Hybrid_Writer(Thread):

    def __init__(self, iterator, fastq_out, h5_path, run_id, scale=1, offset=0, rm_poly=1):
        super().__init__()
        self.log = []
        self.iterator = iterator
        self.fastq_out = fastq_out
        self.basecall_h5_path = h5_path
        self.scale = scale
        self.offset = offset
        self.run_id = run_id
        self.read = {}
        self.read["adaptor_len"] = 0
        self.read["h5_name"] = int(os.path.splitext(os.path.basename(h5_path))[0])
        self.rm_poly = True if int(rm_poly) == 1 else False
        self.ccf_name = int(os.path.splitext(os.path.basename(fastq_out))[0])
        self.logger = logger

    def h5_info_collect(self, read, q_read, err_rate, total_read_base, q_thr=7):
        run_id = read["run_id"]
        data_name = read["data_name"]
        channel_num = read["channel_num"]
        start_index = read["start_index"]
        end_index = read["end_index"]
        duration = read["duration"]
        adaptor_len = read["adaptor_len"]
        h5_name = read["h5_name"]
        q_val ="{:.2f}".format(q_read).zfill(5)
        if float(q_val) >= q_thr:
            is_high_q = True
        else:
            is_high_q = False
        # fastq_read_id = f"{run_id}_{data_name}_{str(channel_num).zfill(5)}_{str(int(start_index)).zfill(10)}_{q_val}"
        fastq_read_id = f"{run_id}_{data_name}_{str(channel_num).zfill(5)}_{str(int(start_index/5)).zfill(9)}_{q_val}"
        read_info = {
            "run_id": run_id,
            "data_name": data_name,
            "channel_num": channel_num,
            "read_id": fastq_read_id,
            "start_index": start_index,
            "end_index": end_index,
            "duration": duration,
            "adaptor_len": adaptor_len,
            "h5_name": h5_name,
            "err_rate": err_rate,
            "q_read": q_read,
            "total_read_base": total_read_base,
        }

        return fastq_read_id, read_info, is_high_q

    def run(self):
        self.valid_seq = 0
        self.total_reads = 0
        self.passed_reads = 0
        self.failed_reads = 0
        self.total_bases = 0
        self.passed_bases = 0
        self.failed_bases = 0
        total_read_base_list = []
        total_read_base_passed_list = []
        speed_list = []
        qvalue_list = []        
        
        out_handle = open(self.fastq_out, "a")

        for read, res in self.iterator:
            seq = res['sequence']
            qstring = res.get('qstring', '*')
            samples = len(read.signal)
            read_id = read.read_id  # 'read_20231216193747_channel2062_Read_0_40.82~40.9_12247131-12270197'
            self.read["run_id"] = self.run_id
            self.read["data_name"] = read_id.split("_")[1]
            self.read["channel_num"] = read_id.split("_Read")[0].split("channel")[-1]
            self.read["start_index"] = int(read_id.split("_")[-1].split("-")[0])
            self.read["end_index"] = int(read_id.split("_")[-1].split("-")[1])
            self.read["duration"] = self.read["end_index"] - self.read["start_index"]
            
            if len(seq):
                if self.rm_poly:
                    if len(seq) <= 5:
                        continue
                    seq, qstring = RMPoly.remove_header_poly(seq, qstring, poly_limit=5)
                    seq, qstring = RMPoly.remove_tail_poly(seq, qstring, poly_limit=15)
                    # seq, qstring = RMPoly.remove_mid_poly(seq, qstring, poly_limit=50)
                    if not seq:
                        continue
                seq_len = len(seq)
                           
                if seq_len > 0: 
                    # q-val calibration after rm-poly
                    qvalues = [ord(x) - 33 for x in qstring]
                    total_read_base = len(qvalues)
                    if total_read_base > 0:
                        err_rate = (
                                sum([math.pow(10, -x / 10) for x in qvalues]) / total_read_base
                        )
                        q_read = -10 * math.log10(err_rate)
                        q_read_adjust = self.scale * q_read + self.offset
                        err_rate_adjust = math.pow(10, -(q_read_adjust / 10))
                    else:
                        err_rate = 0
                        q_read = 0
                        q_read_adjust = 0
                        err_rate_adjust = 0

                    fastq_read_id, read_info, is_high_q = self.h5_info_collect(
                        self.read,
                        q_read_adjust,
                        err_rate_adjust,
                        total_read_base,
                    )            
            
                    self.valid_seq += 1
                    if self.valid_seq == 1:
                        all_read_info = {}
                        for k, v in read_info.items():
                            all_read_info[k] = []

                    for k, v in read_info.items():
                        all_read_info[k].append(v)

                    self.total_reads += 1
                    self.total_bases += seq_len
                    total_read_base_list.append(seq_len)
                    duration = int(read_info["duration"])
                    speed_list.append(seq_len * 5000 / duration)
                    qvalue_list.append(q_read_adjust)
                    
                    out_handle.write("@%s\n%s\n+\n%s\n" % (fastq_read_id, seq, qstring))                  
                    self.log.append(samples)
            else:
                logger.warn("> skipping empty sequence %s", read_id)
        out_handle.close()
        fast5_w = BGIFile(self.basecall_h5_path, "a")
        fast5_w.save_h5(all_read_info)
        fast5_w.close()

class CTCWriter(Thread):
    """
    CTC writer process that writes output numpy training data.
    """
    def __init__(
            self, mode, iterator, aligner, fd=sys.stdout, min_coverage=0.90,
            min_accuracy=0.99, ref_fn=None, groups=None, group_key=None,
    ):
        super().__init__()
        self.fd = fd
        self.log = []
        self.mode = mode
        self.aligner = aligner
        self.iterator = iterator
        self.group_key = group_key
        self.min_coverage = min_coverage
        self.min_accuracy = min_accuracy
        self.output = AlignmentFile(
            fd, 'w' if self.mode == 'wfq' else self.mode, add_sam_header=self.mode != 'wfq',
            reference_filename=ref_fn,
            header=AlignmentHeader.from_references(
                reference_names=aligner.seq_names,
                reference_lengths=[len(aligner.seq(name)) for name in aligner.seq_names],
                text=sam_header(groups),
            )
        )

    def run(self):

        chunks = []
        targets = []
        lengths = []

        with CSVLogger(summary_file(), sep='\t') as summary:
            for read, ctc_data in self.iterator:

                seq = ctc_data['sequence']
                qstring = ctc_data['qstring']
                mean_qscore = ctc_data.get('mean_qscore', mean_qscore_from_qstring(qstring))
                mapping = ctc_data.get('mapping', False)

                self.log.append((read.read_id, len(read.signal)))

                if len(seq) == 0 or mapping is None:
                    continue

                cov = (mapping.q_en - mapping.q_st) / len(seq)
                acc = mapping.mlen / mapping.blen
                refseq = self.aligner.seq(mapping.ctg, mapping.r_st, mapping.r_en)

                if acc < self.min_accuracy or cov < self.min_coverage or 'N' in refseq:
                    continue

                self.output.write(
                    AlignedSegment.fromstring(
                        sam_record(read.read_id, seq, qstring, mapping),
                        self.output.header
                    )
                )
                summary.append(summary_row(read, len(seq), mean_qscore, alignment=mapping))

                if mapping.strand == -1:
                    refseq = mappy.revcomp(refseq)

                target = [int(x) for x in refseq.translate({65: '1', 67: '2', 71: '3', 84: '4'})]
                targets.append(target)
                chunks.append(read.signal)
                lengths.append(len(target))

        if len(chunks) == 0:
            sys.stderr.write("> no suitable ctc data to write\n")
            return

        chunks = np.array(chunks, dtype=np.float16)
        targets_ = np.zeros((chunks.shape[0], max(lengths)), dtype=np.uint8)
        for idx, target in enumerate(targets): targets_[idx, :len(target)] = target
        lengths = np.array(lengths, dtype=np.uint16)
        indices = np.random.permutation(typical_indices(lengths))

        chunks = chunks[indices]
        targets_ = targets_[indices]
        lengths = lengths[indices]

        summary = pd.read_csv(summary_file(), sep='\t')
        summary.iloc[indices].to_csv(summary_file(), sep='\t', index=False)

        output_directory = '.' if sys.stdout.isatty() else dirname(realpath('/dev/fd/1'))
        np.save(os.path.join(output_directory, "chunks.npy"), chunks)
        np.save(os.path.join(output_directory, "references.npy"), targets_)
        np.save(os.path.join(output_directory, "reference_lengths.npy"), lengths)

        sys.stderr.write("> written ctc training data\n")
        sys.stderr.write("  - chunks.npy with shape (%s)\n" % ','.join(map(str, chunks.shape)))
        sys.stderr.write("  - references.npy with shape (%s)\n" % ','.join(map(str, targets_.shape)))
        sys.stderr.write("  - reference_lengths.npy shape (%s)\n" % ','.join(map(str, lengths.shape)))

    def stop(self):
        self.join()
