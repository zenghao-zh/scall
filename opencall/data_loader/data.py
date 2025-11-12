import sys
from torch.utils.data import IterableDataset
import importlib
import re
import h5py
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
import os
from itertools import islice
import numpy as np
from opencall.libs.taiyaki import mapped_signal_files, flipflopfings
import random
import torch.distributed as dist
import numpy as np
import scipy.signal as signal
import gc
import glob
from sklearn import utils

def sliding_mean(data, window_length, step):
    window_means = []
    for i in range(0, len(data) - window_length + 1, step):
        window = data[i:i+window_length]
        mean = np.mean(window,dtype=int)
        window_means.append(mean)
    return window_means


def normalize_sequence(sequence, percentile=1):
    lower_bound = np.percentile(sequence, percentile, )
    upper_bound = np.percentile(sequence, 100 - percentile)

    new_range = upper_bound - lower_bound
    normalized_seq = [(x - lower_bound) / new_range * 2 - 1 for x in sequence]
    return normalized_seq


def subtract_sliding_mean(data, window_length, step):
    sliding_means = sliding_mean(data, window_length, step)
    subtracted_data = data.copy()
    for i, mean in enumerate(sliding_means):
        subtracted_data[i*step:(i+1)*step] -= mean
    subtracted_data[(i+1)*step:] -= mean
    return subtracted_data


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


class TrainingDataSet(Dataset):
    def __init__(self, data_path, index_file_path, tokenization):
        self.read_data = self._load_hd5(data_path)
        self.region_np_orig = self._load_npy(index_file_path)
        self.region_np = self.region_np_orig[:-1, :]
        self.maxlen = self.region_np_orig[-1, :][0]
        self.tokenization = tokenization

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _load_hd5(self, input_path, limit_num=None):
        print("* Loading data from {}\n".format(input_path))
        print("* Reads not filtered by id\n")
        read_ids = None
        if limit_num:
            print("* Limiting number of strands to {}\n".format(limit_num))
        with mapped_signal_files.MappedSignalReader(input_path) as msr:
            alphabet_info = msr.get_alphabet_information()
            # load list of signal_mapping.SignalMapping objects
            read_data = list(islice(msr.reads(read_ids), limit_num))
            print("debug")
        print("* Using alphabet definition: {}\n".format(str(alphabet_info)))

        if len(read_data) == 0:
            print("* No reads remaining for training, exiting.\n")
            exit(1)
        print("* Loaded {} reads.\n".format(len(read_data)))

        return read_data

    def __getitem__(self, index):
        read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
            index, :
        ].tolist()
        read = self.read_data[read_index]
        cur = read.get_current((cur_start, cur_end), standardize=True)
        refs = read.Reference[ref_start:ref_end]

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

        seqs[: len(seqs_orig)] = seqs_orig
        seqlen = len(seqs_orig)
        indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT

        return indata.astype(np.float32), seqs, seqlen

    def __len__(self):
        return self.region_np.shape[0]


class TrainingDataSet2(IterableDataset):
    """DDP training set for npy index and h5 data, load data on runtime"""
    def __init__(self, data_path, index_file_path, tokenization, limit_size, data_len=5000):
        super(TrainingDataSet2).__init__()
        self.data_len = data_len
        self._hd5_init(data_path)
        self.region_np_orig = self._load_npy(index_file_path)
        limit_size = min(len(self.region_np_orig[:-1, :]), limit_size)
        self.region_np = self.region_np_orig[:limit_size, :]
        self.maxlen = self.region_np_orig[-1, :][0]
        self.tokenization = tokenization
        self.index_list = list(range(self.region_np.shape[0]))

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.index_list)

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _hd5_init(self, input_path):
        self.hd5_file = h5py.File(input_path, "r")
        self.batch_list = list(self.hd5_file.keys())
        self.sort_batch_list = sorted(
            self.batch_list, key=lambda x: int(re.search("_(\d+)", x).group(1))
        )
        self.index_map = {}
        index_count = 0
        for batch_i in self.sort_batch_list:
            for read_id in self.hd5_file[batch_i]:
                self.index_map[index_count] = (batch_i, read_id)  # index_count is the read_index in self.region_np  
                index_count += 1

    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        for index in self.index_list:
            if index % total_workers == global_worker_id:
                read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
                    index, :
                ].tolist()
                batch_i, read_id = self.index_map[read_index]
                read = {}
                read["read_id"] = read_id
                read["Signal"] = self.hd5_file[batch_i][read_id]["Signal"][()]
                read["Seq"] = self.hd5_file[batch_i][read_id]["Seq"][()]
                read["Seq_to_signal"] = self.hd5_file[batch_i][read_id][
                    "Seq_to_signal"
                ][()]
                read["digitisation"] = self.hd5_file[batch_i][read_id].attrs[
                    "digitisation"
                ]
                read["offset"] = self.hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = self.hd5_file[batch_i][read_id].attrs["range"]
                read["scale_frompA"] = self.hd5_file[batch_i][read_id].attrs[
                    "scale_frompA"
                ]
                read["shift_frompA"] = self.hd5_file[batch_i][read_id].attrs[
                    "shift_frompA"
                ]

                cur = self._get_current(read, (cur_start, cur_end), standardize=True)
                refs = read["Seq"][ref_start:ref_end]

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

                seqs[: len(seqs_orig)] = seqs_orig
                seqlen = len(seqs_orig)
                indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
                yield (indata.astype(np.float32), seqs, seqlen)
        self.hd5_file.close()

    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.region_np.shape[0]


class TestingDataSet2(IterableDataset):
    """DDP test set for npy index and h5 data, load data on runtime"""
    def __init__(self, data_path, index_file_path, limit_size, tokenization, data_len=5000):
        super(TrainingDataSet2).__init__()
        self.data_len = data_len
        self._hd5_init(data_path)
        self.region_np_orig = self._load_npy(index_file_path)
        limit_size = min(len(self.region_np_orig[:-1, :]), limit_size)
        self.region_np = self.region_np_orig[:limit_size, :]
        self.maxlen = self.region_np_orig[-1, :][0]
        self.tokenization = tokenization
        self.index_list = list(range(self.region_np.shape[0]))

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _hd5_init(self, input_path):
        self.hd5_file = h5py.File(input_path, "r")
        self.batch_list = list(self.hd5_file.keys())
        self.sort_batch_list = sorted(
            self.batch_list, key=lambda x: int(re.search("_(\d+)", x).group(1))
        )
        self.index_map = {}
        index_count = 0
        for batch_i in self.sort_batch_list: 
            for read_id in self.hd5_file[batch_i]:
                self.index_map[index_count] = (batch_i, read_id)  # index_count is the read_index in self.region_np
                index_count += 1

    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        for index in self.index_list:
            read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
                index, :
            ].tolist()
            batch_i, read_id = self.index_map[read_index]
            read = {}
            read["read_id"] = read_id
            read["Signal"] = self.hd5_file[batch_i][read_id]["Signal"][()]
            read["Seq"] = self.hd5_file[batch_i][read_id]["Seq"][()]
            read["Seq_to_signal"] = self.hd5_file[batch_i][read_id]["Seq_to_signal"][()]
            read["digitisation"] = self.hd5_file[batch_i][read_id].attrs["digitisation"]
            read["offset"] = self.hd5_file[batch_i][read_id].attrs["offset"]
            read["range"] = self.hd5_file[batch_i][read_id].attrs["range"]
            read["scale_frompA"] = self.hd5_file[batch_i][read_id].attrs["scale_frompA"]
            read["shift_frompA"] = self.hd5_file[batch_i][read_id].attrs["shift_frompA"]

            cur = self._get_current(read, (cur_start, cur_end), standardize=True)
            refs = read["Seq"][ref_start:ref_end]

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

            seqs[: len(seqs_orig)] = seqs_orig
            seqlen = len(seqs_orig)
            indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
            yield (indata.astype(np.float32), seqs, seqlen)
        self.hd5_file.close()

    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.region_np.shape[0]


class TrainingDataSet3(Dataset):
    def __init__(self, data_dir, tokenization):
        self._load_hd5_npy(data_dir)
        self.tokenization = tokenization

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.region_np)


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
            last_row = dat_npy[-1:]
            data_rows = dat_npy[:-1]
            mask = data_rows[:,5] == 0
            filtered_data = data_rows[mask]
            if filtered_data.shape[0] == 0:
                continue
            dat_npy = filtered_data
            if last_row[0, 0] > self.maxlen:
                self.maxlen = int(last_row[0, 0])
            dat_npy = np.column_stack((dat_npy, np.array([hd5_num]*dat_npy.shape[0])))
            npy_list.append(dat_npy)
            hd5_num += 1
        dat_np = np.concatenate(npy_list, axis = 0).astype(int)
        
        # Copy dat_np 5 times, shuffle each with different seed, then concatenate
        # shuffled_copies = []
        # for i in range(5):
        #     shuffled_copy = utils.shuffle(dat_np, random_state=i)
        #     shuffled_copies.append(shuffled_copy)
        
        # self.region_np = np.concatenate(shuffled_copies, axis=0)
        self.region_np = utils.shuffle(dat_np, random_state=0)


    def _load_hd5(self, read_index, cur_start, cur_end, ref_start, ref_end, hd5_index):
        hd5_file = self.hd5_list[hd5_index]
        per_num = hd5_file.attrs['batch_size']
        batch_num = int(read_index // per_num)
        read = hd5_file['batch_{}'.format(batch_num)][str(read_index)]
        cur = self._get_current(read, (cur_start, cur_end), standardize=True)
        #ref_start2 = min((ref_start + 10), (ref_end-1))
        #ref_end2 = max(1, (ref_end - 10))
        ref_start2 = ref_start
        ref_end2 = ref_end
        refs = read['Seq'][ref_start2:ref_end2]
        del read
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
        try:
            cur, refs = self._load_hd5(read_index, cur_start, cur_end, ref_start, ref_end, hd5_index)
        except Exception as e:
            hd5_file = self.hd5_list[hd5_index]
            per_num = hd5_file.attrs['batch_size']
            batch_num = int(read_index // per_num)
            read = hd5_file['batch_{}'.format(batch_num)]
            print(f"Error loading HD5 file: {e}")
            print(f"Read index: {read_index}, HD5 file: {hd5_file.filename}")
            return None, None, None


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


class TrainingDataSet3_Encoder(Dataset):
    def __init__(self, data_dir, tokenization):
        self._load_hd5_npy(data_dir)
        self.tokenization = tokenization

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
            
            # Keep the last row for maxlen info
            last_row = dat_npy[-1:]
            data_rows = dat_npy[:-1]
            
            ref_len = data_rows[:, 4] - data_rows[:, 3] 
            valid_lengths = (ref_len >= 300) & (ref_len <= 450)
            filtered_data = data_rows[valid_lengths]
            
            # Skip if no valid data after filtering (excluding last row)
            if filtered_data.shape[0] == 0:
                continue
                
            # Combine filtered data with last row
            dat_npy = np.concatenate([filtered_data, last_row])
            

            dat_npy = np.column_stack((dat_npy, np.array([hd5_num]*dat_npy.shape[0])))
            npy_list.append(dat_npy[0:-1, :])
            hd5_num += 1
            
        if not npy_list:  # Check if we have any valid data
            raise ValueError("No valid data found after length filtering (300-450 bases)")
            
        dat_np = np.concatenate(npy_list, axis = 0).astype(int)
        self.region_np = utils.shuffle(dat_np, random_state=0)
        self.maxlen = 1000

    def _load_hd5(self, read_index, cur_start, cur_end, ref_start, ref_end, hd5_index):
        hd5_file = self.hd5_list[hd5_index]
        per_num = hd5_file.attrs['batch_size']
        batch_num = int(read_index // per_num)
        read = hd5_file['batch_{}'.format(batch_num)][str(read_index)]
        cur = self._get_current(read, (cur_start, cur_end), standardize=True)
        refs = read['Seq'][ref_start:ref_end]
        
        # 将refs填充到1000长度，按照Seq_to_signal对应位置填充，其余填0
        seq_to_signal = read['Seq_to_signal'][:]  # 原标签为0-3
        refs_padded = np.zeros(1000, dtype=refs.dtype)
        
        # 根据Seq_to_signal映射，将refs填充到对应位置
        
        if seq_to_signal[-1] == 1000:  # 去掉最后一个位置出现多个碱基的情况
            while seq_to_signal[-1] == 1000:
                refs = refs[:-1]
                ref_end -= 1
                seq_to_signal = seq_to_signal[:-1]
        signal_positions = seq_to_signal[ref_start:ref_end]
        refs = refs + 1 # 原标签为0-3，现在为1-4
        refs_padded[signal_positions] = refs
        # Replace 0s in refs_padded with the first non-zero value before them
        # (for each 0, look left for the first non-zero, use that; if none, leave as 0)
        # 改这里
        non_zero = 0
        for i in range(len(refs_padded)):
            if refs_padded[i] != 0:
                non_zero = refs_padded[i]
            elif non_zero != 0:
                refs_padded[i] = non_zero + 4
        return cur, refs_padded
    
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
        read_index, cur_start, cur_end, ref_start, ref_end ,cycle_num, max_read_ins, max_read_del, hd5_index = self.region_np[index, :].tolist()
        cur, refs = self._load_hd5(read_index, cur_start, cur_end, ref_start, ref_end, hd5_index)

        assert self.tokenization == "kmer"

        indata = np.expand_dims(cur, axis=1).transpose((1, 0))  # CT

        return indata.astype(np.float32), refs, 1000

    def __len__(self):
        return self.region_np.shape[0]


class DataSetMulti_old_index(IterableDataset):
    """DDP testset for multiple npy index and h5 data, load data on runtime"""
    def __init__(self, h5_list, npy_list, train_size, test_size, tokenization, data_len=5000, shuffle_seed=100):
        super(DataSetMulti_old_index).__init__()
        self.data_len = data_len
        self.shuffle_seed = shuffle_seed
        self.h5_list = h5_list
        self.npy_list = npy_list
        assert len(h5_list) == len(npy_list)
        self._hd5_npy_init()
        self.tokenization = tokenization
        self.total_num = self.region_np.shape[0]
        self.test_size = min(self.total_num, test_size)
        self.train_size = min(self.total_num, train_size)
        assert self.train_size > 0
        self.index_list = list(range(self.train_size))
        self.index_val = None
        self.index_train = None
        self.data_size = self.train_size

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.index_list)

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _search_sub_index(self, index):
        # search for the index of sub data
        for i in range(len(self.acc_len)):
            if self.acc_len[i] > index:
                return i
        raise IndexError("This code should not be reached, check _hd5_npy_init!")
            
    def _hd5_npy_init(self):
        self.handle_list = []
        region_np_list = []
        maxlen_list = []
        total_len = 0
        self.acc_len = []  # accumulated len, use to return the index of subdataset, base on np len
        self.index_map_list = []  # indices for each sub dataset
        for i in range(len(self.h5_list)):
            print(f"loading {self.npy_list[i]} in and it's hd5")
            hd5_file = h5py.File(self.h5_list[i], "r")
            region_np_orig = self._load_npy(self.npy_list[i])
            if len(region_np_orig) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0])  # the last element is maxlen
            region_np = region_np_orig[:-1, :]
            sub_data_len = len(region_np)
            total_len += sub_data_len
            self.acc_len.append(total_len) 
            region_np_list.append(region_np)
            self.handle_list.append(hd5_file)
            batch_list = list(hd5_file.keys())
            sorted_batch_list = sorted(
                batch_list, key=lambda x: int(re.search("_(\d+)", x).group(1))
            )
            index_count = 0
            self.index_map_list.append({})
            for batch_i in sorted_batch_list:
                for read_id in hd5_file[batch_i]:
                    self.index_map_list[-1][index_count] = (hd5_file, batch_i, read_id)
                    index_count += 1
        self.maxlen = max(maxlen_list)
        self.region_np = np.concatenate(region_np_list)
        
    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        for index in self.index_list:
            if index % total_workers == global_worker_id:
                read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
                    index, :
                ].tolist()
                sub_index = self._search_sub_index(index)
                hd5_file, batch_i, read_id = self.index_map_list[sub_index][read_index]
                read = {}
                read["read_id"] = read_id
                read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                read["Seq_to_signal"] = hd5_file[batch_i][read_id][
                    "Seq_to_signal"
                ][()]
                read["digitisation"] = hd5_file[batch_i][read_id].attrs[
                    "digitisation"
                ]
                read["offset"] = hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = hd5_file[batch_i][read_id].attrs["range"]
                read["scale_frompA"] = hd5_file[batch_i][read_id].attrs[
                    "scale_frompA"
                ]
                read["shift_frompA"] = hd5_file[batch_i][read_id].attrs[
                    "shift_frompA"
                ]

                cur = self._get_current(read, (cur_start, cur_end), standardize=True)
                refs = read["Seq"][ref_start:ref_end]

                if self.tokenization == "flipflop":
                    seqs_orig = flipflopfings.flipflop_code(refs, 4)
                    indata = cur.astype(np.float32)
                    seqs = np.full((self.maxlen,), -1)
                elif self.tokenization == "kmer":
                    seqs_orig = refs + 1  # cuz the refs is from 0~3
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)
                else:
                    seqs_orig = refs + 1
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)

                seqs[: len(seqs_orig)] = seqs_orig
                seqlen = len(seqs_orig)
                indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
                yield (indata.astype(np.float32), seqs, seqlen)

    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.data_size
    
    def close_handle(self):
        for hd5_file in self.handle_list:
            hd5_file.close()


class DataSetMulti(IterableDataset):
    """DDP testset for multiple npy index and h5 data, load data on runtime"""
    def __init__(self, h5_list, npy_list, train_size, test_size, tokenization, data_len=5000, first_seed=100):
        super(DataSetMulti).__init__()
        self.data_len = data_len
        self.first_seed = first_seed
        self.h5_list = h5_list
        self.npy_list = npy_list
        assert len(h5_list) == len(npy_list)
        self._hd5_npy_init()
        self.tokenization = tokenization
        self.total_num = self.region_np.shape[0]
        self.test_size = min(self.total_num, test_size)
        self.train_size = min(self.total_num, train_size)
        assert self.train_size > 0
        self.index_list = list(range(self.train_size))
        self.index_val = None
        self.index_train = None
        self.data_size = self.train_size

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.index_list)

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _search_sub_index(self, index):
        # search for the index of sub data
        for i in range(len(self.acc_len)):
            if self.acc_len[i] > index:
                return i
        raise IndexError("This code should not be reached, check _hd5_npy_init!")
            
    def _hd5_npy_init(self):
        self.handle_list = []
        region_np_list = []
        maxlen_list = []
        total_len = 0
        self.acc_len = []  # accumulated len, use to return the index of subdataset, base on np len
        self.index_map_list = []  # indices for each sub dataset
        for i in range(len(self.h5_list)):
            print(f"loading {self.npy_list[i]} in and it's hd5")
            hd5_file = h5py.File(self.h5_list[i], "r")
            region_np_orig = self._load_npy(self.npy_list[i])
            if len(region_np_orig) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0])  # the last element is maxlen
            region_np = region_np_orig[:-1, :]
            sub_data_len = len(region_np)
            total_len += sub_data_len
            self.acc_len.append(total_len) 
            region_np_list.append(region_np)
            self.handle_list.append(hd5_file)
        self.maxlen = max(maxlen_list)
        self.region_np = np.concatenate(region_np_list)
        
    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        for index in self.index_list:
            if index % total_workers == global_worker_id:
                read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
                    index, :
                ].tolist()
                sub_index = self._search_sub_index(index)
                hd5_file = self.handle_list[sub_index]
                id_num =  int(np.floor(read_index  / 10000 ))
                batch_i = f"batch_{id_num}"
                read_id = str(read_index)
                read = {}
                read["read_id"] = read_id
                read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                read["Seq_to_signal"] = hd5_file[batch_i][read_id][
                    "Seq_to_signal"
                ][()]
                read["digitisation"] = hd5_file[batch_i][read_id].attrs[
                    "digitisation"
                ]
                read["offset"] = hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = hd5_file[batch_i][read_id].attrs["range"]
                read["scale_frompA"] = hd5_file[batch_i][read_id].attrs[
                    "scale_frompA"
                ]
                read["shift_frompA"] = hd5_file[batch_i][read_id].attrs[
                    "shift_frompA"
                ]

                cur = self._get_current(read, (cur_start, cur_end), standardize=True)
                refs = read["Seq"][ref_start:ref_end]

                if self.tokenization == "flipflop":
                    seqs_orig = flipflopfings.flipflop_code(refs, 4)
                    indata = cur.astype(np.float32)
                    seqs = np.full((self.maxlen,), -1)
                elif self.tokenization == "kmer":
                    seqs_orig = refs + 1  # cuz the refs is from 0~3
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)

                seqs[: len(seqs_orig)] = seqs_orig
                seqlen = len(seqs_orig)
                indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
                yield (indata.astype(np.float32), seqs, seqlen)

    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.data_size
    
    def close_handle(self):
        for hd5_file in self.handle_list:
            hd5_file.close()


class DataSetMulti_ori2(IterableDataset):
    """DDP testset for multiple npy index and h5 data, load data on runtime"""
    def __init__(self, h5_list, npy_list, train_size, test_size, tokenization, filter_flag=0, data_len=5000, shuffle_seed=100, start_step=0, batch_size=32):
        super(DataSetMulti).__init__()
        self.data_len = data_len
        self.shuffle_seed = shuffle_seed
        self.h5_list = h5_list
        self.npy_list = npy_list
        self.batch_size = batch_size
        assert len(h5_list) == len(npy_list)
        self._hd5_npy_init()
        self.tokenization = tokenization
        self.total_num = self.region_np.shape[0]
        self.test_size = min(self.total_num, test_size)
        self.start_step = start_step
        self.skip_per_gpu = batch_size * start_step
        print(f"Skip {self.skip_per_gpu} per gpu")
        assert self.total_num > self.start_step
        self.train_size = min(self.total_num, train_size)
        assert self.train_size > 0
        self.index_list = list(range(self.train_size))
        self.index_val = None
        self.index_train = None
        self.data_size = self.train_size
        self.filter_flag = filter_flag

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.index_list)

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _search_sub_index(self, index):
        # search for the index of sub data
        for i in range(len(self.acc_len)):
            if self.acc_len[i] > index:
                return i
        raise IndexError("This code should not be reached, check _hd5_npy_init!")
            
    def _hd5_npy_init(self):
        self.handle_list = []
        region_np_list = []
        maxlen_list = []
        total_len = 0
        self.acc_len = []  # accumulated len, use to return the index of subdataset, base on np len
        for i in range(len(self.h5_list)):
            try:
                print(f"loading {self.npy_list[i]} in and it's hd5")
                hd5_file = h5py.File(self.h5_list[i], "r")
                region_np_orig = self._load_npy(self.npy_list[i])
            except:
                continue

            if len(region_np_orig) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0])  # the last element is maxlen
            region_np = region_np_orig[:-1, :]
            sub_data_len = len(region_np)
            total_len += sub_data_len
            self.acc_len.append(total_len) 
            region_np_list.append(region_np)
            self.handle_list.append(hd5_file)
        self.maxlen = max(maxlen_list)
        self.region_np = np.concatenate(region_np_list)
        
    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0

        worker_id_lst = list(range(total_workers))
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        worker_id_in_cur_rank = [i * world_size + rank_id for i in worker_id_lst]

        assert global_worker_id in worker_id_in_cur_rank        
        for index in self.index_list:
            if self.skip_per_gpu > 0:
                for wk_id in worker_id_in_cur_rank:
                    if index % total_workers == wk_id:
                        self.skip_per_gpu -= 1
                continue
            
            if index % total_workers == global_worker_id:
                assert self.skip_per_gpu == 0, f"{self.skip_per_gpu}"
                read_index, cur_start, cur_end, ref_start, ref_end = self.region_np[
                    index, :
                ].tolist() 
                sub_index = self._search_sub_index(index)
                hd5_file = self.handle_list[sub_index]
                id_num =  int(np.floor(read_index  / 10000))
                batch_i = f"batch_{id_num}"
                read_id = str(read_index)                     
                read = {}
                read["read_id"] = read_id
                
                if int(self.filter_flag) == 1 :
                    # ref_len = len(hd5_file[batch_i][read_id]["Seq"][()])
                    # if ref_len > 400 or ref_len < 220:
                    #    continue
                    # coverage = hd5_file[batch_i][read_id].attrs["coverage"]
                    # if float(coverage) < 0.995:
                    #    continue
                    # if hd5_file[batch_i][read_id].attrs["caton_std_200"] < 3.5:
                    #    continue

                    read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    scale_frompA, shift_frompA = med_mad(read["Signal"])
                    read["scale_frompA"] = scale_frompA
                    read["shift_frompA"] = shift_frompA

                else:
                    read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    read["scale_frompA"] = hd5_file[batch_i][read_id].attrs[
                        "scale_frompA"
                    ]
                    read["shift_frompA"] = hd5_file[batch_i][read_id].attrs[
                        "shift_frompA"
                    ]

                read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                read["Seq_to_signal"] = hd5_file[batch_i][read_id][
                    "Seq_to_signal"
                ][()]
                read["digitisation"] = hd5_file[batch_i][read_id].attrs[
                    "digitisation"
                ]
                read["offset"] = hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = hd5_file[batch_i][read_id].attrs["range"]

                cur = self._get_current(read, (cur_start, cur_end), standardize=True)

                refs = read["Seq"][ref_start:ref_end]

                if self.tokenization == "flipflop":
                    seqs_orig = flipflopfings.flipflop_code(refs, 4)
                    indata = cur.astype(np.float32)
                    seqs = np.full((self.maxlen,), -1)
                elif self.tokenization == "kmer":
                    seqs_orig = refs + 1  # cuz the refs is from 0~3
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)
                else:
                    seqs_orig = refs + 1
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)

                seqs[: len(seqs_orig)] = seqs_orig
                seqlen = len(seqs_orig)
                indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
                yield (indata.astype(np.float32), seqs, seqlen)


    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.data_size
    
    def close_handle(self):
        for hd5_file in self.handle_list:
            hd5_file.close()


class DataSetMulti(IterableDataset):
    """DDP testset for multiple npy index and h5 data, load data on runtime"""
    def __init__(self, h5_list, npy_list, train_size, test_size, tokenization, filter_flag=0, data_len=5000, shuffle_seed=100, start_step=0, batch_size=32):
        super(DataSetMulti).__init__()
        self.data_len = data_len
        self.shuffle_seed = shuffle_seed
        self.h5_list = h5_list
        self.npy_list = npy_list
        self.batch_size = batch_size
        assert len(h5_list) == len(npy_list)
        self._hd5_npy_init()
        self.tokenization = tokenization
        self.total_num = self.acc_len[-1]
        self.test_size = min(self.total_num, test_size)
        self.start_step = start_step
        self.skip_per_gpu = batch_size * start_step
        print(f"Skip {self.skip_per_gpu} per gpu")
        assert self.total_num > self.start_step
        self.train_size = min(self.total_num, train_size)
        assert self.train_size > 0
        self.index_list = list(range(self.train_size))
        self.index_val = None
        self.index_train = None
        self.data_size = self.train_size
        self.filter_flag = filter_flag

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.index_list)

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _search_sub_index(self, index):
        # search for the index of sub data
        for i in range(len(self.acc_len)):
            if self.acc_len[i] > index:
                return i
        raise IndexError("This code should not be reached, check _hd5_npy_init!")
            
    def _hd5_npy_init(self):
        self.handle_list = []
        maxlen_list = []
        total_len = 0
        self.acc_len = []  # accumulated len, use to return the index of subdataset, base on np len
        for i in range(len(self.h5_list)):
            try:
                print(f"loading {self.npy_list[i]} in and it's hd5")
                hd5_file = h5py.File(self.h5_list[i], "r")
                region_np_orig = self._load_npy(self.npy_list[i])
            except:
                continue

            if int(len(region_np_orig)) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0].tolist())  # maxlen_list.append(region_np_orig[-1, :][0].tolist())
            # region_np = region_np_orig[:-1, :]
            sub_data_len = region_np_orig[-2, 0].tolist() + 1
            total_len += sub_data_len
            self.acc_len.append(total_len) 
            # region_np_list.append(region_np)
            self.handle_list.append(hd5_file)
            region_np_orig = None
            del region_np_orig
            gc.collect()
        self.maxlen = max(maxlen_list)
        # self.region_np = np.concatenate(region_np_list)
        
    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]

    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current

    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0

        worker_id_lst = list(range(total_workers))
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        worker_id_in_cur_rank = [i * world_size + rank_id for i in worker_id_lst]

        assert global_worker_id in worker_id_in_cur_rank        
        for index in self.index_list:
            if self.skip_per_gpu > 0:
                for wk_id in worker_id_in_cur_rank:
                    if index % total_workers == wk_id:
                        self.skip_per_gpu -= 1
                continue
            
            if index % total_workers == global_worker_id:
                assert self.skip_per_gpu == 0, f"{self.skip_per_gpu}"
                sub_index = self._search_sub_index(index)
                if sub_index == 0:
                    read_index = index
                else:
                    read_index = index - self.acc_len[sub_index - 1]
                hd5_file = self.handle_list[sub_index]
                id_num =  int(np.floor(read_index  / 10000))
                batch_i = f"batch_{id_num}"
                read_id = str(read_index)                     
                read = {}
                read["read_id"] = read_id
                
                if int(self.filter_flag) == 1 :
                    # ---- 1
                    # ref_len = len(hd5_file[batch_i][read_id]["Seq"][()])
                    # if ref_len > 400 or ref_len < 220:
                    #    continue
                    # coverage = hd5_file[batch_i][read_id].attrs["coverage"]
                    # if float(coverage) < 0.995:
                    #    continue
                    # if hd5_file[batch_i][read_id].attrs["caton_std_200"] < 3.5:
                    #    continue

                    # ----- 2
                    read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    scale_frompA, shift_frompA = med_mad(read["Signal"])
                    read["scale_frompA"] = scale_frompA
                    read["shift_frompA"] = shift_frompA
                    
                    # ----- 3
                    # read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()][::-1]
                    # refs = read["Seq"][::-1]
                    
                else:
                    # -----2
                    read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    read["scale_frompA"] = hd5_file[batch_i][read_id].attrs[
                        "scale_frompA"
                    ]
                    read["shift_frompA"] = hd5_file[batch_i][read_id].attrs[
                        "shift_frompA"
                    ]
                    
                    # ---- 3
                    # read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    # refs = read["Seq"]
                
                read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                refs = read["Seq"]
                
                read["Seq_to_signal"] = hd5_file[batch_i][read_id][
                    "Seq_to_signal"
                ][()]
                read["digitisation"] = hd5_file[batch_i][read_id].attrs[
                    "digitisation"
                ]
                read["offset"] = hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = hd5_file[batch_i][read_id].attrs["range"]

                cur = self._get_current(read, standardize=True)

                if self.tokenization == "flipflop":
                    seqs_orig = flipflopfings.flipflop_code(refs, 4)
                    indata = cur.astype(np.float32)
                    seqs = np.full((self.maxlen,), -1)
                elif self.tokenization == "kmer":
                    seqs_orig = refs + 1  # cuz the refs is from 0~3
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)
                else:
                    seqs_orig = refs + 1
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)

                seqs[: len(seqs_orig)] = seqs_orig
                seqlen = len(seqs_orig)
                indata = np.expand_dims(indata, axis=1).transpose((1, 0))  # CT
                yield (indata.astype(np.float32), seqs, seqlen)


    def __iter__(self):
        return self._sample_generator()

    def __len__(self):
        return self.data_size
    
    def close_handle(self):
        for hd5_file in self.handle_list:
            hd5_file.close()


class DataSetMultiV2(IterableDataset):
    """DDP testset for multiple npy index and h5 data, load data on runtime"""
    def __init__(self, training_index_f, val_index_f, h5_list, npy_list, train_size, test_size, total_num, tokenization, filter_flag=0, shuffle_seed=100, start_step=0, batch_size=32):
        super(DataSetMulti).__init__()
        self.index_f_len = total_num
        self.data_len = None
        self.shuffle_seed = shuffle_seed
        self.h5_list = h5_list
        self.npy_list = npy_list
        self.batch_size = batch_size
        assert len(h5_list) == len(npy_list)
        self._hd5_npy_init()
        self.tokenization = tokenization
        self.total_num = self.acc_len[-1]
        self.test_size = min(self.total_num, test_size)
        self.start_step = start_step
        self.skip_per_gpu = batch_size * start_step
        print(f"Skip {self.skip_per_gpu} per gpu")
        assert self.total_num > self.start_step
        self.train_size = min(self.total_num, train_size)
        assert self.train_size > 0
        self.index_val = None
        self.index_train = None
        self.data_size = self.train_size
        self.filter_flag = filter_flag
        self.usage = "training"
        self.training_index_f = training_index_f
        self.val_index_f = val_index_f
        self.index_f = training_index_f

    def switch(self):
        if self.usage == "training":
            self.usage = "validation"
            self.index_f = self.val_index_f
            self.data_size = self.test_size
        else:
            self.usage = "training"
            self.index_f = self.training_index_f
            self.data_size = self.train_size


    def shuffle(self, seed):
        pass  # won't do anything, just placeholder for compatibility

    def _load_npy(self, npy_path):
        return np.load(npy_path)

    def _search_sub_index(self, index):
        # search for the index of sub data
        for i in range(len(self.acc_len)):
            if self.acc_len[i] > index:
                return i
        raise IndexError("This code should not be reached, check _hd5_npy_init!")
            
    def _hd5_npy_init(self):
        self.handle_list = []
        maxlen_list = []
        total_len = 0
        self.acc_len = []  # accumulated len, use to return the index of subdataset, base on np len
        for i in range(len(self.h5_list)):
            try:
                print(f"loading {self.npy_list[i]} in and it's hd5")
                hd5_file = h5py.File(self.h5_list[i], "r")
                region_np_orig = self._load_npy(self.npy_list[i])
            except:
                continue

            if int(len(region_np_orig)) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0].tolist())  # maxlen_list.append(region_np_orig[-1, :][0].tolist())
            if self.data_len:
                assert self.data_len == region_np_orig[-2][2].tolist()
            else:
                self.data_len = region_np_orig[-2][2].tolist()
            sub_data_len = region_np_orig[-2, 0].tolist() + 1
            total_len += sub_data_len
            self.acc_len.append(total_len) 
            self.handle_list.append(hd5_file)
            region_np_orig = None
            del region_np_orig
            gc.collect()
        self.maxlen = max(maxlen_list)
        assert self.acc_len[-1] == self.index_f_len
        
    def _get_signal(self, read, region=None):
        if region is None:
            return read["Signal"]
        a, b = region
        return read["Signal"][a:b]


    def _get_current(self, read, region=None, standardize=True):
        signal = self._get_signal(read, region)
        current = (signal + read["offset"]) * read["range"] / read["digitisation"]
        if standardize:
            current = (current - read["shift_frompA"]) / read["scale_frompA"]
        return current


    def _sample_generator(self):
        total_workers = 0
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            total_workers = 1
            worker_id = 0
        else:
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            
        assert total_workers > 0

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank_id = dist.get_rank()
        else:
            world_size = 1
            rank_id = 0

        worker_id_lst = list(range(total_workers))
        total_workers *= world_size
        global_worker_id = worker_id * world_size + rank_id

        worker_id_in_cur_rank = [i * world_size + rank_id for i in worker_id_lst]

        assert global_worker_id in worker_id_in_cur_rank        
        index_handle = open(self.index_f, "r")
        
        for index in index_handle:
            index = int(index.strip())
            if self.skip_per_gpu > 0:
                for wk_id in worker_id_in_cur_rank:
                    if index % total_workers == wk_id:
                        self.skip_per_gpu -= 1
                continue
            
            if index % total_workers == global_worker_id:
                assert self.skip_per_gpu == 0, f"{self.skip_per_gpu}"
                sub_index = self._search_sub_index(index)
                if sub_index == 0:
                    read_index = index
                else:
                    read_index = index - self.acc_len[sub_index - 1]
                hd5_file = self.handle_list[sub_index]
                id_num =  int(np.floor(read_index  / 10000))
                batch_i = f"batch_{id_num}"
                read_id = str(read_index)                     
                read = {}
                read["read_id"] = read_id
                
                if int(self.filter_flag) == 1 :
                    # ---- 1
                    # ref_len = len(hd5_file[batch_i][read_id]["Seq"][()])
                    # if ref_len > 400 or ref_len < 220:
                    #    continue
                    # coverage = hd5_file[batch_i][read_id].attrs["coverage"]
                    # if float(coverage) < 0.995:
                    #    continue
                    # if hd5_file[batch_i][read_id].attrs["caton_std_200"] < 3.5:
                    #    continue
                    try:
                        before_std = hd5_file[batch_i][read_id].attrs["openpore_before_std"]
                        if float(before_std) > 1.8 or float(before_std) == 0:
                            continue
                    except:
                        continue

                    # ----- 2
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    # scale_frompA, shift_frompA = med_mad(read["Signal"])
                    # read["scale_frompA"] = scale_frompA
                    # read["shift_frompA"] = shift_frompA
                    
                    # ----- 3
                    # read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()][::-1]
                    # refs = read["Seq"][::-1]
                    
                    # ----- 4
                    #signal_ori = hd5_file[batch_i][read_id]["Signal"][()]   
                    #read["Signal"] = subtract_sliding_mean(signal_ori, 500, 1)
                    
                #else:
                    # -----2
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    # read["scale_frompA"] = hd5_file[batch_i][read_id].attrs[
                    #     "scale_frompA"
                    # ]
                    # read["shift_frompA"] = hd5_file[batch_i][read_id].attrs[
                    #     "shift_frompA"
                    # ]

                    # ---- 3
                    # read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                    # read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]
                    # refs = read["Seq"]
                    
                    # ---4
                    #read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]   
        
                read["Signal"] = hd5_file[batch_i][read_id]["Signal"][()]   
                read["Seq"] = hd5_file[batch_i][read_id]["Seq"][()]
                refs = read["Seq"]
                read["scale_frompA"] = hd5_file[batch_i][read_id].attrs["scale_frompA"]
                read["shift_frompA"] = hd5_file[batch_i][read_id].attrs["shift_frompA"]
                read["Seq_to_signal"] = hd5_file[batch_i][read_id]["Seq_to_signal"][()]
                read["digitisation"] = hd5_file[batch_i][read_id].attrs["digitisation"]
                read["offset"] = hd5_file[batch_i][read_id].attrs["offset"]
                read["range"] = hd5_file[batch_i][read_id].attrs["range"]

                # if int(self.filter_flag) == 1:
                #     cur = np.array(normalize_sequence(read["Signal"]))
                # else:
                cur = self._get_current(read, standardize=True)

                if self.tokenization == "flipflop":
                    seqs_orig = flipflopfings.flipflop_code(refs, 4)
                    indata = cur.astype(np.float32)
                    seqs = np.full((self.maxlen,), -1)
                elif self.tokenization == "kmer":
                    seqs_orig = refs + 1  # cuz the refs is from 0~3
                    indata = cur
                    seqs = np.full((self.maxlen,), 0)