"""
将 HDF5 数据转换为 memmap 格式 - 流式处理版本，避免内存爆炸

Usage:
    python convert_hdf5_to_pt.py \
        --input_dir /path/to/train \
        --output_dir /path/to/train_mmap \
        --num_workers 8
"""

import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pro_dir)

import argparse
import glob
import h5py
import numpy as np
import torch
from tqdm import tqdm
import json
import multiprocessing as mp


def get_current(signal, offset, range_, digitisation, shift_frompA, scale_frompA):
    """标准化电流信号"""
    current = (signal + offset) * range_ / digitisation
    current = (current - shift_frompA) / scale_frompA
    return current


def count_samples_in_file(hd5_path):
    """统计单个文件的样本数"""
    try:
        npy_path = hd5_path.replace('.hd5', '.npy')
        dat_npy = np.load(npy_path)
        data_rows = dat_npy[:-1]
        mask = data_rows[:, 5] == 0
        return int(mask.sum())
    except:
        return 0


def process_file_to_arrays(args):
    """处理单个 HDF5 文件，返回数组"""
    hd5_path, signal_len, maxlen = args
    
    signals = []
    seqs = []
    seqlens = []
    
    try:
        npy_path = hd5_path.replace('.hd5', '.npy')
        dat_npy = np.load(npy_path)
        data_rows = dat_npy[:-1]
        mask = data_rows[:, 5] == 0
        filtered_data = data_rows[mask]
        
        if len(filtered_data) == 0:
            return None
        
        hd5_file = h5py.File(hd5_path, 'r')
        per_num = hd5_file.attrs['batch_size']
        
        for row in filtered_data:
            read_index = int(row[0])
            cur_start, cur_end = int(row[1]), int(row[2])
            ref_start, ref_end = int(row[3]), int(row[4])
            
            try:
                batch_num = int(read_index // per_num)
                read = hd5_file[f'batch_{batch_num}'][str(read_index)]
                
                signal = read['Signal'][cur_start:cur_end]
                cur = get_current(
                    signal,
                    read.attrs['offset'],
                    read.attrs['range'],
                    read.attrs['digitisation'],
                    read.attrs['shift_frompA'],
                    read.attrs['scale_frompA']
                )
                
                refs = read['Seq'][ref_start:ref_end]
                seqs_orig = refs + 1
                
                # 填充
                seq = np.zeros(maxlen, dtype=np.int64)
                seq[:len(seqs_orig)] = seqs_orig
                
                indata = np.zeros((1, signal_len), dtype=np.float32)
                actual_len = min(len(cur), signal_len)
                indata[0, :actual_len] = cur[:actual_len]
                
                signals.append(indata)
                seqs.append(seq)
                seqlens.append(len(seqs_orig))
                
            except:
                continue
        
        hd5_file.close()
        
        if len(signals) == 0:
            return None
            
        return (
            np.stack(signals, axis=0),
            np.stack(seqs, axis=0),
            np.array(seqlens, dtype=np.int64)
        )
        
    except Exception as e:
        return None


def get_maxlen(input_dir):
    """获取 maxlen"""
    npy_files = glob.glob(f'{input_dir}/*.npy')
    maxlen = 0
    for npy_path in npy_files[:20]:
        try:
            dat_npy = np.load(npy_path)
            if dat_npy[-1, 0] > maxlen:
                maxlen = int(dat_npy[-1, 0])
        except:
            pass
    return maxlen


def convert_streaming(input_dir, output_dir, signal_len=5000, num_workers=8):
    """流式转换 - 边处理边写入，避免内存爆炸"""
    os.makedirs(output_dir, exist_ok=True)
    
    hd5_paths = sorted(glob.glob(f'{input_dir}/*.hd5'))
    if len(hd5_paths) == 0:
        print(f"No HDF5 files found in {input_dir}")
        return
    
    print(f"Found {len(hd5_paths)} HDF5 files")
    
    # 获取 maxlen
    print("Getting maxlen...")
    maxlen = get_maxlen(input_dir)
    print(f"maxlen: {maxlen}")
    
    # 第一遍：统计总样本数
    print("Counting total samples...")
    with mp.Pool(num_workers) as pool:
        counts = list(tqdm(
            pool.imap(count_samples_in_file, hd5_paths),
            total=len(hd5_paths),
            desc="Counting"
        ))
    total_samples = sum(counts)
    print(f"Total samples: {total_samples}")
    
    # 创建 memmap 文件
    signals_path = os.path.join(output_dir, 'signals.npy')
    seqs_path = os.path.join(output_dir, 'seqs.npy')
    seqlens_path = os.path.join(output_dir, 'seqlens.npy')
    
    print("Creating memmap files...")
    signals_mmap = np.lib.format.open_memmap(
        signals_path, mode='w+', dtype=np.float32, shape=(total_samples, 1, signal_len)
    )
    seqs_mmap = np.lib.format.open_memmap(
        seqs_path, mode='w+', dtype=np.int64, shape=(total_samples, maxlen)
    )
    seqlens_mmap = np.lib.format.open_memmap(
        seqlens_path, mode='w+', dtype=np.int64, shape=(total_samples,)
    )
    
    # 第二遍：流式处理并写入
    print(f"Processing with {num_workers} workers (streaming write)...")
    args_list = [(hd5_path, signal_len, maxlen) for hd5_path in hd5_paths]
    
    write_idx = 0
    with mp.Pool(num_workers) as pool:
        for result in tqdm(
            pool.imap(process_file_to_arrays, args_list),
            total=len(args_list),
            desc="Converting"
        ):
            if result is None:
                continue
            
            sigs, seqs, lens = result
            n = len(lens)
            
            signals_mmap[write_idx:write_idx + n] = sigs
            seqs_mmap[write_idx:write_idx + n] = seqs
            seqlens_mmap[write_idx:write_idx + n] = lens
            
            write_idx += n
    
    # 截断到实际大小
    actual_samples = write_idx
    print(f"Actual samples written: {actual_samples}")
    
    del signals_mmap
    del seqs_mmap
    del seqlens_mmap
    
    if actual_samples < total_samples:
        print("Truncating files...")
        signals = np.load(signals_path, mmap_mode='r')[:actual_samples].copy()
        seqs = np.load(seqs_path, mmap_mode='r')[:actual_samples].copy()
        seqlens = np.load(seqlens_path, mmap_mode='r')[:actual_samples].copy()
        np.save(signals_path, signals)
        np.save(seqs_path, seqs)
        np.save(seqlens_path, seqlens)
        del signals, seqs, seqlens
    
    # 保存元数据
    metadata = {
        'format': 'memmap',
        'total_samples': actual_samples,
        'signal_len': signal_len,
        'maxlen': maxlen,
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # 文件大小
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) 
                    for f in ['signals.npy', 'seqs.npy', 'seqlens.npy']) / (1024**3)
    print(f"Total file size: {total_size:.2f} GB")
    print(f"Done! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 to memmap (streaming)')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--signal_len', type=int, default=5000)
    
    args = parser.parse_args()
    
    convert_streaming(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        signal_len=args.signal_len,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
