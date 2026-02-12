#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型对比评估脚本 - 批量比较不同模型/权重的 basecalling 质量

用法:
    ============ 无需参考基因组 (仅 FASTQ 质量统计) ============

    1. 单个模型 basecall + FASTQ 统计:
       python compare_models.py --mode single \
           --model_dir /workspace/huada/task_results/lstm_ctc_crf_optimized_l10_6x_0204 \
           --weights weights_13.tar \
           --reads /workspace/huada/scall/1.fast5 \
           --device cuda:0

    2. 同一模型多 checkpoint 对比:
       python compare_models.py --mode checkpoints \
           --model_dir /workspace/huada/task_results/lstm_ctc_crf_optimized_l10_6x_0204 \
           --reads /workspace/huada/scall/1.fast5 \
           --device cuda:0 --max_checkpoints 5

    3. 多个模型对比:
       python compare_models.py --mode models \
           --model_dirs \
               /workspace/huada/task_results/lstm_ctc_crf_kmer_8x_0105 \
               /workspace/huada/task_results/lstm_ctc_crf_kmer_layer_10_6x \
               /workspace/huada/task_results/lstm_ctc_crf_optimized_l10_6x_0204 \
           --reads /workspace/huada/scall/1.fast5 \
           --device cuda:0

    4. 已有 FASTQ 文件统计对比 (不需要 basecall):
       python compare_models.py --mode fastq_stats \
           --fastq /workspace/huada/scall/1.fastq /path/to/another.fastq

    ============ 需要参考基因组 (计算 identity_rate) ============

    5. 已有 FASTQ + 参考基因组评估:
       python compare_models.py --mode eval_only \
           --fastq /workspace/huada/scall/1.fastq \
           --ref /workspace/refs/Human.fasta

    上面 1-3 模式加 --ref 参数也会自动计算 identity_rate
"""

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '/workspace/huada/scall/opencall')

from cyclonebasecall.batch_basecall import one_fast5_basecall


# ========== FASTQ Parsing & Stats (无需参考基因组) ==========
def parse_fastq(fastq_path):
    """
    解析 FASTQ 文件，返回每条 read 的序列和质量信息。

    Yields:
        (read_id, sequence, quality_string)
    """
    with open(fastq_path, 'r') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()  # + line
            qual = f.readline().strip()
            read_id = header[1:].split()[0] if header.startswith('@') else header
            yield read_id, seq, qual


def qstring_to_qscores(qstring):
    """将质量字符串转为 Q-score 数组 (Phred+33)"""
    return np.array([ord(c) - 33 for c in qstring], dtype=np.float64)


def mean_qscore(qscores):
    """计算 mean Q-score (使用正确的 Phred 概率平均)"""
    if len(qscores) == 0:
        return 0.0
    mean_err = np.mean(np.power(10.0, -qscores / 10.0))
    return -10.0 * np.log10(max(mean_err, 1e-10))


def compute_n50(lengths):
    """计算 N50"""
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    cumsum = 0
    for length in sorted_lengths:
        cumsum += length
        if cumsum >= total / 2:
            return length
    return 0


def fastq_stats(fastq_path):
    """
    从 FASTQ 文件提取全面的统计指标（无需参考基因组）。

    返回指标:
        - num_reads:      总 reads 数
        - total_bases:    总碱基数
        - mean_read_len:  平均 read 长度
        - median_read_len: 中位 read 长度
        - n50:            N50
        - max_read_len:   最长 read
        - min_read_len:   最短 read
        - mean_qscore:    平均 Q-score (Phred)
        - median_qscore:  中位 Q-score
        - q10_pass_rate:  Q>=10 的 reads 比例
        - q20_pass_rate:  Q>=20 的 reads 比例
        - gc_content:     GC 含量
    """
    read_lengths = []
    read_mean_qscores = []
    base_counts = Counter()

    for read_id, seq, qual in parse_fastq(fastq_path):
        if len(seq) == 0:
            continue
        read_lengths.append(len(seq))
        base_counts.update(seq.upper())

        qs = qstring_to_qscores(qual)
        read_mean_qscores.append(mean_qscore(qs))

    if not read_lengths:
        return {"error": "No reads found in FASTQ file"}

    read_lengths = np.array(read_lengths)
    read_mean_qscores = np.array(read_mean_qscores)
    total_bases = int(np.sum(read_lengths))

    gc = base_counts.get('G', 0) + base_counts.get('C', 0)
    at = base_counts.get('A', 0) + base_counts.get('T', 0)
    gc_content = gc / max(gc + at, 1)

    return {
        "num_reads": len(read_lengths),
        "total_bases": total_bases,
        "mean_read_len": float(np.mean(read_lengths)),
        "median_read_len": float(np.median(read_lengths)),
        "n50": compute_n50(read_lengths.tolist()),
        "max_read_len": int(np.max(read_lengths)),
        "min_read_len": int(np.min(read_lengths)),
        "mean_qscore": float(np.mean(read_mean_qscores)),
        "median_qscore": float(np.median(read_mean_qscores)),
        "q10_pass_rate": float(np.mean(read_mean_qscores >= 10)),
        "q20_pass_rate": float(np.mean(read_mean_qscores >= 20)),
        "gc_content": gc_content,
    }


# ========== Basecalling ==========
def run_basecall(model_dir, reads_path, output_fastq, device="cuda:0",
                 weights_file=None, chunksize=5000, overlap=500, batchsize=64):
    """
    对单个 fast5 文件进行 basecalling，输出 FASTQ 文件。
    """
    weight_managed = False
    default_weight = os.path.join(model_dir, "weights.tar")
    backup_weight = os.path.join(model_dir, "weights.tar.bak")

    if weights_file:
        weight_path = os.path.join(model_dir, weights_file)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        if os.path.exists(default_weight) or os.path.islink(default_weight):
            os.rename(default_weight, backup_weight)
        os.symlink(weight_path, default_weight)
        weight_managed = True
        print(f"  [weights] Using: {weights_file}")

    try:
        params = {
            "model_name": "",
            "reads_directory": reads_path,
            "fastq_path": output_fastq,
            "res_append": False,
            "scale": 1.0,
            "offset": 0.0,
            "model_dir": model_dir,
            "auto_trim_adaptor": 0,
            "n_proc": 1,
            "max_reads_num": 0,
            "device": device,
            "chunksize": chunksize,
            "stride": 5,
            "overlap": overlap,
            "batchsize": batchsize,
            "reads_type": "bgi",
            "is_trim_adaptor": False,
            "min_orig_read_len": 100,
            "min_trim_adaptor_read_len": 100,
            "verbose": False
        }
        one_fast5_basecall(params)
        print(f"  [basecall] FASTQ saved to: {output_fastq}")
    finally:
        if weight_managed:
            if os.path.islink(default_weight):
                os.remove(default_weight)
            if os.path.exists(backup_weight):
                os.rename(backup_weight, default_weight)


# ========== Evaluation with ref (identity_rate) ==========
def run_eval_with_ref(fastq_path, ref_path):
    """评估 FASTQ 的 identity_rate（需要参考基因组）"""
    from cyclonebasecall.evaluation.mapping_and_parsing import get_n_reads_results

    if not os.path.exists(fastq_path):
        raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference genome not found: {ref_path}")

    print(f"  [eval] Mapping to {os.path.basename(ref_path)} ...")
    res = get_n_reads_results(fastq_path, ref_path, mapping_tool="minimap2")
    return res


# ========== Pretty Print ==========
def print_fastq_stats_results(results_list):
    """打印 FASTQ 统计对比表"""
    metrics = [
        ("num_reads",       "Reads",      "d"),
        ("total_bases",     "Total Bases", "d"),
        ("mean_read_len",   "Mean Len",   ".1f"),
        ("median_read_len", "Median Len", ".1f"),
        ("n50",             "N50",        "d"),
        ("mean_qscore",     "Mean Q",     ".2f"),
        ("median_qscore",   "Median Q",   ".2f"),
        ("q10_pass_rate",   "Q10 Pass%",  ".1%"),
        ("q20_pass_rate",   "Q20 Pass%",  ".1%"),
        ("gc_content",      "GC%",        ".1%"),
    ]

    name_width = max(max(len(name) for name, _ in results_list), 15)

    print("\n" + "=" * 120)
    print("FASTQ QUALITY STATISTICS (无需参考基因组)")
    print("=" * 120)

    header = f"{'Model':<{name_width}}"
    for _, display, _ in metrics:
        header += f" | {display:>12}"
    print(header)
    print("-" * len(header))

    for name, res in results_list:
        if "error" in res:
            print(f"{name:<{name_width}} | ERROR: {res['error']}")
            continue
        row = f"{name:<{name_width}}"
        for key, _, fmt in metrics:
            val = res.get(key, "N/A")
            if val == "N/A":
                row += f" | {'N/A':>12}"
            else:
                row += f" | {val:>12{fmt}}"
        print(row)

    print("=" * 120)

    # Interpretation
    print("\n指标说明:")
    print("  Mean Q   = 平均 Phred 质量分数 (越高越好, Q10=90%准确率, Q20=99%, Q30=99.9%)")
    print("  Q10 Pass = Q-score >= 10 的 reads 比例 (模型自信度指标)")
    print("  Q20 Pass = Q-score >= 20 的 reads 比例 (高质量 reads 比例)")
    print("  N50      = 按碱基累积达到50%时的 read 长度")
    print("  GC%      = GC 含量 (人类基因组参考值 ~40-42%)")


def print_identity_results(results_list):
    """打印含 identity_rate 的对比表"""
    metrics = [
        ("identity_rate",        "Identity",    ".4f"),
        ("Modal_Identity_Rate",  "Modal Id",    ".4f"),
        ("Median_Identity_Rate", "Median Id",   ".4f"),
        ("mismatch_rate",        "Mismatch",    ".4f"),
        ("insertion_rate",       "Insertion",    ".4f"),
        ("deletion_rate",        "Deletion",     ".4f"),
        ("mapping_rate",         "Mapping%",    ".4f"),
    ]

    name_width = max(max(len(name) for name, _ in results_list), 15)

    print("\n" + "=" * 110)
    print("IDENTITY RATE RESULTS (需参考基因组比对)")
    print("=" * 110)

    header = f"{'Model':<{name_width}}"
    for _, display, _ in metrics:
        header += f" | {display:>12}"
    print(header)
    print("-" * len(header))

    for name, res in results_list:
        if "error" in res:
            print(f"{name:<{name_width}} | ERROR: {res['error']}")
            continue
        row = f"{name:<{name_width}}"
        for key, _, fmt in metrics:
            val = res.get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f" | {val:>12{fmt}}"
            else:
                row += f" | {'N/A':>12}"
        print(row)

    print("=" * 110)


# ========== Core pipeline: basecall + evaluate ==========
def basecall_and_evaluate(model_dir, reads_path, output_fastq, device,
                          weights_file=None, ref_path=None):
    """
    执行 basecall → FASTQ stats, 如果有 ref 则额外计算 identity_rate。
    返回 (fastq_stats_dict, identity_dict_or_None)
    """
    run_basecall(model_dir, reads_path, output_fastq, device=device,
                 weights_file=weights_file)

    print(f"  [stats] Computing FASTQ statistics ...")
    stats = fastq_stats(output_fastq)

    identity_res = None
    if ref_path:
        try:
            identity_res = run_eval_with_ref(output_fastq, ref_path)
        except Exception as e:
            print(f"  [WARN] Identity eval failed: {e}")

    return stats, identity_res


# ========== Mode Handlers ==========
def mode_single(args):
    """单个模型 + 单个权重"""
    output_dir = args.output_dir or os.path.dirname(args.reads)
    model_name = os.path.basename(args.model_dir)
    weight_name = args.weights.replace(".tar", "") if args.weights else "default"
    fastq_path = os.path.join(output_dir, f"{model_name}_{weight_name}.fastq")
    label = f"{model_name}/{weight_name}"

    print(f"\n>>> Basecalling: {label}")
    stats, identity_res = basecall_and_evaluate(
        args.model_dir, args.reads, fastq_path, args.device,
        weights_file=args.weights, ref_path=args.ref)

    print_fastq_stats_results([(label, stats)])
    if identity_res:
        print_identity_results([(label, identity_res)])

    return [(label, stats, identity_res)]


def mode_checkpoints(args):
    """同一模型多 checkpoint 对比"""
    model_name = os.path.basename(args.model_dir)
    weight_files = sorted(glob.glob(os.path.join(args.model_dir, "weights_*.tar")))

    if not weight_files:
        print(f"No weight files found in {args.model_dir}")
        return []

    if args.max_checkpoints and len(weight_files) > args.max_checkpoints:
        indices = [int(i * (len(weight_files) - 1) / (args.max_checkpoints - 1))
                   for i in range(args.max_checkpoints)]
        weight_files = [weight_files[i] for i in indices]

    output_dir = args.output_dir or os.path.dirname(args.reads)
    stats_list = []
    identity_list = []
    total = len(weight_files)

    for i, wf in enumerate(weight_files):
        wf_name = os.path.basename(wf)
        weight_name = wf_name.replace(".tar", "")
        fastq_path = os.path.join(output_dir, f"{model_name}_{weight_name}.fastq")
        label = f"{model_name}/{weight_name}"

        print(f"\n[{i+1}/{total}] {label}")
        try:
            stats, identity_res = basecall_and_evaluate(
                args.model_dir, args.reads, fastq_path, args.device,
                weights_file=wf_name, ref_path=args.ref)
            stats_list.append((label, stats))
            if identity_res:
                identity_list.append((label, identity_res))
        except Exception as e:
            print(f"  [ERROR] {e}")
            stats_list.append((label, {"error": str(e)}))

    print_fastq_stats_results([(n, s) for n, s in stats_list if "error" not in s])
    if identity_list:
        print_identity_results(identity_list)

    return stats_list


def mode_models(args):
    """多个不同模型对比"""
    output_dir = args.output_dir or os.path.dirname(args.reads)
    stats_list = []
    identity_list = []
    total = len(args.model_dirs)

    for i, model_dir in enumerate(args.model_dirs):
        model_name = os.path.basename(model_dir)

        weight_files = sorted(glob.glob(os.path.join(model_dir, "weights_*.tar")))
        if not weight_files:
            if os.path.exists(os.path.join(model_dir, "weights.tar")):
                wf_name = None
                weight_label = "weights"
            else:
                print(f"\n[{i+1}/{total}] SKIP {model_name}: no weight files found")
                continue
        else:
            def extract_num(path):
                name = os.path.basename(path).replace("weights_", "").replace(".tar", "")
                try:
                    return int(name)
                except ValueError:
                    return -1
            weight_files.sort(key=extract_num)
            wf_name = os.path.basename(weight_files[-1])
            weight_label = wf_name.replace(".tar", "")

        fastq_path = os.path.join(output_dir, f"{model_name}_{weight_label}.fastq")
        label = f"{model_name}/{weight_label}"

        print(f"\n[{i+1}/{total}] {label}")
        try:
            stats, identity_res = basecall_and_evaluate(
                model_dir, args.reads, fastq_path, args.device,
                weights_file=wf_name, ref_path=args.ref)
            stats_list.append((label, stats))
            if identity_res:
                identity_list.append((label, identity_res))
        except Exception as e:
            print(f"  [ERROR] {e}")
            stats_list.append((label, {"error": str(e)}))

    print_fastq_stats_results([(n, s) for n, s in stats_list if "error" not in s])
    if identity_list:
        print_identity_results(identity_list)

    return stats_list


def mode_fastq_stats(args):
    """仅对已有 FASTQ 文件做质量统计（不需要 basecall，不需要 ref）"""
    fastq_files = args.fastq if isinstance(args.fastq, list) else [args.fastq]
    stats_list = []

    for fq in fastq_files:
        name = os.path.basename(fq).replace(".fastq", "")
        print(f"\nAnalyzing: {fq}")
        try:
            stats = fastq_stats(fq)
            stats_list.append((name, stats))
        except Exception as e:
            print(f"  [ERROR] {e}")

    if stats_list:
        print_fastq_stats_results(stats_list)
    return stats_list


def mode_eval_only(args):
    """对已有 FASTQ 做完整评估（FASTQ stats + identity_rate）"""
    fastq_files = args.fastq if isinstance(args.fastq, list) else [args.fastq]
    stats_list = []
    identity_list = []

    for fq in fastq_files:
        name = os.path.basename(fq).replace(".fastq", "")
        print(f"\nEvaluating: {fq}")
        try:
            stats = fastq_stats(fq)
            stats_list.append((name, stats))
            if args.ref:
                identity_res = run_eval_with_ref(fq, args.ref)
                identity_list.append((name, identity_res))
        except Exception as e:
            print(f"  [ERROR] {e}")

    if stats_list:
        print_fastq_stats_results(stats_list)
    if identity_list:
        print_identity_results(identity_list)
    return stats_list


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(
        description="模型对比评估脚本 - 批量比较不同模型/权重的 basecalling 质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("--mode", required=True,
                        choices=["single", "checkpoints", "models", "fastq_stats", "eval_only"],
                        help="运行模式: single(单模型), checkpoints(同模型多checkpoint), "
                             "models(多模型), fastq_stats(仅FASTQ统计), eval_only(完整评估)")
    parser.add_argument("--model_dir", type=str,
                        help="模型目录路径 (single/checkpoints 模式)")
    parser.add_argument("--model_dirs", nargs="+", type=str,
                        help="多个模型目录路径 (models 模式)")
    parser.add_argument("--weights", type=str, default=None,
                        help="指定权重文件名, 如 weights_13.tar (single 模式)")
    parser.add_argument("--reads", type=str, default="/workspace/huada/scall/1.fast5",
                        help="fast5 文件路径")
    parser.add_argument("--ref", type=str, default=None,
                        help="参考基因组 FASTA 文件路径 (可选, 有则额外计算 identity_rate)")
    parser.add_argument("--fastq", nargs="+", type=str,
                        help="已有 FASTQ 文件路径 (fastq_stats/eval_only 模式)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="GPU 设备 (默认 cuda:0)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="FASTQ 输出目录 (默认与 reads 同目录)")
    parser.add_argument("--max_checkpoints", type=int, default=None,
                        help="最多评估多少个 checkpoint (checkpoints 模式)")
    parser.add_argument("--save_json", type=str, default=None,
                        help="保存结果到 JSON 文件")

    args = parser.parse_args()

    if args.mode == "single":
        assert args.model_dir, "--model_dir required"
        results = mode_single(args)
    elif args.mode == "checkpoints":
        assert args.model_dir, "--model_dir required"
        results = mode_checkpoints(args)
    elif args.mode == "models":
        assert args.model_dirs, "--model_dirs required"
        results = mode_models(args)
    elif args.mode == "fastq_stats":
        assert args.fastq, "--fastq required"
        results = mode_fastq_stats(args)
    elif args.mode == "eval_only":
        assert args.fastq, "--fastq required"
        results = mode_eval_only(args)
    else:
        parser.print_help()
        return

    # Save to JSON
    if args.save_json and results:
        save_data = {}
        for item in results:
            name = item[0]
            res = item[1] if len(item) >= 2 else {}
            if "error" not in res:
                save_data[name] = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in res.items()}
        with open(args.save_json, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.save_json}")


if __name__ == "__main__":
    main()
