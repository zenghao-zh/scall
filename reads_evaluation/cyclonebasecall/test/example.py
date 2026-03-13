import argparse
import os
import sys
from glob import glob
from cyclonebasecall.batch_basecall import one_fast5_basecall


def main():
    parser = argparse.ArgumentParser(description="Basecall a fast5 file")
    parser.add_argument("--reads", default="/workspace/huada/scall/1.fast5", help="fast5 file path")
    parser.add_argument("--fastq", default="/workspace/huada/scall/1.fastq", help="output fastq path")
    parser.add_argument("--model_dir", default="/workspace/huada/task_results/lstm_ctc_crf_optimized_l10_6x_0204")
    parser.add_argument("--weights", default=None, help="weight file name, e.g. weights_59.tar")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunksize", type=int, default=4796)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--decode", default="beam_search", choices=["beam_search", "viterbi"],
                        help="Decode method: beam_search (koi) or viterbi")
    parser.add_argument("--quant", action="store_true", default=False,
                        help="Load quantized model with FakeQuant layers")
    parser.add_argument("--io_quant", default=None,
                        help="Path to io_quant .pth file (auto-detected if not set)")
    parser.add_argument("--act_scales", default=None,
                        help="Path to act_scales .pth file (auto-detected if not set)")
    args = parser.parse_args()

    # 处理 weights 软链接
    default_weight = os.path.join(args.model_dir, "weights.tar")
    backup_weight = os.path.join(args.model_dir, "weights.tar.bak")
    weight_managed = False

    if args.weights and args.weights != "weights.tar":
        weight_path = os.path.join(args.model_dir, args.weights)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")
        if os.path.exists(default_weight) or os.path.islink(default_weight):
            os.rename(default_weight, backup_weight)
        os.symlink(weight_path, default_weight)
        weight_managed = True
        print(f"[weights] Using: {args.weights}")

    # Auto-detect io_quant and act_scales paths if --quant is set
    io_quant_path = args.io_quant
    act_scales_path = args.act_scales
    if args.quant and not io_quant_path:
        candidates = glob(os.path.join(args.model_dir, "*_io_quant.pth"))
        if candidates:
            io_quant_path = candidates[0]
            print(f"[quant] Auto-detected io_quant: {io_quant_path}")
        else:
            raise FileNotFoundError(f"No *_io_quant.pth found in {args.model_dir}")
    if args.quant and not act_scales_path:
        candidates = glob(os.path.join(args.model_dir, "*_act_scales.pth"))
        if candidates:
            act_scales_path = candidates[0]
            print(f"[quant] Auto-detected act_scales: {act_scales_path}")
        else:
            raise FileNotFoundError(f"No *_act_scales.pth found in {args.model_dir}")

    try:
        params = {
            "model_name": "",
            "reads_directory": args.reads,
            "fastq_path": args.fastq,
            "res_append": False,
            "scale": 1.0,
            "offset": 0.0,
            "model_dir": args.model_dir,
            "auto_trim_adaptor": 0,
            "n_proc": 1,
            "max_reads_num": 0,
            "device": args.device,
            "chunksize": args.chunksize,
            "stride": 5,
            "overlap": args.overlap,
            "batchsize": args.batchsize,
            "reads_type": "bgi",
            "is_trim_adaptor": False,
            "min_orig_read_len": 100,
            "min_trim_adaptor_read_len": 100,
            "verbose": False,
            "decode_method": args.decode,
            "io_quant_path": io_quant_path if args.quant else None,
            "act_scales_path": act_scales_path if args.quant else None,
        }
        one_fast5_basecall(params)
        print(f"[done] FASTQ saved: {args.fastq}")
    finally:
        if weight_managed:
            if os.path.islink(default_weight):
                os.remove(default_weight)
            if os.path.exists(backup_weight):
                os.rename(backup_weight, default_weight)


if __name__ == "__main__":
    main()
