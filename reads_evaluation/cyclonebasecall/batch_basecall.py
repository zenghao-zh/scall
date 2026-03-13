import sys
import os
import numpy as np
from tqdm import tqdm
from time import perf_counter
from datetime import timedelta
from cyclonebasecall.bonito.reader import Reader
from cyclonebasecall.bonito.io import Hybrid_Writer, Writer, biofmt
from cyclonebasecall.bonito.multiprocessing import process_cancel
from cyclonebasecall.bonito.util import column_to_set, load_symbol, load_model, load_quant_model, init
from cyclonebasecall.util import get_bgi_reads
from cyclonebasecall.conf import PROJ_DIR, DOWNLOAD_URL
from cyclonebasecall.util import DownloadZipFile


def one_fast5_basecall(params):
    # 0. download models
    if os.path.exists(params["model_dir"]):
        params["model_directory"] = params["model_dir"]
    else:
        model_name = params["model_name"]
        dzip = DownloadZipFile(
            url=DOWNLOAD_URL,
            zip_file_name="{}.zip".format(model_name),
            des_dir="{}/models".format(PROJ_DIR)
        )
        dzip.download()
        params["model_directory"] = "{}/models/{}".format(PROJ_DIR, model_name)

    # 1. remove old fastq
    fastq_path = params["fastq_path"]
    if not params["res_append"]:
        if os.path.exists(fastq_path):
            os.remove(fastq_path)
            
    if params["verbose"]:
        h5_path = params["fastq_path"].replace('.fastq', '.h5')
        if os.path.exists(h5_path):
            os.remove(h5_path)

    init(seed=25, device=params["device"])
    fmt = biofmt()

    # Use quantized model loading if io_quant_path and act_scales_path are provided
    io_quant_path = params.get("io_quant_path")
    act_scales_path = params.get("act_scales_path")
    if io_quant_path and act_scales_path:
        use_bf16 = params.get("bf16", False)
        sys.stderr.write(f"> loading quantized model (FakeQuant, bf16={use_bf16})\n")
        model = load_quant_model(
            params["model_directory"],
            device=params["device"],
            io_quant_path=io_quant_path,
            act_scales_path=act_scales_path,
            bf16=use_bf16,
        )
    else:
        model = load_model(
            params["model_directory"],
            device=params["device"],
            chunksize=params["chunksize"],
            overlap=params["overlap"],
            batchsize=params["batchsize"],
            quantize=None,
            use_koi=True,
        )
    basecall = load_symbol(params["model_directory"], "basecall")

    groups = []
    num_reads = None
    if params["reads_type"] == "ont":
        try:
            reader = Reader(params["reads_directory"], recursive=False)
            sys.stderr.write("> reading %s\n" % reader.fmt)
        except FileNotFoundError:
            sys.stderr.write("> error: no suitable files found in %s\n" % params["reads_directory"])
            exit(1)

        reads = reader.get_reads(
            params["reads_directory"], n_proc=8, recursive=False,
            read_ids=column_to_set(filename=None), skip=False,
            cancel=process_cancel()
        )
    else:
        reads = get_bgi_reads(
            params["reads_directory"], n_proc=1, is_trim_adaptor=params["is_trim_adaptor"], auto_trim_adaptor=params["auto_trim_adaptor"],
            min_orig_read_len = params["min_orig_read_len"], min_trim_adaptor_read_len = params["min_trim_adaptor_read_len"])


    # next(reads)
    results = basecall(
        model, reads, reverse=False,
        batchsize=params["batchsize"],
        chunksize=params["chunksize"],
        overlap=params["overlap"],
        scale=params["scale"],
        offset=params["offset"],
        model_stride=params["stride"],
        decode_method=params.get("decode_method", "beam_search"),
    )
    writer = Writer(
        fmt.mode,
        tqdm(results, desc="> calling", unit=" reads", leave=False,
                        total=num_reads, smoothing=0, ascii=True, ncols=100),
        aligner=None,
        group_key=params["model_directory"],
        verbose=params.get("verbose", False),
        ref_fn=None,
        groups=groups,
        fastq_path=fastq_path
    )

    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)
    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")



def prepare_model(params):
    # 0. download models
    if os.path.exists(params["model_dir"]):
        params["model_directory"] = params["model_dir"]
    else:
        model_name = params["model_name"]
        dzip = DownloadZipFile(
            url=DOWNLOAD_URL,
            zip_file_name="{}.zip".format(model_name),
            des_dir="{}/models".format(PROJ_DIR)
        )
        dzip.download()
        params["model_directory"] = "{}/models/{}".format(PROJ_DIR, model_name)

    init(seed=25, device=params["device"])

    model = load_model(
        params["model_directory"],
        device=params["device"],
        chunksize=params["chunksize"],
        overlap=params["overlap"],
        batchsize=params["batchsize"],
        quantize=None,
        use_koi=True,
    )
    return model, params["model_directory"]


def call(model, params):
    # 1. remove old fastq
    fastq_path = params["fastq_path"]
    basecall_h5 = params["basecall_h5"]
    run_id = params["run_id"]
    if not params["res_append"]:
        if os.path.exists(fastq_path):
            os.remove(fastq_path)
    basecall = load_symbol(params["model_directory"], "basecall")
    fmt = biofmt()
    groups = []
    num_reads = None
    if params["reads_type"] == "ont":
        try:
            reader = Reader(params["reads_directory"], recursive=False)
            sys.stderr.write("> reading %s\n" % reader.fmt)
        except FileNotFoundError:
            sys.stderr.write("> error: no suitable files found in %s\n" % params["reads_directory"])
            exit(1)

        reads = reader.get_reads(
            params["reads_directory"], n_proc=8, recursive=False,
            read_ids=column_to_set(filename=None), skip=False,
            cancel=process_cancel()
        )
    else:
        reads = get_bgi_reads(
            params["reads_directory"], n_proc=1, is_trim_adaptor=params["is_trim_adaptor"], auto_trim_adaptor=params["auto_trim_adaptor"],
            min_orig_read_len = params["min_orig_read_len"], min_trim_adaptor_read_len = params["min_trim_adaptor_read_len"])

        # next(reads)
        results = basecall(
            model, reads, reverse=False,
            batchsize=params["batchsize"],
            chunksize=params["chunksize"],
            overlap=params["overlap"],
            scale=params["scale"],
            offset=params["offset"]
        )
        writer = Hybrid_Writer(
            tqdm(results, desc="> calling", unit=" reads", leave=False,
                           total=num_reads, smoothing=0, ascii=True, ncols=100),
            fastq_out=fastq_path,
            h5_path=basecall_h5,
            run_id = run_id
            
        )
    
        t0 = perf_counter()
        writer.start()
        writer.join()
        duration = perf_counter() - t0
        num_samples = sum(writer.log)
        sys.stderr.write("> completed reads: %s\n" % len(writer.log))
        sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
        sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
        sys.stderr.write("> done\n")
