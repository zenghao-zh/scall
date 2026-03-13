
import sys
import os
import numpy as np
from tqdm import tqdm
from time import perf_counter
from datetime import timedelta
from cyclonebasecall.bonito.reader import Reader
from cyclonebasecall.bonito.io import Writer, biofmt
from cyclonebasecall.bonito.multiprocessing import process_cancel
from cyclonebasecall.bonito.util import column_to_set, load_symbol, load_model, init
from cyclonebasecall.util import get_bgi_reads
from cyclonebasecall.conf import PROJ_DIR, DOWNLOAD_URL
from cyclonebasecall.util import DownloadZipFile
import json


# get model json
model_json_dir = "{}/models".format(PROJ_DIR)
djson = DownloadZipFile(url=DOWNLOAD_URL, zip_file_name="model.json", des_dir=model_json_dir, force=True)
djson.download()
with open("{}/model.json".format(model_json_dir), "r")as f:
    model_dict = json.load(f)
# loading model
models_loaded_dict = {}
for model_name_, model_attr in model_dict.items():
    dzip = DownloadZipFile(
        url=DOWNLOAD_URL,
        zip_file_name="{}.zip".format(model_name_),
        des_dir="{}/models".format(PROJ_DIR)
    )
    dzip.download()
    models_loaded_dict[model_name_] = load_model(
            "{}/models/{}".format(PROJ_DIR, model_name_),
            device="cuda:0",
            weights=0,
            quantize=None,
            use_koi=True,
        )


def one_fast5_basecall(params):
    # 1. remove old fastq
    fastq_path = params["fastq_path"]
    if not params["res_append"]:
        if os.path.exists(fastq_path):
            os.remove(fastq_path)

    init(seed=25, device=params["device"])
    fmt = biofmt()

    model = models_loaded_dict[params["model_name"]]
    params["model_directory"] = "{}/models/{}".format(PROJ_DIR, params["model_name"])
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
            params["reads_directory"], n_proc=params["n_proc"], is_trim_adaptor=params["is_trim_adaptor"],
            min_orig_read_len = params["min_orig_read_len"], min_trim_adaptor_read_len = params["min_trim_adaptor_read_len"])

    try:
        next(reads)
        results = basecall(
            model, reads, reverse=False,
            batchsize=params["batchsize"],
            chunksize=params["chunksize"],
            overlap=params["overlap"],
            scale=model_dict[params["model_name"]]["base_calibration"]["scale"],
            offset=model_dict[params["model_name"]]["base_calibration"]["offset"]
        )
        writer = Writer(
            fmt.mode,
            tqdm(results, desc="> calling", unit=" reads", leave=False,
                           total=num_reads, smoothing=0, ascii=True, ncols=100),
            aligner=None,
            group_key=params["model_directory"],
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
    except Exception as e:
        raise IOError(e)

