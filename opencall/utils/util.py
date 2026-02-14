"""
opencall utils
"""
from opencall.utils.mylr import linear_warmup_cosine_decay, linear_warmup_cosine_decay_for_continue
from opencall.data_loader.data import TrainingDataSet2,TrainingDataSet3, TestingDataSet2, DataSetMulti, DataSetMultiV2
import os
import os.path as osp
import re
import random
from itertools import groupby
from operator import itemgetter
from importlib import import_module
from collections import defaultdict, OrderedDict
import time
import toml
import torch
import parasail
import numpy as np
from torch.cuda import get_device_capability
import shutil
import torch.distributed as dist

try:
    from claragenomics.bindings import cuda
    from claragenomics.bindings.cudapoa import CudaPoaBatch
except ImportError:
    pass


__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "data")
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__dir__, "models/configs")

split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
default_data = os.path.join(__data__, "dna_r9.4.1")
default_config = os.path.join(__configs__, "dna_r9.4.1@v3.1.toml")


def load_symbol(config_dict, symbol):
    imported = import_module(config_dict["model"]["package"])
    return getattr(imported, symbol)


def load_model(model_dir, weight_path, device, half=None):
    """
    Load a model from disk
    """
    # model_name = os.path.basename(model_dir)
    device = torch.device(device)
    config = toml.load(os.path.join(model_dir, "config.toml"))
    # weights = os.path.join(model_dir, '{}.params'.format(model_name))
    # weights = os.path.join(model_dir, '{}.params'.format(model_name))
    # weights = os.path.join(model_dir, 'weights_10.tar')

    basecall_params = config.get("basecaller", {})
    config["basecaller"] = basecall_params

    Model = load_symbol(config, "Model")
    model = Model(config)
    state_dict = torch.load(weight_path, map_location=device)
    state_dict = {
        k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()
    }
    model.load_state_dict(state_dict)
    if half:
        model = model.half()
    model.eval()
    model.to(device)
    return model


def model_eval(dataloader, model_dir, weight_path, is_half, device):
    # data = loading_npy_data(npy_dir)
    # dataloader = DataLoader(
    #     dataset=data,
    #     batch_size=batchsize,
    #     num_workers=4, pin_memory=True, shuffle=False
    # )

    model = load_model(model_dir=model_dir, weight_path=weight_path, device=device, half=is_half)
    targets, seqs = [],[]
    t0 = time.perf_counter()
    total_samples = 0
    min_coverage = 0.95
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=min_coverage)
    with torch.no_grad():
        for num, (data, target, *_) in enumerate(dataloader):
            # print(num)
            targets.extend(torch.unbind(target, 0))
            if is_half:
                data = data.type(torch.float16).to(device)
            else:
                data = data.to(device)
            total_samples += data.shape[0] * data.shape[2]
            log_probs = model(data)
            if hasattr(model, "decode_batch"):
                start_time = time.perf_counter()
                seqs.extend(model.decode_batch(log_probs))
                end_time = time.perf_counter()
                # print(f"Decode batch time: {end_time - start_time} seconds for {len(seqs)} sequences")
            else:
                start_time = time.perf_counter()
                seqs.extend([model.decode(p) for p in permute(log_probs, "TNC", "NTC")])
                end_time = time.perf_counter()
                print(f"Decode time: {end_time - start_time} seconds for {len(seqs)} sequences")
    duration = time.perf_counter() - t0

    refs = [decode_ref(target, model.alphabet, encoder_only=True) for target in targets]
    accuracies = [
        accuracy_with_cov(ref, seq) if len(seq) else 0.0 for ref, seq in zip(refs, seqs)
    ]
    res_bases_num = np.sum([len(seq) for seq in seqs])
    chunks_num = len(refs)

    res_mean = np.mean(accuracies)
    res_median = np.median(accuracies)
    res_speed = total_samples / duration
    res_base_speed = res_bases_num / duration
    print("* mean      %.2f%%" % np.mean(accuracies))
    print("* median    %.2f%%" % np.median(accuracies))
    print("* time      %.2f" % duration)
    print("* samples/s:    %.2E" % (total_samples / duration))
    print("* bases/s:    %.2E" % res_base_speed)
    print("* val chunks num:    %.0f" % chunks_num)
    return res_mean, res_median, duration, res_speed, res_base_speed, chunks_num


def loading_hd5_in_one_dir(args, data_dir, batch_size, gen_dataloader=True, encoder_only=False):
    data_name = os.path.basename(data_dir)
    # if not os.path.exists("{}/{}_train.npy".format(data_dir, data_name)):
    #     print("split idx file ...")
    #     parmas_dict = {
    #         "data_name": data_name,
    #         "index_npy_path": "/workspace/datasets/{}/{}.npy".format(data_name, data_name),
    #         "validation_size": 20000,
    #     }
    #     split_idx_file(parmas_dict)

    # 1) loading hd5 and get train/val data
    print("loading hd5 ...")
    assert os.path.basename(data_dir) == 'train'
    
    if encoder_only:
        from opencall.data_loader.data import TrainingDataSet3_Encoder
        train_dataset_orig = TrainingDataSet3_Encoder(data_dir, tokenization=args.tokenization)
        test_dataset = TrainingDataSet3_Encoder(os.path.join(os.path.dirname(data_dir), 'val'), tokenization=args.tokenization)
    else:
        train_dataset_orig = TrainingDataSet3(data_dir, tokenization=args.tokenization)
        test_dataset = TrainingDataSet3(os.path.join(os.path.dirname(data_dir), 'val'), tokenization=args.tokenization)

    print("train_dataset: {}".format(len(train_dataset_orig)))
    print("test_dataset: {}".format(len(test_dataset)))
    # 2) trim training data
    if args.limit_train_size > 0:
        if args.limit_train_size < len(train_dataset_orig):
            train_dataset, _ = torch.utils.data.random_split(
                train_dataset_orig, [args.limit_train_size, len(train_dataset_orig)-args.limit_train_size])
        else:
            train_dataset = train_dataset_orig
    else:
        train_dataset = train_dataset_orig
    return train_dataset, test_dataset


def loading_hd5_two_time(args, data_dir, batch_size, gen_dataloader=True):
    data_name = os.path.basename(data_dir)
    # if not os.path.exists("{}/{}_train.npy".format(data_dir, data_name)):
    #     print("split idx file ...")
    #     parmas_dict = {
    #         "data_name": data_name,
    #         "index_npy_path": "/workspace/datasets/{}/{}.npy".format(data_name, data_name),
    #         "validation_size": 20000,
    #     }
    #     split_idx_file(parmas_dict)

    # 1) loading hd5 and get train/val data
    print("loading hd5 ...")
    if os.path.basename(data_dir) == 'train':
        train_dataset_orig = TrainingDataSet3(data_dir, tokenization=args.tokenization)
        test_dataset = TrainingDataSet3(os.path.join(os.path.dirname(data_dir), 'val'), tokenization=args.tokenization)
    else:
        hd5_path = "{}/{}.hd5".format(data_dir, os.path.basename(data_dir))
        index_train_path = "{}/{}_train.npy".format(data_dir, os.path.basename(data_dir))
        index_val_path = "{}/{}_val.npy".format(data_dir, os.path.basename(data_dir))
        train_dataset_orig = TrainingDataSet2(hd5_path, index_train_path, tokenization=args.tokenization)
        test_dataset = TrainingDataSet2(hd5_path, index_val_path, tokenization=args.tokenization)
    print("train_dataset: {}".format(len(train_dataset_orig)))
    print("test_dataset: {}".format(len(test_dataset)))
    # 2) trim training data
    if args.limit_train_size > 0:
        if args.limit_train_size < len(train_dataset_orig):
            train_dataset, _ = torch.utils.data.random_split(
                train_dataset_orig, [args.limit_train_size, len(train_dataset_orig)-args.limit_train_size])
        else:
            train_dataset = train_dataset_orig
    else:
        train_dataset = train_dataset_orig
    # 3) gen dataloader
    if gen_dataloader:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True )
        valid_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        return train_loader, valid_loader
    else:
        return train_dataset, test_dataset


def load_multi_hd5(args, sub_datasets, batch_size, start_step=0):
    # 1. only have one npy and one h5 in a single directory
    h5_list = []
    npy_list = []
    for data_dir in sub_datasets:
        hour = osp.basename(data_dir)
        run_id = osp.basename(osp.dirname(data_dir))
        system = osp.basename(osp.dirname(osp.dirname(data_dir)))
        hd5_path = f"{data_dir}/{system}_{run_id}_{hour}.hd5"
        index_train_path = f"{data_dir}/{system}_{run_id}_{hour}.npy"
        h5_list.append(hd5_path)
        npy_list.append(index_train_path)
        
    train_size = args.limit_train_size if hasattr(args, "limit_train_size") else 1000000000000
    val_size = args.val_size if hasattr(args, "val_size") else 2000  
    dataset = DataSetMulti(h5_list, npy_list, train_size=train_size, test_size=val_size, filter_flag=args.filter_flag,
                           tokenization=args.tokenization, start_step=start_step, batch_size=batch_size, shuffle_seed=args.shuffle_seed)
    train_size = len(dataset)
    # assert train_size > val_size, "Traing set should be larger than validation set"
    print("train_dataset: {}".format(train_size))
    print("test_dataset: {}".format(val_size))
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=args.num_wk, pin_memory=True
    )

    return train_loader


def load_multi_hd5_v2(args, sub_datasets, batch_size, workdir, start_step=0):

    def _init_index():
        import gc
        import numpy as np
        training_index_f = os.path.join(workdir, "training.idx")
        val_index_f = os.path.join(workdir, "validation.idx")
        index_done = os.path.join(workdir, "index.done")
        maxlen_list = []
        total_len = 0
        acc_len = []
        for i in range(len(npy_list)):
            try:
                print(f"loading {npy_list[i]}")
                region_np_orig = np.load(npy_list[i])
            except:
                continue

            if int(len(region_np_orig)) <= 1:
                continue
            maxlen_list.append(region_np_orig[-1, :][0].tolist())
            sub_data_len = region_np_orig[-2, 0].tolist() + 1
            total_len += sub_data_len
            acc_len.append(total_len) 

            region_np_orig = None
            del region_np_orig
            gc.collect()
        max_len = max(maxlen_list)
        data_len = acc_len[-1]
        try:
            dist.get_rank()
            ddp_flag = True

        except:
            ddp_flag = False
            
        if not ddp_flag or dist.get_rank() == 0:    
            total_index = list(range(data_len))
            random.seed(1)  # epoch = 1 
            random.shuffle(total_index)
        
        train_size = args.limit_train_size if hasattr(args, "limit_train_size") else 1000000000000
        val_size = args.val_size if hasattr(args, "val_size") else 2000
        train_size = min(data_len, train_size)
        val_size = min(data_len, val_size)  
        
        if not ddp_flag or dist.get_rank() == 0:
            with open(val_index_f, 'w') as f:
                for i in range(data_len - val_size, data_len):
                    f.write(f"{total_index[i]}\n")
            
            with open(training_index_f, 'w') as f:
                for i in range(train_size):
                    f.write(f"{total_index[i]}\n")
            
            with open(index_done, 'w') as f:
                f.write("Indeices saving was finished !!!")
        else:
            t_start = time.time()
            while not os.path.exists(index_done):
                time.sleep(1)
                t_pass = time.time() - t_start
                print(f"Now waiting for rank0 to prepare index, it has already been waited for {t_pass} secs.")
        
        return train_size, val_size, max_len, training_index_f, val_index_f, data_len

    npy_list = []
    h5_list = []
    for data_dir in sub_datasets:
        hour = osp.basename(data_dir)
        run_id = osp.basename(osp.dirname(data_dir))
        system = osp.basename(osp.dirname(osp.dirname(data_dir)))
        hd5_path = f"{data_dir}/{system}_{run_id}_{hour}.hd5"
        index_train_path = f"{data_dir}/{system}_{run_id}_{hour}.npy"
        h5_list.append(hd5_path)
        npy_list.append(index_train_path)    

    train_size, val_size, max_len, training_index_f, val_index_f, total_len = _init_index()

    dataset = DataSetMultiV2(training_index_f, val_index_f, h5_list, npy_list, train_size, val_size, total_len, filter_flag=args.filter_flag,
                           tokenization=args.tokenization, start_step=start_step, batch_size=batch_size, shuffle_seed=args.shuffle_seed)
    train_size = len(dataset)
    # assert train_size > val_size, "Traing set should be larger than validation set"
    print("train_dataset: {}".format(train_size))
    print("test_dataset: {}".format(val_size))
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=args.num_wk, pin_memory=True
    )

    return train_loader


def loading_hd5_train(args):
    print("loading hd5 ...")
    hd5_path = "{}/{}.hd5".format(args.data_dir, os.path.basename(args.data_dir))
    index_train_path = "{}/{}_train.npy".format(args.data_dir, os.path.basename(args.data_dir))
    train_dataset = TrainingDataSet2(
        hd5_path, index_train_path, limit_size=args.limit_train_size, tokenization=args.tokenization, 
    )
    print("train_dataset: {}".format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_wk,
        pin_memory=True,
    )
    return train_loader


def loading_hd5_valid(args):
    print("loading hd5 ...")
    hd5_path = "{}/{}.hd5".format(args.data_dir, os.path.basename(args.data_dir))
    index_val_path = "{}/{}_val.npy".format(args.data_dir, os.path.basename(args.data_dir))
    test_dataset = TestingDataSet2(
        hd5_path, index_val_path, limit_size=args.val_size, tokenization=args.tokenization
    )
    print("test_dataset: {}".format(len(test_dataset)))

    valid_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        num_workers=args.num_wk,
        pin_memory=True,
    )
    return valid_loader


def network(config_path):
    config = toml.load(config_path)
    model = load_symbol(config, "Model")(config)
    return model


def get_dataset(args):
    print("[loading data]")
    # train_loader_kwargs, valid_loader_kwargs = load_numpy(
    #     args.limit_train_size, args.data_dir, args.part_num
    # )
    # loader_kwargs = {
    #     "batch_size": args.batch_size, "num_workers": 4, "pin_memory": True
    # }
    # train_dataset = train_loader_kwargs.get("dataset")

    train_dataset, valid_dataset = loading_hd5_two_time(
        args, args.data_dir, args.batch_size, gen_dataloader=False
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=8,
        sampler=train_sampler,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, valid_loader


def get_dataset_in_one_dir(args, dist=True, encoder_only=False):
    print("[loading data]")
    train_dataset, valid_dataset = loading_hd5_in_one_dir(
        args, args.data_dir, args.batch_size, gen_dataloader=False, encoder_only=encoder_only
    )

    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,  # Changed from 0 to 4 for better data loading performance
            sampler=train_sampler,
            pin_memory=True,
        )
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(train_dataset)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=0,
            sampler=train_sampler,
            pin_memory=True,
        )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, valid_loader


def get_dataset_from_pt(args, dist=True):
    """
    从预转换的数据文件加载数据集 (高效版本)
    
    支持两种格式:
    - pt: PyTorch Tensor 格式 (快速, 支持预加载)
    - memmap: NumPy 内存映射格式 (最快, 适合超大数据集)
    
    Args:
        args: 包含 data_dir, batch_size, val_batch_size 等参数
        dist: 是否使用分布式采样器
    
    Returns:
        train_loader, valid_loader
    """
    from opencall.data_loader.data import FastDataset
    from torch.utils.data import Subset
    
    print("[loading data from optimized format]")
    
    train_dir = args.data_dir
    val_dir = os.path.join(os.path.dirname(args.data_dir), 'val_mmap')
    
    # 加载数据集 (FastDataset 会自动检测格式)
    # preload=False 避免多进程重复加载导致内存爆炸
    train_dataset = FastDataset(train_dir, preload=False)
    
    # 验证集大小限制
    val_size = getattr(args, 'val_size', 20000)
    
    # 验证集可选
    if os.path.exists(val_dir):
        valid_dataset = FastDataset(val_dir, preload=False)
        # 如果验证集太大，截取一部分
        if len(valid_dataset) > val_size:
            print(f"[Info] Limiting validation set from {len(valid_dataset)} to {val_size}")
            valid_dataset = Subset(valid_dataset, range(val_size))
    else:
        print(f"[Warning] Validation directory not found: {val_dir}")
        print(f"[Info] Using last {val_size} samples from train data for validation")
        # 从训练集末尾取一部分作为验证集
        total_len = len(train_dataset)
        val_size = min(val_size, total_len // 10)  # 最多取 10% 作为验证
        val_indices = range(total_len - val_size, total_len)
        valid_dataset = Subset(train_dataset, val_indices)
    
    print(f"train_dataset: {len(train_dataset)}")
    print(f"valid_dataset: {len(valid_dataset)}")
    
    # 获取 num_workers 参数
    # memmap 格式数据读取很快，不需要太多 workers
    num_workers = getattr(args, 'num_workers', 2)
    
    # 检测数据格式，memmap 格式建议使用较少的 workers
    is_memmap = hasattr(train_dataset, '_dataset') and hasattr(train_dataset._dataset, 'signals')
    if is_memmap and num_workers > 4:
        print(f"[Info] memmap format detected, reducing num_workers from {num_workers} to 2")
        num_workers = 2
    
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            drop_last=True,  # DDP 训练建议 drop_last 避免不均匀 batch
        )
    else:
        from torch.utils.data import RandomSampler
        train_sampler = RandomSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=num_workers,
            sampler=train_sampler,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=min(num_workers, 2),  # 验证集不需要太多 workers
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,  # 验证不频繁，不需要 persistent
    )
    
    return train_loader, valid_loader



def get_lr_scheduler(
    epochs, optimizer, data_len, last_epoch=0, end_ratio=0.01, warmup_steps=100
):
    lr_scheduler_fn = linear_warmup_cosine_decay(
        end_ratio=end_ratio, warmup_steps=warmup_steps
    )
    return lr_scheduler_fn(optimizer, data_len, epochs, last_epoch)


def get_lr_scheduler_continue(
    optimizer, data_len, end_ratio=0.01, warmup_steps=100, start_step=0
):
    lr_scheduler_fn = linear_warmup_cosine_decay_for_continue(
        end_ratio=end_ratio, warmup_steps=warmup_steps, start_step=start_step,
    )
    return lr_scheduler_fn(optimizer, data_len)


def log_func(msg, path):
    if path is None:
        print(msg)
        return
    
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Use proper file writing instead of os.system
        with open(path, 'a', encoding='utf-8') as f:
            f.write(f"{msg}\n")
            f.flush()
    except Exception as e:
        # Fallback to print if file writing fails
        print(f"Failed to write to log file {path}: {e}")
        print(msg)


def init(seed, device, deterministic=True):
    """
    Initialise random libs and setup cudnn

    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cpu":
        return
    # torch.cuda.set_device(device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    assert torch.cuda.is_available()


def permute(x, input_layout, output_layout):
    """
    Permute `x` from `input_layout` to `output_layout`

    >>> permute(x, 'TNC', 'NTC')
    """
    if input_layout == output_layout:
        return x
    return x.permute(*[input_layout.index(x) for x in output_layout])


def concat(xs, dim=0):
    """
    Type agnostic concat.
    """
    if isinstance(xs[0], torch.Tensor):
        return torch.cat(xs, dim=dim)
    elif isinstance(xs[0], np.ndarray):
        return np.concatenate(xs, axis=dim)
    elif isinstance(xs[0], list):
        return [x for l in xs for x in l]
    elif isinstance(xs[0], str):
        return "".join(xs)
    elif isinstance(xs[0], dict):
        return {k: concat([x[k] for x in xs], dim) for k in xs[0].keys()}
    else:
        raise TypeError


def select_range(x, start, end, dim=0):
    """
    Type agnostic range select.
    """
    if isinstance(x, dict):
        return {k: select_range(v, start, end, dim) for (k, v) in x.items()}
    if dim == 0 or isinstance(x, list):
        return x[start:end]
    return x[(*(slice(None),) * dim, slice(start, end))]


def size(x, dim=0):
    """
    Type agnostic size.
    """
    if hasattr(x, "shape"):
        return x.shape[dim]
    elif dim == 0:
        return len(x)
    raise TypeError


def half_supported():
    """
    Returns whether FP16 is support on the GPU
    """
    try:
        return get_device_capability()[0] >= 7
    except:
        return False


def phred(prob, scale=1.0, bias=0.0):
    """
    Converts `prob` into a ascii encoded phred quality score between 0 and 40.
    """
    p = max(1 - prob, 1e-4)
    q = -10 * np.log10(p) * scale + bias
    return chr(int(np.round(q) + 33))


def mean_qscore_from_qstring(qstring):
    """
    Convert qstring into a mean qscore
    """
    if len(qstring) == 0:
        return 0.0
    qs = np.array(qstring, "c").view(np.uint8) - 33
    mean_err = np.exp(qs * (-np.log(10) / 10.0)).mean()
    return -10 * np.log10(max(mean_err, 1e-4))


ascii_mapping = torch.tensor([0, 65, 67, 71, 84], dtype=torch.uint8)
def decode_ref(encoded, labels, encoder_only=False):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    if encoder_only:
        valid_mask = (encoded >= 1) & (encoded <= 4)
        valid_values = encoded[valid_mask].to(torch.int64)
        ascii_values = ascii_mapping[valid_values]
        return ascii_values.cpu().numpy().tobytes().decode('ascii')
    else:
        return "".join(labels[e] for e in encoded if e)


def decode_flipflop_ref(encoded, labels):
    """
    Convert a integer encoded reference into a string and remove blanks
    """
    return "".join(labels[e] for e in encoded if e)


def column_to_set(filename, idx=0, skip_header=False):
    """
    Pull a column from a file and return a set of the values.
    """
    if filename and os.path.isfile(filename):
        with open(filename, "r") as tsv:
            if skip_header:
                next(tsv)
            return {line.strip().split()[idx] for line in tsv.readlines()}


def chunk(signal, chunksize, overlap):
    """
    Convert a read into overlapping chunks before calling
    """
    T = signal.shape[0]
    if chunksize == 0:
        chunks = signal[None, :]
    elif T < chunksize:
        chunks = torch.nn.functional.pad(signal, (chunksize - T, 0))[None, :]
    else:
        stub = (T - overlap) % (chunksize - overlap)
        chunks = signal[stub:].unfold(0, chunksize, chunksize - overlap)
        if stub > 0:
            chunks = torch.cat([signal[None, :chunksize], chunks], dim=0)
    return chunks.unsqueeze(1)


def stitch(chunks, chunksize, overlap, length, stride, reverse=False):
    """
    Stitch chunks together with a given overlap
    """
    if chunks.shape[0] == 1:
        return chunks.squeeze(0)

    semi_overlap = overlap // 2
    start, end = semi_overlap // stride, (chunksize - semi_overlap) // stride
    stub = (length - overlap) % (chunksize - overlap)
    first_chunk_end = (stub + semi_overlap) // stride if (stub > 0) else end

    if reverse:
        chunks = list(chunks)
        return concat(
            [
                chunks[-1][:-start],
                *(x[-end:-start] for x in reversed(chunks[1:-1])),
                chunks[0][-first_chunk_end:],
            ]
        )
    else:
        return concat(
            [chunks[0, :first_chunk_end], *chunks[1:-1, start:end], chunks[-1, start:]]
        )


def batchify(items, batchsize, dim=0):
    """
    Batch up items up to `batch_size`.
    """
    stack, pos = [], 0
    for k, v in items:
        breaks = range(batchsize - pos, size(v, dim), batchsize)
        for start, end in zip([0, *breaks], [*breaks, size(v, dim)]):
            sub_batch = select_range(v, start, end, dim)
            stack.append(((k, (pos, pos + end - start)), sub_batch))
            if pos + end - start == batchsize:
                ks, vs = zip(*stack)
                yield ks, concat(vs, dim)
                stack, pos = [], 0
            else:
                pos += end - start

    if len(stack):
        ks, vs = zip(*stack)
        yield ks, concat(vs, dim)


def unbatchify(batches, dim=0):
    """
    Reconstruct batches.
    """
    batches = (
        (k, select_range(v, start, end, dim))
        for sub_batches, v in batches
        for k, (start, end) in sub_batches
    )
    return (
        (k, concat([v for (k, v) in group], dim))
        for k, group in groupby(batches, itemgetter(0))
    )


def load_symbol(config, symbol):
    """
    Dynamic load a symbol from module specified in model config.
    """
    if not isinstance(config, dict):
        if not os.path.isdir(config) and os.path.isdir(
            os.path.join(__models__, config)
        ):
            dirname = os.path.join(__models__, config)
        else:
            dirname = config
        config = toml.load(os.path.join(dirname, "config.toml"))
    imported = import_module(config["model"]["package"])
    return getattr(imported, symbol)


def match_names(state_dict, model):
    keys_and_shapes = lambda state_dict: zip(
        *[
            (k, s)
            for s, i, k in sorted(
                [(v.shape, i, k) for i, (k, v) in enumerate(state_dict.items())]
            )
        ]
    )
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])


def parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """
    cigstr = result.cigar.decode.decode()
    first = re.search(split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = "" if cliplen == 0 else "{}S".format(cliplen)
    if first_op == "I":
        pre = "{}S".format(int(first_count) + cliplen)
    elif first_op == "D":
        pre = clip
        rstart = int(first_count)
    else:
        pre = "{}{}".format(clip, prefix)

    mid = cigstr[len(prefix) :]
    end_clip = len(seq) - result.end_query - 1
    suf = "{}S".format(end_clip) if end_clip > 0 else ""
    new_cigstr = "".join((pre, mid, suf))
    return rstart, new_cigstr


def accuracy(ref, seq, balanced=False, min_coverage=0.0):
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    counts = defaultdict(int)

    q_coverage = len(alignment.traceback.query) / len(seq)
    r_coverage = len(alignment.traceback.ref) / len(ref)

    if r_coverage < min_coverage:
        return 0.0

    _, cigar = parasail_to_sam(alignment, seq)

    for count, op in re.findall(split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts["="] - counts["I"]) / (
            counts["="] + counts["X"] + counts["D"]
        )
    else:
        accuracy = counts["="] / (counts["="] + counts["I"] + counts["X"] + counts["D"])
    return accuracy * 100


def print_alignment(ref, seq):
    """
    Print the alignment between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(seq, ref, 8, 4, parasail.dnafull)
    print(alignment.traceback.ref)
    print(alignment.traceback.comp)
    print(alignment.traceback.query)

    print("  Score=%s" % alignment.score)
    return alignment.score


def poa(groups, max_poa_sequences=100, gpu_mem_per_batch=0.9):
    """
    Generate consensus for POA groups.

    Args:
        groups : A list of lists of sequences for which consensus is to be generated.
    """
    free, total = cuda.cuda_get_mem_info(cuda.cuda_get_device())
    gpu_mem_per_batch *= free
    batch = CudaPoaBatch(
        max_poa_sequences, gpu_mem_per_batch, stream=None, output_type="consensus"
    )
    results = []

    for i, group in enumerate(groups, start=1):
        group_status, seq_status = batch.add_poa_group(group)

        # Once batch is full, run POA processing
        if group_status == 1 or i == len(groups):
            batch.generate_poa()

            consensus, coverage, status = batch.get_consensus()
            results.extend(consensus)

            batch.reset()
            group_status, seq_status = batch.add_poa_group(group)

    return results


def safe_mkdir(filepath):
    """Create a directory if there isn't one already."""
    if os.path.exists(filepath):
        assert filepath.split("/")[-1] not in [
            "outputs",
            "raw_data",
            "datasets",
            "models",
            "refs",
        ]
        shutil.rmtree(filepath, True)
        print(filepath + " removed")
        try:
            os.mkdir(filepath)
        except OSError:
            print("Fail to creat folder{}".format(filepath))
        else:
            print(filepath + " created")
    else:
        try:
            os.mkdir(filepath)
        except OSError:
            print("Fail to creat folder".format(filepath))
        else:
            print(filepath + " created")
