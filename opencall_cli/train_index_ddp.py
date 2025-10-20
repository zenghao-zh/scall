import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pro_dir)

import glob
import re
from opencall.utils.util import model_eval, init, network, get_dataset_in_one_dir, get_lr_scheduler, log_func
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime
import time
import argparse
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
from opencall.data_loader.split_index import split_idx_file
import json
import numpy as np
import random


# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '23324'

def get_final_params_file_path(res_dir):
    weight_files = glob.glob(os.path.join(res_dir, "weights_*.tar"))
    if len(weight_files) > 0:
        weights_num = max(
            [int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files]
        )
        final_params_file_path = "{}/weights_{}.tar".format(
            res_dir, weights_num + 1
        )
    else:
        final_params_file_path = "{}/weights_0.tar".format(res_dir)
    return final_params_file_path


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

             
def main():
    # 0. parse args
    torch_seed = 1
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    np.random.seed(torch_seed)
    random.seed(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = get_parser().parse_args()
    local_rank = args.local_rank

    # 1. init backend
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    init(args.seed, local_rank, (not args.nondeterministic))

    config_file_path = f"{pro_dir}/opencall/configs/{args.model}.toml"
    res_dir = f'/workspace/huada/task_results/index_ddp_{args.output_name}'

    log_path = "{}/training.log".format(res_dir)
    if dist.get_rank() == 0:
        # create res folder
        if not os.path.exists(res_dir):
            os.system("mkdir -p {}".format(res_dir))
            os.system("cp {} {}/config.toml".format(config_file_path, res_dir))
        # setting tensorboard log
        log_writer = SummaryWriter(os.path.join(res_dir, "logs")) if res_dir else None
        # record all parameters
        msg = "{} {} {}".format("=" * 20, "START TRAINING", "=" * 20)
        log_func(msg, log_path)
        training_params = {}
        for key, value in vars(args).items():
            log_func("{}: {}".format(key, value), log_path)
            training_params[key] = value
        log_func("res_dir: {}".format(res_dir), log_path)
        # split idx file, make preparation for data loading

    # 3. get data
    train_loader, valid_loader = get_dataset_in_one_dir(args)
    data_len = len(train_loader)
    if dist.get_rank() == 0:
        log_func("data len: {}".format(data_len), log_path)

    # 2. loading pre-trained model
    model_orig = network(config_file_path).to(local_rank)

    if os.path.exists(args.pre_trained_params_file):
        print("loading pretrained model: {}".format(args.pre_trained_params_file))
        model_orig.load_state_dict(torch.load(args.pre_trained_params_file))
    else:
        weight_files = glob.glob(os.path.join(res_dir, "weights_*.tar"))
        if len(weight_files) > 0:
            weights_num = max(
                [int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files]
            )
            params_file = os.path.join(res_dir, "weights_%s.tar" % weights_num)
            print("loading pretrained model: {}".format(params_file))
            model_orig.load_state_dict(torch.load(params_file))

    # if args.model == "conformer":
    if "former" in args.model:
        model = DDP(model_orig, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        # model = DDP(model_orig, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model_orig, device_ids=[local_rank], output_device=local_rank)

    # 3. define opt
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, eps=1e-06)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_lr_scheduler(
        epochs=data_len * args.epoch_num,
        optimizer=optimizer,
        data_len=1,
        last_epoch=0,
        end_ratio=0.01,
        warmup_steps=args.warmup_steps,
    )

    # 4. start training ...
    start_time = time.time()
    grad_accum_split = 1
    amp_scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    
    try:
        for epoch in range(int(args.epoch_num)):
            train_loader.sampler.set_epoch(epoch)
            #######################
            # train one epoch
            #######################
            model.train()
            for num, batch in enumerate(train_loader):
                t0 = time.time()
                step = 0
                smoothed_loss = None
    
                #######################
                # train one step
                #######################
                optimizer.zero_grad()
                losses = None
                with amp.autocast(enabled=args.use_amp):
                    for data_, targets_, lengths_ in zip(
                        *map(lambda t: t.chunk(grad_accum_split, dim=0), batch)
                    ):
                        data_, targets_, lengths_ = (
                            data_.to(local_rank, non_blocking=True),
                            targets_.to(local_rank, non_blocking=True),
                            lengths_.to(local_rank, non_blocking=True),
                        )
                        scores_ = model(data_)
                        losses_ = model_orig.loss(scores_, targets_, lengths_)
    
                        if not isinstance(losses_, dict):
                            losses_ = {"loss": losses_}
    
                        total_loss = (
                            losses_.get("total_loss", losses_["loss"]) / grad_accum_split
                        )
                        amp_scaler.scale(total_loss).backward()
    
                        losses = {
                            k: (
                                (v.item() / grad_accum_split)
                                if losses is None
                                else (v.item() / grad_accum_split) + losses[k]
                            )
                            for k, v in losses_.items()
                        }
                amp_scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=2.0
                ).item()
                clip_gradient(optimizer, 1)
                amp_scaler.step(optimizer)
                amp_scaler.update()
    
                lr = lr_scheduler.get_last_lr()[0]
                lr_scheduler.step()
    
                t1 = time.time()
                tooktime = time.time() - start_time
                if dist.get_rank() == 0:
                    perc = 1 if num == 0 else num * 100 / data_len
                    msg = "Epoch: {}, batch num: {}, finished: {:.2f}%, loss: {:.6f}, lr: {:.6f}, took time: {:.2f}/{:.2f}/{:.2f}h".format(
                        epoch,
                        num,
                        perc,
                        losses.get("loss"),
                        lr,
                        t1 - t0,
                        tooktime,
                        tooktime * 100 / perc / 3600,
                    )
                    log_func(msg, log_path)
                    print(msg)
                    if log_writer and step % 1000 == 0:
                        smoothed_loss = (
                            losses["loss"]
                            if smoothed_loss is None
                            else (0.01 * losses["loss"] + 0.99 * smoothed_loss)
                        )
                        log_writer.add_scalar("loss", smoothed_loss, step)
                    if num % 10000 == 0:
                        final_params_file_path = get_final_params_file_path(res_dir)
                        model_state = (
                            model.module.state_dict()
                            if hasattr(model, "module")
                            else model.state_dict()
                        )
                        torch.save(model_state, final_params_file_path)
                        res = model_eval(
                            dataloader=valid_loader,
                            model_dir=res_dir,
                            weight_path=final_params_file_path,
                            is_half=True,
                            device=local_rank,
                        )
                        log_func("mean:      {:.2f}".format(res[0]), log_path)
                        log_func("median:    {:.2f}".format(res[1]), log_path)
                        log_func("tooktime:  {:.2f}".format(res[2]), log_path)
                        log_func("samples/sec:    {:.2E}".format(res[3]), log_path)
                        log_func("bases/sec:    {:.2f}".format(res[4]), log_path)
                        log_func("val chunks num:    {:.0f}".format(res[5]), log_path)
                        
                        # save train params, chunk evaluation info
                        chunk_acc = {}
                        chunk_acc['mean'] = round(res[0], 2)
                        chunk_acc['median'] = round(res[1], 2)
                        chunk_acc['tooktime'] = round(res[2], 2)
                        chunk_acc['speed(samples/sec)'] = round(res[3], 2)
                        chunk_acc['speed(bases/sec)'] = round(res[4], 2)
                        chunk_acc['val_chunks_num'] = int(res[5])
                        with open('{}/training_params.json'.format(res_dir), 'w') as json_file:
                            json_file.write(json.dumps(training_params, indent = 4))
                
                        with open('{}/chunk_acc.json'.format(res_dir), 'w') as json_file:
                            json_file.write(json.dumps(chunk_acc, indent = 4))                   
    
                step += 1
    
                if 0 < args.limit_train_size < num:
                    log_func("max training batch, skip", log_path)
                    break
    except Exception as e:
        print(e)

    if dist.get_rank() == 0:
        log_func("saving model ...", log_path)

        final_params_file_path = get_final_params_file_path(res_dir)

        model_state = (
            model.module.state_dict()
            if hasattr(model, "module")
            else model.state_dict()
        )
        torch.save(model_state, final_params_file_path)

        log_func("{} {} {}".format("=" * 20, "FINISHED", "=" * 20), log_path)

        # evaluation
        res = model_eval(
            dataloader=valid_loader,
            model_dir=res_dir,
            weight_path=final_params_file_path,
            is_half=True,
            device=local_rank,
        )
        log_func("mean:      {:.2f}".format(res[0]), log_path)
        log_func("median:    {:.2f}".format(res[1]), log_path)
        log_func("tooktime:  {:.2f}".format(res[2]), log_path)
        log_func("samples/sec:    {:.2E}".format(res[3]), log_path)
        log_func("bases/sec:    {:.2f}".format(res[4]), log_path)
        log_func("val chunks num:    {:.0f}".format(res[5]), log_path)
        
        # save train params, chunk evaluation info
        chunk_acc = {}
        chunk_acc['mean'] = round(res[0], 2)
        chunk_acc['median'] = round(res[1], 2)
        chunk_acc['tooktime'] = round(res[2], 2)
        chunk_acc['speed(samples/sec)'] = round(res[3], 2)
        chunk_acc['speed(bases/sec)'] = round(res[4], 2)
        chunk_acc['val_chunks_num'] = int(res[5])
        with open('{}/training_params.json'.format(res_dir), 'w') as json_file:
            json_file.write(json.dumps(training_params, indent = 4))

        with open('{}/chunk_acc.json'.format(res_dir), 'w') as json_file:
            json_file.write(json.dumps(chunk_acc, indent = 4))



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="/workspace/basecall_data/train_data/index/test_index"
    )
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pre_trained_params_file", type=str, default="")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit_train_size", default=0, type=int)
    parser.add_argument("--tokenization", default="kmer", type=str)
    parser.add_argument("--val_size", default=20000, type=int)
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--model", default="lstm_ctc_crf", type=str)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--gpu_nums", default=8, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--part_num", default=0, type=int)
    parser.add_argument("--data_type", default="index", type=str)
    parser.add_argument("--output_name", default="none", type=str)
    return parser


if __name__ == "__main__":
    main()
