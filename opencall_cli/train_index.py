#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Time    :   2023/02/21 17:08:22
@Author  :   Xu Yan, Junjie Zhen
@Email   :   yanxu@genomics.cn, zhenjunjie@genomics.cn
"""

import config
import os
import sys
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pro_dir)

import glob
import re
from opencall.utils.util import model_eval, init, network, get_dataset_in_one_dir, get_lr_scheduler, log_func
import datetime
import time
import argparse
import torch
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import json
import numpy as np
import random


def clip_gradient(optimizer, grad_clip):
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 1. init backend
    init(args.seed, device, (not args.nondeterministic))

    config_file_path = f"{pro_dir}/opencall/configs/{args.model}.toml"
    res_dir = os.path.join(args.res_prefix, args.output_name)

    log_path = f"{res_dir}/training.log"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
        os.system(f"cp {config_file_path} {res_dir}/config.toml")
    log_writer = SummaryWriter(os.path.join(res_dir, "logs")) if res_dir else None
    msg = f"{'=' * 20} START TRAINING {'=' * 20}"
    log_func(msg, log_path)
    training_params = {}
    for key, value in vars(args).items():
        log_func(f"{key}: {value}", log_path)
        training_params[key] = value
    log_func(f"res_dir: {res_dir}", log_path)

    # 3. get data
    train_loader, valid_loader = get_dataset_in_one_dir(args,dist=False)
    data_len = len(train_loader)
    log_func(f"data len: {data_len}", log_path)

    # 2. loading pre-trained model
    model = network(config_file_path).to(device)

    if args.pre_trained_params_file and os.path.exists(args.pre_trained_params_file):
        print(f"loading pretrained model: {args.pre_trained_params_file}")
        model.load_state_dict(torch.load(args.pre_trained_params_file, map_location=device))
    else:
        weight_files = glob.glob(os.path.join(res_dir, "weights_*.tar"))
        if len(weight_files) > 0:
            weights_num = max(
                [int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files]
            )
            params_file = os.path.join(res_dir, f"weights_{weights_num}.tar")
            print(f"loading pretrained model: {params_file}")
            model.load_state_dict(torch.load(params_file, map_location=device))

    # 3. define opt
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

    # try:
    for epoch in range(int(args.epoch_num)):
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        # step2 = 0
        for num, batch in enumerate(train_loader):
            t0 = time.time()
            step = 0

            smoothed_loss = None
            optimizer.zero_grad()
            losses = None
            with amp.autocast(enabled=args.use_amp):
                for data_, targets_, lengths_ in zip(
                    *map(lambda t: t.chunk(grad_accum_split, dim=0), batch)
                ):
                    data_, targets_, lengths_ = (
                        data_.to(device, non_blocking=True),
                        targets_.to(device, non_blocking=True),
                        lengths_.to(device, non_blocking=True),
                    )
                    # if step2 == 0:
                    #     print(targets_[0])
                    # step2 += 1
                    scores_ = model(data_)
                    losses_ = model.loss(scores_, targets_, lengths_)
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
            perc = 1 if num == 0 else num * 100 / data_len
            msg = f"Epoch: {epoch}, batch num: {num}, finished: {perc:.2f}%, loss: {losses.get('loss'):.6f}, lr: {lr:.6f}, took time: {t1 - t0:.2f}/{tooktime:.2f}/{tooktime * 100 / perc / 3600:.2f}h"
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
                final_params_file_path = f"{res_dir}/weights_0.tar"
                model_state = model.state_dict()
                torch.save(model_state, final_params_file_path)
                res = model_eval(
                    dataloader=valid_loader,
                    model_dir=res_dir,
                    weight_path=final_params_file_path,
                    is_half=True,
                    device=device,
                )
                log_func(f"mean:      {res[0]:.2f}", log_path)
                log_func(f"median:    {res[1]:.2f}", log_path)
                log_func(f"tooktime:  {res[2]:.2f}", log_path)
                log_func(f"samples/sec:    {res[3]:.2E}", log_path)
                log_func(f"bases/sec:    {res[4]:.2f}", log_path)
                log_func(f"val chunks num:    {res[5]:.0f}", log_path)
                chunk_acc = {}
                chunk_acc['mean'] = round(res[0], 2)
                chunk_acc['median'] = round(res[1], 2)
                chunk_acc['tooktime'] = round(res[2], 2)
                chunk_acc['speed(samples/sec)'] = round(res[3], 2)
                chunk_acc['speed(bases/sec)'] = round(res[4], 2)
                chunk_acc['val_chunks_num'] = int(res[5])
                with open(f'{res_dir}/training_params.json', 'w') as json_file:
                    json_file.write(json.dumps(training_params, indent=4))
                with open(f'{res_dir}/chunk_acc.json', 'w') as json_file:
                    json_file.write(json.dumps(chunk_acc, indent=4))
            step += 1
            if 0 < args.limit_train_size < num:
                log_func("max training batch, skip", log_path)
                break
    # except Exception as e:
    #     print(e)

    log_func("saving model ...", log_path)
    final_params_file_path = f"{res_dir}/weights_0.tar"
    model_state = model.state_dict()
    torch.save(model_state, final_params_file_path)
    log_func(f"{'=' * 20} FINISHED {'=' * 20}", log_path)
    res = model_eval(
        dataloader=valid_loader,
        model_dir=res_dir,
        weight_path=final_params_file_path,
        is_half=True,
        device=device,
    )
    log_func(f"mean:      {res[0]:.2f}", log_path)
    log_func(f"median:    {res[1]:.2f}", log_path)
    log_func(f"tooktime:  {res[2]:.2f}", log_path)
    log_func(f"samples/sec:    {res[3]:.2E}", log_path)
    log_func(f"bases/sec:    {res[4]:.2f}", log_path)
    log_func(f"val chunks num:    {res[5]:.0f}", log_path)
    chunk_acc = {}
    chunk_acc['mean'] = round(res[0], 2)
    chunk_acc['median'] = round(res[1], 2)
    chunk_acc['tooktime'] = round(res[2], 2)
    chunk_acc['speed(samples/sec)'] = round(res[3], 2)
    chunk_acc['speed(bases/sec)'] = round(res[4], 2)
    chunk_acc['val_chunks_num'] = int(res[5])
    with open(f'{res_dir}/training_params.json', 'w') as json_file:
        json_file.write(json.dumps(training_params, indent=4))
    with open(f'{res_dir}/chunk_acc.json', 'w') as json_file:
        json_file.write(json.dumps(chunk_acc, indent=4))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="./train_data/normal_data/train"
    )
    parser.add_argument("--epoch_num", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--pre_trained_params_file", type=str, default="")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--limit_train_size", default=1000000000, type=int)
    parser.add_argument("--tokenization", default="kmer", type=str)
    parser.add_argument("--val_size", default=20000, type=int)
    parser.add_argument("--use_amp", default=True, type=bool)
    parser.add_argument("--device", default="cuda:1", type=str)
    parser.add_argument("--debug", default=0, type=int)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--model", default="quartznet", type=str)
    parser.add_argument("--val_batch_size", default=16, type=int)
    parser.add_argument("--gpu_nums", default=1, type=int)
    parser.add_argument("--warmup_steps", default=100, type=int)
    parser.add_argument("--part_num", default=0, type=int)
    parser.add_argument("--data_type", default="index", type=str)
    parser.add_argument("--output_name", default="none7", type=str)
    parser.add_argument("--res_prefix", default="/workspace/train_res", type=str)
    return parser

if __name__ == "__main__":
    main()
