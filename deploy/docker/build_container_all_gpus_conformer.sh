#!/bin/bash


cd ../../
opencall_dir=$(pwd)
cd deploy/docker

docker run \
    --rm \
    -itd \
    --gpus all  \
    --name zjj_container_7777 \
    -p 7777:22 \
    --ipc=host \
    -v /home/zjj:/workspace/zjj \
    -v /mnt/seqdata/basecall_data:/workspace/basecall_data \
    mhd_training:1.0 
    #opencall:0.22
    #weight_ = weight_.cpu().numpy()


