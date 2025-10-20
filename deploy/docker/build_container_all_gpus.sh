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
    -v ${opencall_dir}:/workspace/OpenCall \
    -v /home/zjj:/workspace/zjj \
    -v /mnt/seqdata/basecall_data:/workspace/basecall_data \
    moffet_worker_x_cyclonebasecall:v4.3 
    #opencall:0.22


docker run \
    --rm \
    -itd \
    --gpus all  \
    --name huada \
    -p 7777:22 \
    --ipc=host \
    -v /ssd7/huada:/workspace/huada \
    opencall:0.22 