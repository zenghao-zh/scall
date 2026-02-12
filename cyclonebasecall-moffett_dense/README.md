# cyclonebasecall

> * 方便通过脚本高效完成basecall
> * 方便basecall模型部署
> * 方便basecall模型评估

- input: 单个fast5文件路径或文件夹路径
- output: 一个或者多个fastq文件


## Requirements
- python3.8
- 第三方依赖包参见requirements.txt  (建议直接使用opencall:0.22+版本镜像)


## Usage

### 1. 如何通过脚本的的方式完成一份数据的basecall?
#### 准备工作
> * GPU服务器(GPU, opencall:0.22+镜像)
> * 数据(fast5文件或者文件夹)

#### 环境搭建
创建容器(默认yf_container_7999)

```sh
bash build_container.sh
```

安装依赖(moffett版本没有容器，自行安装requirement即可)
```sh
python -m pip install -e .
cd cycloneio && python setup.py install # 之前有提供
python -m pip install tables
```

#### basecall
```sh
python cyclonebasecall/test/example.py
```
