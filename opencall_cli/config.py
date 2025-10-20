import os.path as osp
import os
import sys
sys.path.append(osp.dirname(osp.dirname(__file__)))
HOST = "bgi"

if HOST == "bgi":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12357"
    from torch.utils.tensorboard import SummaryWriter
    writer_flag = True

if HOST == "wuchao":
    LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
    RANK = int(os.getenv('RANK', -1))
    WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
    writer_flag = False
    sys.path.append(osp.join(osp.dirname(osp.dirname(__file__)),"opencall/libs"))
