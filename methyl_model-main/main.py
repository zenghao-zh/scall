import os,re,sys,time
from os.path import join as ospj
import os.path as osp
from glob import glob
import torch

from transformers import BertConfig

PRJ_DIR = osp.dirname(osp.abspath(__file__))
print(PRJ_DIR)
sys.path.append(PRJ_DIR)
from model import DssmBertModel

model_path = f"{PRJ_DIR}/wy_human_v0.1"

def test_torch_model_calling():
    device = 'cuda:0'
    config = BertConfig.from_pretrained(model_path)
    model = DssmBertModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16)
    model=model.to(device)
    model.eval()

    for batch_size in [512, 1024, 2048, 4096, 8192]:
        for idx in range(5):
            input_ids = torch.load(f"{PRJ_DIR}/data/bsz_{batch_size}/input_ids_{idx}.pth").to(torch.int32).to(device)
            input_base_feat = torch.load(f"{PRJ_DIR}/data/bsz_{batch_size}/input_base_feat_{idx}.pth").to(torch.float32).to(device)
            input_signals = torch.load(f"{PRJ_DIR}/data/bsz_{batch_size}/input_signals_{idx}.pth").to(torch.float32).to(device)
            logits = torch.load(f"{PRJ_DIR}/data/bsz_{batch_size}/logits_{idx}.pth").to(torch.float32).to(device)   
            probs = torch.load(f"{PRJ_DIR}/data/bsz_{batch_size}/probs_{idx}.pth").to(torch.float32).to(device)   

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                o_logits = model(input_ids, input_base_feat, input_signals).logits
            o_logits = o_logits.to(torch.float32)
            o_probs = torch.softmax(o_logits, dim=-1)

            # print(f"logits:{o_logits.shape}, {o_logits[:3]}")
            # print(f"probs:{o_probs.shape}, {o_probs[:3]}")

            logits_diff = (logits - o_logits).abs().max().item()
            probs_diff = (probs - o_probs).abs().max().item()
            logits_match = torch.allclose(logits, o_logits, atol=1e-3, rtol=1e-4)
            probs_match = torch.allclose(probs, o_probs, atol=1e-3, rtol=1e-4)
            print(f"bsz_{batch_size}/idx_{idx}: logits {'相同' if logits_match else '不同'} (max_diff={logits_diff:.6f}), probs {'相同' if probs_match else '不同'} (max_diff={probs_diff:.6f})")

def test_model_structure():
    config = BertConfig.from_pretrained(model_path)
    model = DssmBertModel.from_pretrained(model_path, config=config, attn_implementation="eager")
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    

    print(f"参数量 (M): {total_params / 1e6:.2f}M")
    print(f"可训练参数量 (M): {trainable_params / 1e6:.2f}M")

    print(model)
    
    # 按模块统计参数量
    # print("\n各模块参数量统计:")
    # for name, module in model.named_modules():
    #     if len(list(module.children())) == 0:  # 只统计叶子模块
    #         params = sum(p.numel() for p in module.parameters())
    #         if params > 0:
    #             print(f"{name}: {params:,} ({params / 1e6:.2f}M)")
    

if __name__ == "__main__":
    # test_model_structure()
    test_torch_model_calling()