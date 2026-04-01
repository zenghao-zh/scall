"""
可视化 SPU 和 GPU backbone 输出的差异

从保存的 .pt 文件中加载数据，选择样本并绘制对比图
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_sample_comparison(spu_output, gpu_output, sample_idx, save_dir=None):
    """
    绘制单个样本的 SPU vs GPU 对比图
    
    Args:
        spu_output: (T, N, F) SPU backbone 输出
        gpu_output: (T, N, F) GPU backbone 输出
        sample_idx: 要可视化的样本索引
        save_dir: 保存图片的目录
    """
    # 提取单个样本的数据 (T, F)
    spu_sample = spu_output[:, sample_idx, :].cpu().numpy()
    gpu_sample = gpu_output[:, sample_idx, :].cpu().numpy()
    
    T, F = spu_sample.shape
    print(f"样本 {sample_idx} 的数据形状: T={T}, F={F}")
    
    # 计算差异
    diff = spu_sample - gpu_sample
    abs_diff = np.abs(diff)
    
    # 统计信息
    print(f"\n样本 {sample_idx} 统计:")
    print(f"  SPU range: [{spu_sample.min():.4f}, {spu_sample.max():.4f}]")
    print(f"  GPU range: [{gpu_sample.min():.4f}, {gpu_sample.max():.4f}]")
    print(f"  Abs diff mean: {abs_diff.mean():.6f}")
    print(f"  Abs diff max: {abs_diff.max():.6f}")
    print(f"  MSE: {np.mean(diff ** 2):.6e}")
    
    # 创建图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. SPU 输出热图
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.imshow(spu_sample.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title(f'SPU Output (Sample {sample_idx})', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Features')
    plt.colorbar(im1, ax=ax1)
    
    # 2. GPU 输出热图
    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.imshow(gpu_sample.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title(f'GPU Output (Sample {sample_idx})', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Features')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 绝对差异热图
    ax3 = plt.subplot(3, 3, 3)
    im3 = ax3.imshow(abs_diff.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax3.set_title(f'Absolute Difference', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Features')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 按时间步的平均值对比
    ax4 = plt.subplot(3, 3, 4)
    spu_mean_t = spu_sample.mean(axis=1)
    gpu_mean_t = gpu_sample.mean(axis=1)
    ax4.plot(spu_mean_t, label='SPU', alpha=0.7, linewidth=1.5)
    ax4.plot(gpu_mean_t, label='GPU', alpha=0.7, linewidth=1.5)
    ax4.set_title('Mean over Features (per Time Step)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time Steps')
    ax4.set_ylabel('Mean Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 按时间步的差异
    ax5 = plt.subplot(3, 3, 5)
    diff_mean_t = abs_diff.mean(axis=1)
    ax5.plot(diff_mean_t, color='red', linewidth=1.5)
    ax5.set_title('Mean Absolute Difference per Time Step', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Mean Abs Diff')
    ax5.grid(True, alpha=0.3)
    
    # 6. 按特征的差异
    ax6 = plt.subplot(3, 3, 6)
    diff_mean_f = abs_diff.mean(axis=0)
    ax6.plot(diff_mean_f, color='orange', linewidth=1.5)
    ax6.set_title('Mean Absolute Difference per Feature', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Features')
    ax6.set_ylabel('Mean Abs Diff')
    ax6.grid(True, alpha=0.3)
    
    # 7. 差异分布直方图
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(diff.flatten(), bins=100, alpha=0.7, edgecolor='black')
    ax7.set_title('Difference Distribution', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Difference (SPU - GPU)')
    ax7.set_ylabel('Frequency')
    ax7.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. 选择几个时间步详细对比
    ax8 = plt.subplot(3, 3, 8)
    time_steps = [T//4, T//2, 3*T//4]
    for t in time_steps:
        ax8.plot(spu_sample[t, :], label=f'SPU t={t}', alpha=0.6, linewidth=1)
        ax8.plot(gpu_sample[t, :], label=f'GPU t={t}', alpha=0.6, linewidth=1, linestyle='--')
    ax8.set_title('Feature Values at Selected Time Steps', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Feature Index')
    ax8.set_ylabel('Value')
    ax8.legend(fontsize=8, ncol=2)
    ax8.grid(True, alpha=0.3)
    
    # 9. 相对误差热图
    ax9 = plt.subplot(3, 3, 9)
    # 避免除零，加上小的 epsilon
    gpu_abs = np.abs(gpu_sample) + 1e-8
    rel_diff = abs_diff / gpu_abs
    im9 = ax9.imshow(rel_diff.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax9.set_title('Relative Difference', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Time Steps')
    ax9.set_ylabel('Features')
    plt.colorbar(im9, ax=ax9)
    
    plt.suptitle(f'SPU vs GPU Backbone Output Comparison - Sample {sample_idx}', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'spu_gpu_comparison_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {save_path}")
    
    plt.show()
    plt.close()


def plot_multiple_samples_summary(spu_output, gpu_output, num_samples=10, save_dir=None):
    """
    绘制多个样本的误差汇总图
    
    Args:
        spu_output: (T, N, F) SPU backbone 输出
        gpu_output: (T, N, F) GPU backbone 输出
        num_samples: 要分析的样本数量
        save_dir: 保存图片的目录
    """
    T, N, F = spu_output.shape
    
    # 选择样本（均匀分布）
    sample_indices = np.linspace(0, N-1, min(num_samples, N), dtype=int)
    
    # 计算每个样本的统计信息
    mse_list = []
    mean_abs_diff_list = []
    max_abs_diff_list = []
    
    for idx in sample_indices:
        spu_sample = spu_output[:, idx, :].cpu().numpy()
        gpu_sample = gpu_output[:, idx, :].cpu().numpy()
        diff = spu_sample - gpu_sample
        abs_diff = np.abs(diff)
        
        mse_list.append(np.mean(diff ** 2))
        mean_abs_diff_list.append(abs_diff.mean())
        max_abs_diff_list.append(abs_diff.max())
    
    # 创建汇总图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. MSE 对比
    ax1 = axes[0, 0]
    ax1.bar(range(len(sample_indices)), mse_list, color='steelblue', edgecolor='black')
    ax1.set_title('MSE per Sample', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('MSE')
    ax1.set_xticks(range(len(sample_indices)))
    ax1.set_xticklabels(sample_indices, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Mean Abs Diff 对比
    ax2 = axes[0, 1]
    ax2.bar(range(len(sample_indices)), mean_abs_diff_list, color='coral', edgecolor='black')
    ax2.set_title('Mean Absolute Difference per Sample', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Mean Abs Diff')
    ax2.set_xticks(range(len(sample_indices)))
    ax2.set_xticklabels(sample_indices, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Max Abs Diff 对比
    ax3 = axes[1, 0]
    ax3.bar(range(len(sample_indices)), max_abs_diff_list, color='lightgreen', edgecolor='black')
    ax3.set_title('Max Absolute Difference per Sample', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Max Abs Diff')
    ax3.set_xticks(range(len(sample_indices)))
    ax3.set_xticklabels(sample_indices, rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. 统计分布
    ax4 = axes[1, 1]
    stats_data = [mse_list, mean_abs_diff_list, max_abs_diff_list]
    labels = ['MSE', 'Mean Abs Diff', 'Max Abs Diff']
    
    # 创建箱线图
    bp = ax4.boxplot(stats_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['steelblue', 'coral', 'lightgreen']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_title('Error Statistics Distribution', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'SPU vs GPU Error Summary ({len(sample_indices)} samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 保存图片
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'spu_gpu_error_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n汇总图已保存到: {save_path}")
    
    plt.show()
    plt.close()
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"多样本误差统计汇总 ({len(sample_indices)} 个样本)")
    print(f"{'='*60}")
    print(f"MSE:")
    print(f"  Mean: {np.mean(mse_list):.6e}")
    print(f"  Std:  {np.std(mse_list):.6e}")
    print(f"  Min:  {np.min(mse_list):.6e}")
    print(f"  Max:  {np.max(mse_list):.6e}")
    print(f"\nMean Abs Diff:")
    print(f"  Mean: {np.mean(mean_abs_diff_list):.6f}")
    print(f"  Std:  {np.std(mean_abs_diff_list):.6f}")
    print(f"  Min:  {np.min(mean_abs_diff_list):.6f}")
    print(f"  Max:  {np.max(mean_abs_diff_list):.6f}")
    print(f"\nMax Abs Diff:")
    print(f"  Mean: {np.mean(max_abs_diff_list):.6f}")
    print(f"  Std:  {np.std(max_abs_diff_list):.6f}")
    print(f"  Min:  {np.min(max_abs_diff_list):.6f}")
    print(f"  Max:  {np.max(max_abs_diff_list):.6f}")


def main():
    parser = argparse.ArgumentParser(description='可视化 SPU vs GPU backbone 输出差异')
    parser.add_argument('--data_file', type=str, required=True,
                        help='保存的 .pt 数据文件路径')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='要可视化的样本索引 (默认: 0)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='汇总图中的样本数量 (默认: 10)')
    parser.add_argument('--save_dir', type=str, default='/workspace/huada/scall/visualizations',
                        help='保存图片的目录')
    parser.add_argument('--show_summary', action='store_true',
                        help='显示多样本汇总图')
    
    args = parser.parse_args()
    
    # 加载数据
    print(f"加载数据: {args.data_file}")
    data = torch.load(args.data_file, map_location='cpu')
    
    spu_output = data['spu_backbone_output']
    gpu_output = data['gpu_backbone_output']
    file_id = data.get('file_id', 'unknown')
    
    T, N, F = spu_output.shape
    print(f"数据形状: T={T}, N={N}, F={F}")
    print(f"文件 ID: {file_id}")
    
    # 检查样本索引
    if args.sample_idx >= N:
        print(f"错误: 样本索引 {args.sample_idx} 超出范围 [0, {N-1}]")
        return
    
    # 绘制单个样本的详细对比
    print(f"\n{'='*60}")
    print(f"绘制样本 {args.sample_idx} 的详细对比图")
    print(f"{'='*60}")
    plot_sample_comparison(spu_output, gpu_output, args.sample_idx, args.save_dir)
    
    # 如果需要，绘制多样本汇总
    if args.show_summary:
        print(f"\n{'='*60}")
        print(f"绘制多样本误差汇总图")
        print(f"{'='*60}")
        plot_multiple_samples_summary(spu_output, gpu_output, args.num_samples, args.save_dir)


if __name__ == "__main__":
    main()
