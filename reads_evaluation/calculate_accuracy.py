#!/usr/bin/env python3
"""
简化版SAM文件准确率计算脚本
只计算准确率并输出图片，移除其他复杂分析
"""

import matplotlib.pyplot as plt
import numpy as np
import pysam
import re
import os


def calculate_identity_from_cigar(cigarstring):
    """Calculate read identity from CIGAR string using formula:
    identity_rate = total_match / (total_ref_base + total_base_ins)"""
    if not cigarstring:
        return None
        
    # Parse CIGAR string
    pattern = r'(\d+)([MIDNSHP=X])'
    operations = re.findall(pattern, cigarstring)
    
    total_match = 0      # Count of matches (M/=)
    total_ref_base = 0   # Count of bases in reference (M/=/X/D)
    total_base_ins = 0   # Count of inserted bases (I)
    
    for length, op in operations:
        length = int(length)
        if op == '=':  # Matches
            total_match += length
            total_ref_base += length
        elif op == 'X':  # Mismatches
            total_ref_base += length
        elif op == 'D':  # Deletions
            total_ref_base += length
        elif op == 'I':  # Insertions
            total_base_ins += length
        elif op == 'S': # softclip
            pass
        else:
            raise ValueError(f"Invalid CIGAR operation: {op}")
    
    denominator = total_ref_base + total_base_ins
    if denominator == 0:
        return None
        
    identity = (total_match / denominator) * 100
    return  total_ref_base,identity


def parse_sam(filename):
    """Parse SAM file and return read identities with basic information"""
    read_data = []
    total_reads = 0
    mapped_reads = 0
    total_ref_base_all = 0
    
    with pysam.AlignmentFile(filename, "r") as samfile:
        for read in samfile:

            if read.flag not in [0,4,16]:
                continue

            if read.flag == 4:
                total_reads += 1
                continue
            mapped_reads += 1
            total_reads += 1
            
            total_ref_base, identity = calculate_identity_from_cigar(read.cigarstring)
            if identity is not None:
                read_info = {
                    'read_name': read.query_name,
                    'identity_rate': identity,
                    'ref_length': total_ref_base,
                }
                read_data.append(read_info)
                total_ref_base_all += total_ref_base
    
    total_ref_base_mean = total_ref_base_all / total_reads if total_reads > 0 else 0
    return read_data, total_reads, mapped_reads, total_ref_base_mean


def calculate_identity_distribution(identities, interval=0.001):
    """Calculate identity distribution with specified interval"""
    # Convert percentage interval to absolute value
    bin_size = interval * 100
    
    # Create bins
    min_identity = min(identities)
    max_identity = max(identities)
    bins = np.arange(min_identity - (min_identity % bin_size),
                    max_identity + bin_size,
                    bin_size)
    
    # Calculate histogram
    hist, bin_edges = np.histogram(identities, bins=bins)
    
    # Find mode
    mode_idx = np.argmax(hist)
    mode_value = (bin_edges[mode_idx] + bin_edges[mode_idx + 1]) / 2
    
    return hist, bin_edges, mode_value


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Calculate identity rate from SAM file")
    parser.add_argument("--sam", type=str, default="./1.sam", help="SAM file path")
    parser.add_argument("--output_png", type=str, default=None, help="Output PNG path (default: same dir as SAM)")
    args = parser.parse_args()

    sam_file = args.sam
    print(f"Analyzing SAM file: {sam_file}")
    read_data, total_reads, mapped_reads, total_ref_base_mean = parse_sam(sam_file)
    
    # Extract identity rates for analysis
    identities = [read['identity_rate'] for read in read_data]
    
    if not identities:
        print("No valid reads found for analysis.")
        return
    
    mean_identity = np.mean(identities)
    median_identity = np.median(identities)
    print(f"Mean identity: {mean_identity:.2f}%")
    print(f"Median identity: {median_identity:.2f}%")
    
    # Calculate mapping rate
    mapping_rate = (mapped_reads / total_reads) * 100
    print(f"\nTotal reads: {total_reads}")
    print(f"Mapped reads: {mapped_reads}")
    print(f"Mapping rate: {mapping_rate:.2f}%")
    
    # Calculate statistics
    hist, bins, mode = calculate_identity_distribution(identities)
    
    # Plot identity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(identities, bins=bins, alpha=0.7)
    plt.axvline(mode, color='red', linestyle='dashed', label=f'Mode: {mode:.2f}%')
    plt.axvline(mean_identity, color='green', linestyle='dashed', label=f'Mean: {mean_identity:.2f}%')
    plt.axvline(median_identity, color='blue', linestyle='dashed', label=f'Median: {median_identity:.2f}%')

    plt.xlabel('Identity (%)')
    plt.ylabel('Count')
    plt.title(f'Identity Distribution - {sam_file}')
    
    bin_width = bins[1] - bins[0] if len(bins) > 1 else 0
    info_text = f'Bin Width: {bin_width:.2f}%\nTotal Ref Bases: {total_ref_base_mean:.2f}\nMapping Rate: {mapping_rate:.2f}%'
    plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    if args.output_png:
        output_file = args.output_png
    else:
        sam_dir = os.path.dirname(sam_file) or "."
        sam_basename = os.path.splitext(os.path.basename(sam_file))[0]
        output_file = os.path.join(sam_dir, f"{sam_basename}.png")
    plt.savefig(output_file)
    plt.close()
    print(f'\nIdentity distribution plot saved as {output_file}')
    
    # Print summary statistics
    print(f"\nModal identity: {mode:.2f}%")
    print(f"Median identity: {median_identity:.2f}%")


if __name__ == '__main__':
    main()
