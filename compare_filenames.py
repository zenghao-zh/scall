#!/usr/bin/env python3
"""
Script to compare filenames between two directories
"""
import os
import sys
from pathlib import Path
from collections import defaultdict


def get_files_from_directory(directory_path, extension=None):
    """
    Get all files from a directory recursively
    
    Args:
        directory_path: Path to directory
        extension: Optional file extension filter (e.g., '.hd5')
    
    Returns:
        Set of filenames (just the filename, not the full path)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Warning: Directory does not exist: {directory_path}")
        return set()
    
    if not directory.is_dir():
        print(f"Warning: Path is not a directory: {directory_path}")
        return set()
    
    files = set()
    if extension:
        files = {f.name for f in directory.rglob(f"*{extension}")}
    else:
        files = {f.name for f in directory.rglob("*") if f.is_file()}
    
    return files


def compare_directories(dir1, dir2, extension=None):
    """
    Compare filenames between two directories
    
    Args:
        dir1: First directory path
        dir2: Second directory path
        extension: Optional file extension filter
    """
    print("=" * 80)
    print("COMPARING FILENAMES BETWEEN DIRECTORIES")
    print("=" * 80)
    print(f"\nDirectory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    if extension:
        print(f"Extension filter: {extension}")
    print("\n" + "=" * 80)
    
    # Get files from both directories
    files1 = get_files_from_directory(dir1, extension)
    files2 = get_files_from_directory(dir2, extension)
    
    # Find common and unique files
    common_files = files1 & files2
    only_in_dir1 = files1 - files2
    only_in_dir2 = files2 - files1
    
    # Print statistics
    print("\nSTATISTICS:")
    print("-" * 80)
    print(f"Total files in Directory 1: {len(files1)}")
    print(f"Total files in Directory 2: {len(files2)}")
    print(f"Common files (in both):     {len(common_files)}")
    print(f"Only in Directory 1:        {len(only_in_dir1)}")
    print(f"Only in Directory 2:        {len(only_in_dir2)}")
    
    # Print common files
    if common_files:
        print("\n" + "=" * 80)
        print(f"COMMON FILES ({len(common_files)} files):")
        print("=" * 80)
        for filename in sorted(common_files):
            print(f"  {filename}")
    
    # Print files only in directory 1
    if only_in_dir1:
        print("\n" + "=" * 80)
        print(f"ONLY IN DIRECTORY 1 ({len(only_in_dir1)} files):")
        print("=" * 80)
        for filename in sorted(only_in_dir1):
            print(f"  {filename}")
    
    # Print files only in directory 2
    if only_in_dir2:
        print("\n" + "=" * 80)
        print(f"ONLY IN DIRECTORY 2 ({len(only_in_dir2)} files):")
        print("=" * 80)
        for filename in sorted(only_in_dir2):
            print(f"  {filename}")
    
    # Save results to file
    output_file = "filename_comparison.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FILENAME COMPARISON REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Directory 1: {dir1}\n")
        f.write(f"Directory 2: {dir2}\n")
        if extension:
            f.write(f"Extension filter: {extension}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICS:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total files in Directory 1: {len(files1)}\n")
        f.write(f"Total files in Directory 2: {len(files2)}\n")
        f.write(f"Common files (in both):     {len(common_files)}\n")
        f.write(f"Only in Directory 1:        {len(only_in_dir1)}\n")
        f.write(f"Only in Directory 2:        {len(only_in_dir2)}\n")
        
        if common_files:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"COMMON FILES ({len(common_files)} files):\n")
            f.write("=" * 80 + "\n")
            for filename in sorted(common_files):
                f.write(f"  {filename}\n")
        
        if only_in_dir1:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ONLY IN DIRECTORY 1 ({len(only_in_dir1)} files):\n")
            f.write("=" * 80 + "\n")
            for filename in sorted(only_in_dir1):
                f.write(f"  {filename}\n")
        
        if only_in_dir2:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ONLY IN DIRECTORY 2 ({len(only_in_dir2)} files):\n")
            f.write("=" * 80 + "\n")
            for filename in sorted(only_in_dir2):
                f.write(f"  {filename}\n")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Default directories
    default_dir1 = "/workspace/huada/all_refs_label_for_ctc/train_data/train"
    default_dir2 = "/workspace/huada/hg002_label_for_softmax/train_data/train"
    
    if len(sys.argv) == 1:
        # No arguments, use defaults
        dir1 = default_dir1
        dir2 = default_dir2
        extension = ".hd5"
        print(f"Using default directories:")
        print(f"  Directory 1: {dir1}")
        print(f"  Directory 2: {dir2}")
        print(f"  Extension: {extension}\n")
    elif len(sys.argv) == 3:
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]
        extension = None
    elif len(sys.argv) == 4:
        dir1 = sys.argv[1]
        dir2 = sys.argv[2]
        extension = sys.argv[3]
    else:
        print("Usage: python compare_filenames.py [dir1] [dir2] [extension]")
        print(f"\nDefault: python compare_filenames.py")
        print(f"  Directory 1: {default_dir1}")
        print(f"  Directory 2: {default_dir2}")
        print(f"  Extension: .hd5")
        print("\nExample: python compare_filenames.py /path/to/dir1 /path/to/dir2 .hd5")
        sys.exit(1)
    
    compare_directories(dir1, dir2, extension)

