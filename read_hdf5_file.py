#!/usr/bin/env python3
"""
Script to read and display information from HDF5 files in a directory
"""
import sys
import h5py
import numpy as np
import os
from pathlib import Path


def print_attrs(name, obj):
    """Print attributes of an HDF5 object"""
    if obj.attrs:
        print(f"    Attributes:")
        for key, val in obj.attrs.items():
            print(f"      {key}: {val}")


def print_dataset_info(name, obj):
    """Print detailed information about a dataset"""
    if isinstance(obj, h5py.Dataset):
        print(f"\n  Dataset: {name}")
        print(f"    Shape: {obj.shape}")
        print(f"    Dtype: {obj.dtype}")
        print(f"    Size: {obj.size} elements")
        
        # Calculate size in memory
        size_bytes = obj.size * obj.dtype.itemsize
        if size_bytes < 1024:
            size_str = f"{size_bytes} B"
        elif size_bytes < 1024**2:
            size_str = f"{size_bytes/1024:.2f} KB"
        elif size_bytes < 1024**3:
            size_str = f"{size_bytes/1024**2:.2f} MB"
        else:
            size_str = f"{size_bytes/1024**3:.2f} GB"
        print(f"    Memory size: {size_str}")
        
        # Print attributes
        print_attrs(name, obj)
        
        # Show a sample of the data if it's small enough
        if obj.size > 0 and obj.size <= 20:
            print(f"    Data: {obj[()]}")
        elif obj.size > 0:
            # Show shape and first few elements
            try:
                if len(obj.shape) == 1:
                    print(f"    Sample data (first 5): {obj[:5]}")
                elif len(obj.shape) == 2:
                    print(f"    Sample data (first 3 rows, first 5 cols):")
                    sample = obj[:3, :5]
                    for row in sample:
                        print(f"      {row}")
                else:
                    print(f"    Sample data (first element): shape={obj[0].shape}, dtype={obj[0].dtype}")
            except Exception as e:
                print(f"    (Could not read sample data: {e})")


def read_hdf5_file(filepath, verbose=True):
    """Read and display information about an HDF5 file"""
    if verbose:
        print(f"Reading HDF5 file: {filepath}")
        print("=" * 80)
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Always read file-level attributes (to check accessibility)
            if verbose:
                print("\nFile-level attributes:")
            if f.attrs:
                for key, val in f.attrs.items():
                    if verbose:
                        print(f"  {key}: {val}")
            else:
                if verbose:
                    print("  (No file-level attributes)")
            
            # Always get top-level keys (to check structure)
            keys = list(f.keys())
            if verbose:
                print(f"\nTop-level keys: {keys}")
                print("\nDataset/Group details:")
                print("-" * 80)
            
            # Always visit all items to check data integrity
            def visit_func(name, obj):
                if isinstance(obj, h5py.Group):
                    # Check group attributes
                    _ = dict(obj.attrs.items())
                    if verbose:
                        print(f"\n  Group: {name}/")
                        print_attrs(name, obj)
                elif isinstance(obj, h5py.Dataset):
                    # Check dataset properties and try to read a sample
                    _ = obj.shape
                    _ = obj.dtype
                    _ = obj.size
                    
                    # Try to read a small sample to verify data accessibility
                    if obj.size > 0:
                        try:
                            if len(obj.shape) == 1:
                                _ = obj[:min(5, obj.shape[0])]
                            elif len(obj.shape) == 2:
                                _ = obj[:min(3, obj.shape[0]), :min(5, obj.shape[1])]
                            else:
                                _ = obj[0]
                        except Exception as e:
                            raise Exception(f"Error reading dataset '{name}': {e}")
                    
                    if verbose:
                        print_dataset_info(name, obj)
            
            # Visit all items in the file
            f.visititems(visit_func)
            
            # If there are datasets at root level
            for key in keys:
                if isinstance(f[key], h5py.Dataset):
                    obj = f[key]
                    # Check dataset
                    _ = obj.shape
                    _ = obj.dtype
                    _ = obj.size
                    
                    # Try to read a small sample
                    if obj.size > 0:
                        try:
                            if len(obj.shape) == 1:
                                _ = obj[:min(5, obj.shape[0])]
                            elif len(obj.shape) == 2:
                                _ = obj[:min(3, obj.shape[0]), :min(5, obj.shape[1])]
                            else:
                                _ = obj[0]
                        except Exception as e:
                            raise Exception(f"Error reading root dataset '{key}': {e}")
                    
                    if verbose:
                        print_dataset_info(key, f[key])
            
            if verbose:
                print("\n" + "=" * 80)
                print("Successfully read the HDF5 file!")
            
    except Exception as e:
        if verbose:
            print(f"\nError reading file: {e}")
            import traceback
            traceback.print_exc()
        return False, str(e)
    
    return True, None


def process_directory(directory_path):
    """Process all HDF5 files in a directory"""
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory does not exist: {directory_path}")
        return 1
    
    if not directory.is_dir():
        print(f"Error: Path is not a directory: {directory_path}")
        return 1
    
    # Find all .hd5 files
    hd5_files = list(directory.glob("*.hd5"))
    
    if not hd5_files:
        print(f"No .hd5 files found in {directory_path}")
        return 1
    
    print(f"Found {len(hd5_files)} HDF5 files in {directory_path}")
    print("=" * 80)
    
    failed_files = []
    success_count = 0
    
    for i, filepath in enumerate(hd5_files, 1):
        print(f"\n[{i}/{len(hd5_files)}] Processing: {filepath.name}")
        success, error_msg = read_hdf5_file(filepath, verbose=False)
        
        if success:
            print(f"  ✓ Success")
            success_count += 1
        else:
            print(f"  ✗ Failed: {error_msg}")
            failed_files.append((filepath.name, error_msg))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(hd5_files)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\n" + "=" * 80)
        print("FAILED FILES:")
        print("=" * 80)
        for filename, error_msg in failed_files:
            print(f"\n  File: {filename}")
            print(f"  Error: {error_msg}")
    else:
        print("\nAll files processed successfully! ✓")
    
    return 0


if __name__ == "__main__":
    # Default directory to process
    default_dir = "/workspace/huada/all_refs_label_for_ctc/train_data/train"
    
    if len(sys.argv) == 1:
        # No arguments provided, use default directory
        directory_path = default_dir
        print(f"No directory specified, using default: {directory_path}\n")
    elif len(sys.argv) == 2:
        directory_path = sys.argv[1]
    else:
        print("Usage: python read_hdf5_file.py [directory_path]")
        print(f"Default: python read_hdf5_file.py {default_dir}")
        print("Example: python read_hdf5_file.py /path/to/hdf5/files")
        sys.exit(1)
    
    exit_code = process_directory(directory_path)
    sys.exit(exit_code)

