#!/usr/bin/env python3
"""
Script to read and display information from a single HDF5 file
"""
import sys
import h5py
import numpy as np


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


def read_hdf5_file(filepath):
    """Read and display information about an HDF5 file"""
    print(f"Reading HDF5 file: {filepath}")
    print("=" * 80)
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Print file-level attributes
            print("\nFile-level attributes:")
            if f.attrs:
                for key, val in f.attrs.items():
                    print(f"  {key}: {val}")
            else:
                print("  (No file-level attributes)")
            
            # Print top-level keys
            print(f"\nTop-level keys: {list(f.keys())}")
            
            # Print information about each dataset/group
            print("\nDataset/Group details:")
            print("-" * 80)
            
            def visit_func(name, obj):
                if isinstance(obj, h5py.Group):
                    print(f"\n  Group: {name}/")
                    print_attrs(name, obj)
                elif isinstance(obj, h5py.Dataset):
                    print_dataset_info(name, obj)
            
            # Visit all items in the file
            f.visititems(visit_func)
            
            # If there are datasets at root level
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    print_dataset_info(key, f[key])
            
            print("\n" + "=" * 80)
            print("Successfully read the HDF5 file!")
            
    except Exception as e:
        print(f"\nError reading file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python read_hdf5_file.py <hdf5_file_path>")
        print("Example: python read_hdf5_file.py /workspace/huada/all_refs_label_for_ctc/train_data/250F600274011_first_hour1_1_44.hd5")
        sys.exit(1)
    
    filepath = sys.argv[1]
    exit_code = read_hdf5_file(filepath)
    sys.exit(exit_code)

