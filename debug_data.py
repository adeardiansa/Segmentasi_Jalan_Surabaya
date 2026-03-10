#!/usr/bin/env python3
"""Debug data loading to check format."""

import sys
import os.path as osp

# Add the parent directory to sys.path
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmcv.utils import Config
from mmseg.datasets import build_dataloader, build_dataset

cfg = Config.fromfile('configs/cityscapes_indo_segformer.py')

# Build dataset
dataset = build_dataset(cfg.data.train)
print(f"Dataset length: {len(dataset)}")

# Get first sample
sample = dataset[0]
print(f"\nFirst sample keys: {sample.keys()}")
for key, value in sample.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    else:
        print(f"  {key}: {type(value)}")

# Build dataloader
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=cfg.data.samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1,
    dist=False,
    seed=0,
    drop_last=True)

# Get first batch
batch = next(iter(dataloader))
print(f"\nFirst batch keys: {batch.keys()}")
for key, value in batch.items():
    if isinstance(value, list):
        print(f"  {key}: list with {len(value)} items")
        if len(value) > 0:
            item = value[0]
            print(f"    type of first item: {type(item)}")
            if hasattr(item, 'shape'):
                print(f"    shape={item.shape}, dtype={item.dtype}")
            elif isinstance(item, (tuple, list)):
                print(f"    tuple/list with {len(item)} elements")
                for j, sub_item in enumerate(item):
                    if hasattr(sub_item, 'shape'):
                        print(f"      [{j}]: shape={sub_item.shape}")
            else:
                print(f"    value: {item}")
