#!/usr/bin/env python3
"""Simplified debug to check img shape in batch."""

import sys
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from mmcv.utils import Config
from mmseg.datasets import build_dataloader, build_dataset

cfg = Config.fromfile('configs/cityscapes_indo_segformer.py')
dataset = build_dataset(cfg.data.train)
dataloader = build_dataloader(
    dataset,
    samples_per_gpu=cfg.data.samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    num_gpus=1,
    dist=False,
    seed=0,
    drop_last=True)

batch = next(iter(dataloader))
img_batch = batch['img']

if img_batch:
    first_img = img_batch[0]
    print(f"First image shape: {first_img.shape}")
else:
    print("No images in the batch.")