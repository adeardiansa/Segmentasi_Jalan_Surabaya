#!/usr/bin/env python
import sys
from mmcv.utils import Config

try:
    cfg = Config.fromfile('configs/cityscapes_indo_segformer.py')
    print("✓ Config loaded successfully!")
    print(f"\nDataset configuration:")
    print(f"  Data root: {cfg.data_root}")
    print(f"  Train pipeline steps: {len(cfg.train_pipeline)}")
    print(f"  Test pipeline steps: {len(cfg.test_pipeline)}")
    print(f"\nModel: {cfg.model.type}")
    print(f"Backbone: {cfg.model.backbone.type if 'backbone' in cfg.model else 'N/A'}")
    print(f"\nTraining config:")
    print(f"  Max iterations: {cfg.runner.max_iters}")
    print(f"  Batch size: {cfg.data.samples_per_gpu}")
    print(f"  Workers: {cfg.data.workers_per_gpu}")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)
