#!/usr/bin/env python
"""Test dataset loading untuk Cityscapes Indo"""

import sys
import os
from pathlib import Path

print("=" * 60)
print("Testing Indonesian Cityscapes Dataset")
print("=" * 60)

# Check data directories
data_root = Path('data/cityscapes_indo')
print(f"\n1. Checking data directory: {data_root.absolute()}")
print(f"   Exists: {data_root.exists()}")

if data_root.exists():
    # Check subdirectories
    left_img_train = data_root / 'leftImg8bit' / 'train'
    gtfine_train = data_root / 'gtFine' / 'train'
    
    print(f"\n2. Checking leftImg8bit/train:")
    print(f"   Exists: {left_img_train.exists()}")
    if left_img_train.exists():
        images = list(left_img_train.glob('**/*.png'))
        print(f"   Images found: {len(images)}")
        if images:
            print(f"   Example: {images[0].name}")
    
    print(f"\n3. Checking gtFine/train:")
    print(f"   Exists: {gtfine_train.exists()}")
    if gtfine_train.exists():
        labels = list(gtfine_train.glob('**/*labelTrainIds.png'))
        print(f"   Label files found: {len(labels)}")
        if labels:
            print(f"   Example: {labels[0].name}")
    
    # Try to load one image and label
    print(f"\n4. Testing image loading:")
    try:
        from PIL import Image
        import numpy as np
        
        images = sorted(left_img_train.glob('**/*.png'))
        if images:
            img = Image.open(images[0])
            print(f"   ✓ Successfully loaded: {images[0].name}")
            print(f"   Shape: {img.size}, Mode: {img.mode}")
            
            # Try to load corresponding label
            label_name = images[0].stem + '_gtFine_labelTrainIds.png'
            label_path = gtfine_train / images[0].parent.name / label_name
            
            if not label_path.exists():
                # Try different naming convention
                label_path = gtfine_train / images[0].parent.name / images[0].stem.replace('_leftImg8bit', '_gtFine_labelTrainIds') + '.png'
            
            if label_path.exists():
                label = Image.open(label_path)
                print(f"   ✓ Successfully loaded label: {label_path.name}")
                print(f"   Label shape: {label.size}")
            else:
                print(f"   ✗ Label not found at: {label_path}")
                # List available files in label directory
                label_dir = gtfine_train / images[0].parent.name
                if label_dir.exists():
                    available = list(label_dir.glob('*.png'))
                    print(f"   Available files in {label_dir.name}:")
                    for f in available[:5]:
                        print(f"     - {f.name}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("Dataset check complete")
print("=" * 60)
