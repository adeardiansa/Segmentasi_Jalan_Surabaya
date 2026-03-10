#!/usr/bin/env python3
"""Convert Cityscapes Indonesia polygons to labelTrainIds images."""

import json
import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Load category mapping
with open('data/cityscapes_indo/category_mapping.json', 'r') as f:
    category_data = json.load(f)

# Create mapping from category name to trainId
name_to_trainid = {}
for cat in category_data['categories']:
    name_to_trainid[cat['name']] = cat['trainId']

print("Category mapping:")
for cat in sorted(category_data['categories'], key=lambda x: x['trainId']):
    print(f"  {cat['trainId']}: {cat['name']}")

# Process each split
for split in ['train', 'val', 'test']:
    json_dir = f'data/cityscapes_indo/gtFine/{split}'
    img_dir = f'data/cityscapes_indo/gtFine/{split}'
    
    if not osp.exists(json_dir):
        print(f"Directory {json_dir} not found, skipping...")
        continue
    
    json_files = sorted(Path(json_dir).glob('*_gtFine_polygons.json'))
    print(f"\nProcessing {split} split: {len(json_files)} files")
    
    for json_file in tqdm(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get image size from the original color image
        basename = json_file.stem.replace('_gtFine_polygons', '')
        color_img_path = json_file.parent / f'{basename}_gtFine_color.png'
        
        if not color_img_path.exists():
            print(f"Warning: Color image not found for {basename}")
            continue
        
        color_img = Image.open(color_img_path)
        height, width = color_img.size[::-1]
        
        # Create labelTrainIds image
        label_img = np.full((height, width), 255, dtype=np.uint8)  # 255 = void
        
        # Parse polygons and draw on label image
        for obj in data.get('objects', []):
            polygon = obj.get('polygon', [])
            label = obj.get('label', None)
            
            if not polygon or label is None:
                continue
            
            # Get trainId from label
            if label not in name_to_trainid:
                print(f"Warning: Unknown label '{label}' in {json_file.name}")
                continue
            
            trainid = name_to_trainid[label]
            
            # Convert polygon to numpy array
            polygon_array = np.array(polygon, dtype=np.int32)
            
            if len(polygon_array) > 2:
                # Use PIL ImageDraw to fill polygon
                from PIL import ImageDraw
                img_pil = Image.fromarray(label_img)
                draw = ImageDraw.Draw(img_pil)
                
                # Convert to tuple of coordinates
                poly_coords = [tuple(p) for p in polygon_array]
                draw.polygon(poly_coords, fill=trainid)
                label_img = np.array(img_pil)
        
        # Save labelTrainIds image
        output_path = json_file.parent / f'{basename}_gtFine_labelTrainIds.png'
        Image.fromarray(label_img).save(output_path)

print("\nDone!")
