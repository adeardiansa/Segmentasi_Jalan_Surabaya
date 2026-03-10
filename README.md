# 🛣️ Semantic Segmentation of Surabaya Road Infrastructure
## Based on Domain Adaptation with HRDA (High-Resolution Domain Adaptation)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.0-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

**Semantic segmentation of road imagery in Surabaya using an Unsupervised Domain Adaptation (UDA) approach — leveraging synthetic GTA5 data to train a model capable of understanding real road conditions in Indonesia.**

</div>

---

## 📌 Project Description

This project focused on **semantic segmentation of road infrastructure in the City of Surabaya** using a **Domain Adaptation** approach. The main challenge in semantic segmentation in the Indonesian context is the scarcity of *annotated data* for local road conditions, which have characteristics different from international benchmark datasets (such as Cityscapes, which is based on European cities).

### What is Domain Adaptation?

Domain Adaptation is a *transfer learning* technique that enables a model trained on a **source domain** (e.g., synthetic data from the GTA5 game) to perform well on a **target domain** (real road imagery in Surabaya), despite significant data distribution differences (*domain gap*).

```
Source Domain                    Target Domain
┌─────────────────────┐         ┌─────────────────────────┐
│  GTA5 (Synthetic)   │──HRDA──▶│  Surabaya Roads (Real)  │
│  + Cityscapes       │         │  (Indonesian Cityscapes) │
│  (with labels)      │         │  (unlabeled/few labels)  │
└─────────────────────┘         └─────────────────────────┘
```

### Why HRDA?

**HRDA (High-Resolution Domain Adaptation)** is a state-of-the-art method that combines two resolution scales:
- **Context crop (low resolution)**: Captures the overall contextual information of the image
- **Detail crop (high resolution)**: Preserves fine details such as road markings, curb edges, and signs

This dual-scale approach significantly improves segmentation accuracy compared to single-resolution methods, especially for small objects commonly found on urban roads.

---

## 🏗️ Model Architecture

```
Input Image (1024×1024)
         │
    ┌────┴────┐
    │  HRDA   │  ← HRDAEncoderDecoder
    └────┬────┘
         │
  ┌──────┴──────┐
  │             │
  ▼             ▼
Context      Detail
Crop (0.5x)  Crop (512×512)
  │             │
  ▼             ▼
MiT-B5       MiT-B5        ← Mix Transformer Backbone (SegFormer)
Encoder      Encoder
  │             │
  ▼             ▼
DAFormer     DAFormer       ← Decoder with Separable ASPP
Head         Head
  │             │
  └──────┬──────┘
         │ Scale Attention (class-wise)
         ▼
   Segmentation Map
```

### Key Components

| Component | Details |
|---|---|
| **Backbone** | Mix Transformer B5 (MiT-B5) — pre-trained on ImageNet |
| **Decoder** | DAFormer with Separable ASPP |
| **UDA Method** | DACS (Domain Adaptation via Cross-domain mixed Sampling) |
| **Training Resolution** | 1024×1024 (High-Resolution) |
| **Detail Crop** | 512×512 |
| **Context Scale** | 0.5× (512×512) |

---

## 📂 Project Structure

```
HRDA1 train yes/
├── configs/                    # Model and training configurations
│   ├── _base_/
│   │   ├── datasets/           # Dataset configurations
│   │   ├── models/             # Model architectures
│   │   ├── schedules/          # Optimizer & learning rate schedules
│   │   └── uda/                # UDA settings (DACS, etc.)
│   ├── hrda/
│   │   └── gtaHR2csHR_hrda.py  # Main HRDA configuration
│   └── cityscapes_indo_segformer.py  # SegFormer config for Indo dataset
│
├── mmseg/                      # Core segmentation framework (MMSegmentation)
├── mmsegmentation/             # MMSegmentation library
├── tools/                      # Training & evaluation scripts
│   ├── train.py                # Main training script
│   ├── test.py                 # Evaluation/inference script
│   └── dist_train.sh           # Multi-GPU training
│
├── data/                       # Datasets (not included in repo)
│   └── cityscapes_indo/        # Surabaya road dataset (Indonesian Cityscapes)
│       ├── leftImg8bit/        # Original images
│       ├── gtFine/             # Ground truth annotations
│       └── category_mapping.json
│
├── pretrained/                 # Pre-trained models (MiT-B5)
├── work_dirs/                  # Training outputs (checkpoints, logs)
├── resources/                  # Images, demos, architecture overview
│
├── convert_annotations.py      # Convert polygon annotations → labelTrainIds
├── experiments.py              # Experiment configuration generator
├── run_experiments.py          # Automated experiment runner
├── test_dataset.py             # Dataset validation
├── validate_config.py          # Configuration validation
├── requirements.txt            # Python dependencies
└── TRAINING_CITYSCAPES_INDO.md # Training guide for Indo dataset
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites

- Python 3.8+
- CUDA 11.0
- GPU with at least 16GB VRAM (recommended: NVIDIA RTX 3090 / TITAN RTX)

### 2. Clone Repository

```bash
git clone https://github.com/adeardiansa/Segmentasi_Jalan_Surabaya.git
cd Segmentasi_Jalan_Surabaya
```

### 3. Create Virtual Environment

```bash
conda create -n hrda python=3.8
conda activate hrda
```

### 4. Install Dependencies

```bash
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

# Install MMSegmentation
cd mmsegmentation
pip install -e .
cd ..
```

### 5. Download Pre-trained Backbone

```bash
# Download MiT-B5 pre-trained weights
mkdir -p pretrained
# Place mit_b5.pth inside the pretrained/ folder
```

---

## 📊 Dataset

### Indonesian Cityscapes (Surabaya Roads)

The target dataset is collected from road imagery in the City of Surabaya in a format similar to Cityscapes:

```
data/cityscapes_indo/
├── leftImg8bit/
│   ├── train/     # Training images
│   ├── val/       # Validation images
│   └── test/      # Test images
├── gtFine/
│   ├── train/     # Training annotations
│   ├── val/       # Validation annotations
│   └── test/      # Test annotations
├── train.txt      # Training file list
├── val.txt        # Validation file list
├── test.txt       # Test file list
└── category_mapping.json  # Category-to-trainId mapping
```

### Annotation Conversion

If annotations are still in polygon JSON format, run the conversion first:

```bash
python convert_annotations.py
```

### Source Domain Datasets

| Dataset | Type | Number of Images | Notes |
|---|---|---|---|
| **GTA5** | Synthetic (Game) | ~24,000 | Grand Theft Auto V renderings |
| **Cityscapes** | Real (European) | ~3,000 | European cities (Düsseldorf, etc.) |
| **Indonesian Cityscapes** | Real (Indonesian) | Variable | **Surabaya City Roads** |

---

## 🚀 Training

### Option 1: HRDA Training (Recommended)

Main training with the HRDA method — GTA5 → Indonesian Cityscapes:

```bash
python tools/train.py configs/hrda/gtaHR2csHR_hrda.py \
    --work-dir work_dirs/hrda_surabaya
```

### Option 2: SegFormer Baseline

Standard SegFormer training as a baseline:

```bash
python tools/train.py configs/cityscapes_indo_segformer.py \
    --work-dir work_dirs/segformer_baseline
```

### Option 3: Multi-GPU Training

```bash
./tools/dist_train.sh configs/hrda/gtaHR2csHR_hrda.py <NUM_GPUS> \
    --work-dir work_dirs/hrda_surabaya_multigpu

# Example with 2 GPUs:
./tools/dist_train.sh configs/hrda/gtaHR2csHR_hrda.py 2 \
    --work-dir work_dirs/hrda_surabaya
```

### Key Hyperparameters

| Parameter | Value |
|---|---|
| **Optimizer** | AdamW |
| **Learning Rate** | 6e-5 |
| **Scheduler** | Polynomial + Warmup (10%) |
| **Max Iterations** | 40,000 |
| **Batch Size** | 2 per GPU |
| **Input Resolution** | 1024×1024 |
| **Detail Crop Size** | 512×512 |
| **Context Scale** | 0.5× |
| **HR Loss Weight (λd)** | 0.1 |

### Monitor Training

```bash
tensorboard --logdir work_dirs/hrda_surabaya
```

---

## 🧪 Evaluation

### Evaluation on Validation Set

```bash
python tools/test.py configs/hrda/gtaHR2csHR_hrda.py \
    work_dirs/hrda_surabaya/latest.pth \
    --eval mIoU
```

### Inference on New Images

```python
from mmseg.apis import inference_segmentor, init_segmentor
import cv2

# Initialize model
config_file = 'configs/hrda/gtaHR2csHR_hrda.py'
checkpoint_file = 'work_dirs/hrda_surabaya/latest.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Run segmentation
img = 'path/to/surabaya_road_image.jpg'
result = inference_segmentor(model, img)

# Save visualization result
model.show_result(img, result, out_file='segmentation_result.png', opacity=0.5)
```

### Evaluation Metrics

| Metric | Description |
|---|---|
| **mIoU** | Mean Intersection over Union (primary metric) |
| **mAcc** | Mean Accuracy per class |
| **aAcc** | Overall pixel accuracy |

---

## 🗂️ Segmentation Classes

The model performs segmentation on road infrastructure elements, following the Cityscapes label format adapted to Surabaya road conditions:

| ID | Class | Description |
|---|---|---|
| 0 | road | Asphalt road surface |
| 1 | sidewalk | Sidewalk/pavement |
| 2 | building | Buildings along the road |
| 3 | wall | Walls |
| 4 | fence | Fences |
| 5 | pole | Poles (lamp posts, signs) |
| 6 | traffic light | Traffic lights |
| 7 | traffic sign | Traffic signs |
| 8 | vegetation | Trees/plants |
| 9 | terrain | Terrain/ground |
| 10 | sky | Sky |
| 11 | person | Pedestrians |
| 12 | rider | Bicycle/motorcycle riders |
| 13 | car | Cars |
| 14 | truck | Trucks |
| 15 | bus | Buses |
| 16 | train | Trains |
| 17 | motorcycle | Motorcycles |
| 18 | bicycle | Bicycles |

---

## 🔧 Configuration & Experiments

### Generate Experiment Configurations

```bash
# Generate configuration for a specific experiment (e.g., exp ID 40 = Final HRDA)
python run_experiments.py --exp 40
```

### Dataset Validation

```bash
python test_dataset.py
```

### Configuration Validation

```bash
python validate_config.py
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|---|---|
| `No such file or directory: data/cityscapes_indo` | Ensure the dataset is placed with the correct directory structure |
| Error when loading annotations | Run `convert_annotations.py` to convert the label format |
| CUDA Out of Memory | Reduce `batch_size` to 1, or `crop_size` to 512×512 |
| Poor segmentation results | Check the quality of target dataset annotations, increase training iterations |

### Configuration for Limited GPU

Edit the config file and adjust:

```python
data = dict(
    samples_per_gpu=1,    # Reduce batch size
    workers_per_gpu=2,
)
# In the model config:
hr_crop_size = [256, 256]  # Reduce detail crop size
```

---

## 🧠 Core Methods

### HRDA (High-Resolution Domain Adaptation)

HRDA combines predictions from two scales to address the trade-off between high resolution and global context:

1. **Multi-Scale Processing**: Images are processed at two scales — full resolution (detail) and downsampled 0.5× (context)
2. **Scale Attention**: The model learns attention weights for each class separately (*class-wise attention*)
3. **Detail Loss (λd=0.1)**: Additional supervision on high-scale predictions to sharpen details

### DACS (Domain Adaptation via Cross-domain mixed Sampling)

The UDA method used for *self-training* with pseudo-labels:
- Generates pseudo-labels on the target domain using the model being trained
- Combines patches from source and target images (*copy-paste mixing*)
- Applies Rare Class Sampling (RCS) to address class imbalance

### DAFormer Decoder

A transformer-based decoder with computationally efficient Separable ASPP, optimized for UDA tasks.

---

## 📚 References

- **HRDA** — Hoyer, L., Dai, D., & Van Gool, L. (2022). *HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation*. ECCV 2022. [[Paper]](https://arxiv.org/abs/2204.13132) [[Code]](https://github.com/lhoyer/HRDA)
- **DAFormer** — Hoyer, L., Dai, D., & Van Gool, L. (2022). *DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation*. CVPR 2022.
- **DACS** — Tranheden, W., et al. (2021). *DACS: Domain Adaptation via Cross-domain Mixed Sampling*. WACV 2021.
- **SegFormer** — Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021.

---

## 📄 License

This code is developed on top of [HRDA](https://github.com/lhoyer/HRDA) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), both licensed under **Apache License 2.0**. Please refer to the `LICENSE` file and the `resources/license_*` folders for details on each component.

</div>
