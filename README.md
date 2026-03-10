# 🛣️ Segmentasi Semantik Infrastruktur Jalan Surabaya
## Berbasis Domain Adaptation dengan HRDA (High-Resolution Domain Adaptation)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.1-EE4C2C?logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.0-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-Apache%202.0-green)
![Skripsi](https://img.shields.io/badge/Skripsi-S1-orange)

**Segmentasi semantik pada citra jalan kota Surabaya menggunakan pendekatan Unsupervised Domain Adaptation (UDA) — memanfaatkan data sintetis GTA5 untuk melatih model yang mampu memahami kondisi jalan nyata di Indonesia.**

</div>

---

## 📌 Deskripsi Proyek

Proyek ini merupakan penelitian skripsi yang berfokus pada **segmentasi semantik infrastruktur jalan di Kota Surabaya** menggunakan pendekatan **Domain Adaptation**. Tantangan utama dalam segmentasi semantik di konteks Indonesia adalah keterbatasan data berlabel (*annotated data*) pada kondisi jalan lokal yang memiliki karakteristik berbeda dengan dataset benchmark internasional (seperti Cityscapes yang berbasis kota-kota Eropa).

### Apa itu Domain Adaptation?

Domain Adaptation adalah teknik *transfer learning* yang memungkinkan model yang dilatih pada **domain sumber** (misalnya, data sintetis dari game GTA5) untuk bekerja dengan baik pada **domain target** (citra jalan nyata di Surabaya), meskipun terdapat perbedaan distribusi data yang signifikan (*domain gap*).

```
Domain Sumber (Source)          Domain Target
┌─────────────────────┐         ┌─────────────────────────┐
│  GTA5 (Sintetis)    │──HRDA──▶│  Jalan Surabaya (Nyata) │
│  + Cityscapes       │         │  (Indonesian Cityscapes) │
│  (dengan label)     │         │  (tanpa/sedikit label)   │
└─────────────────────┘         └─────────────────────────┘
```

### Kenapa HRDA?

**HRDA (High-Resolution Domain Adaptation)** adalah metode state-of-the-art yang menggabungkan dua skala resolusi:
- **Context crop (resolusi rendah)**: Menangkap informasi konteks gambar secara keseluruhan
- **Detail crop (resolusi tinggi)**: Mempertahankan detail halus seperti marka jalan, tepi trotoar, dan rambu

Pendekatan dua-skala ini secara signifikan meningkatkan akurasi segmentasi dibanding metode resolusi tunggal, terutama pada objek kecil yang sering ditemui di jalan kota.

---

## 🏗️ Arsitektur Model

```
Input Citra (1024×1024)
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
DAFormer     DAFormer       ← Decoder dengan Separable ASPP
Head         Head
  │             │
  └──────┬──────┘
         │ Scale Attention (class-wise)
         ▼
   Segmentation Map
```

### Komponen Utama

| Komponen | Detail |
|---|---|
| **Backbone** | Mix Transformer B5 (MiT-B5) — pre-trained ImageNet |
| **Decoder** | DAFormer dengan Separable ASPP |
| **UDA Method** | DACS (Domain Adaptation via Cross-domain mixed Sampling) |
| **Resolusi Training** | 1024×1024 (High-Resolution) |
| **Detail Crop** | 512×512 |
| **Context Scale** | 0.5× (512×512) |

---

## 📂 Struktur Proyek

```
HRDA1 train yes/
├── configs/                    # Konfigurasi model dan training
│   ├── _base_/
│   │   ├── datasets/           # Konfigurasi dataset
│   │   ├── models/             # Arsitektur model
│   │   ├── schedules/          # Optimizer & learning rate
│   │   └── uda/                # Pengaturan UDA (DACS, dll.)
│   ├── hrda/
│   │   └── gtaHR2csHR_hrda.py  # Konfigurasi utama HRDA
│   └── cityscapes_indo_segformer.py  # Konfigurasi SegFormer untuk dataset Indo
│
├── mmseg/                      # Inti framework segmentasi (MMSegmentation)
├── mmsegmentation/             # Library MMSegmentation
├── tools/                      # Script training & evaluasi
│   ├── train.py                # Script utama training
│   ├── test.py                 # Script evaluasi/inference
│   └── dist_train.sh           # Training multi-GPU
│
├── data/                       # Dataset (tidak disertakan di repo)
│   └── cityscapes_indo/        # Dataset jalan Surabaya (Indonesian Cityscapes)
│       ├── leftImg8bit/        # Gambar asli
│       ├── gtFine/             # Anotasi ground truth
│       └── category_mapping.json
│
├── pretrained/                 # Model pre-trained (MiT-B5)
├── work_dirs/                  # Output training (checkpoint, log)
├── resources/                  # Gambar, demo, overview arsitektur
│
├── convert_annotations.py      # Konversi anotasi polygon → labelTrainIds
├── experiments.py              # Generator konfigurasi eksperimen
├── run_experiments.py          # Runner eksperimen otomatis
├── test_dataset.py             # Validasi dataset
├── validate_config.py          # Validasi konfigurasi
├── requirements.txt            # Dependensi Python
└── TRAINING_CITYSCAPES_INDO.md # Panduan training dataset Indo
```

---

## ⚙️ Instalasi & Persiapan

### 1. Prasyarat

- Python 3.8+
- CUDA 11.0
- GPU dengan minimal 16GB VRAM (direkomendasikan: NVIDIA RTX 3090 / TITAN RTX)

### 2. Clone Repository

```bash
git clone https://github.com/adeardiansa/Segmentasi_Jalan_Surabaya.git
cd Segmentasi_Jalan_Surabaya
```

### 3. Buat Environment Virtual

```bash
conda create -n hrda python=3.8
conda activate hrda
```

### 4. Install Dependensi

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
# Letakkan mit_b5.pth di folder pretrained/
```

---

## 📊 Dataset

### Indonesian Cityscapes (Jalan Surabaya)

Dataset target dikumpulkan dari citra jalan Kota Surabaya dengan format serupa Cityscapes:

```
data/cityscapes_indo/
├── leftImg8bit/
│   ├── train/     # Gambar training
│   ├── val/       # Gambar validasi
│   └── test/      # Gambar test
├── gtFine/
│   ├── train/     # Anotasi training
│   ├── val/       # Anotasi validasi
│   └── test/      # Anotasi test
├── train.txt      # Daftar file training
├── val.txt        # Daftar file validasi
├── test.txt       # Daftar file test
└── category_mapping.json  # Pemetaan kategori ke trainId
```

### Konversi Anotasi

Jika anotasi masih dalam format polygon JSON, jalankan konversi terlebih dahulu:

```bash
python convert_annotations.py
```

### Dataset Sumber (Source Domain)

| Dataset | Jenis | Jumlah Gambar | Keterangan |
|---|---|---|---|
| **GTA5** | Sintetis (Game) | ~24,000 | Grand Theft Auto V renderings |
| **Cityscapes** | Nyata (Eropa) | ~3,000 | Kota-kota Eropa (Düsseldorf, dll.) |
| **Indonesian Cityscapes** | Nyata (Indonesia) | Variabel | **Jalan Kota Surabaya** |

---

## 🚀 Training

### Opsi 1: HRDA Training (Rekomendasi)

Training utama dengan metode HRDA — GTA5 → Indonesian Cityscapes:

```bash
python tools/train.py configs/hrda/gtaHR2csHR_hrda.py \
    --work-dir work_dirs/hrda_surabaya
```

### Opsi 2: SegFormer Baseline

Training SegFormer standar sebagai baseline:

```bash
python tools/train.py configs/cityscapes_indo_segformer.py \
    --work-dir work_dirs/segformer_baseline
```

### Opsi 3: Multi-GPU Training

```bash
./tools/dist_train.sh configs/hrda/gtaHR2csHR_hrda.py <JUMLAH_GPU> \
    --work-dir work_dirs/hrda_surabaya_multigpu

# Contoh dengan 2 GPU:
./tools/dist_train.sh configs/hrda/gtaHR2csHR_hrda.py 2 \
    --work-dir work_dirs/hrda_surabaya
```

### Hyperparameter Utama

| Parameter | Nilai |
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

## 🧪 Evaluasi

### Evaluasi pada Validation Set

```bash
python tools/test.py configs/hrda/gtaHR2csHR_hrda.py \
    work_dirs/hrda_surabaya/latest.pth \
    --eval mIoU
```

### Inference pada Gambar Baru

```python
from mmseg.apis import inference_segmentor, init_segmentor
import cv2

# Inisialisasi model
config_file = 'configs/hrda/gtaHR2csHR_hrda.py'
checkpoint_file = 'work_dirs/hrda_surabaya/latest.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# Jalankan segmentasi
img = 'path/to/gambar_jalan_surabaya.jpg'
result = inference_segmentor(model, img)

# Simpan hasil visualisasi
model.show_result(img, result, out_file='hasil_segmentasi.png', opacity=0.5)
```

### Metrik Evaluasi

| Metrik | Keterangan |
|---|---|
| **mIoU** | Mean Intersection over Union (metrik utama) |
| **mAcc** | Mean Accuracy per kelas |
| **aAcc** | Overall pixel accuracy |

---

## 🗂️ Kelas Segmentasi

Model ini melakukan segmentasi terhadap elemen infrastruktur jalan, mengikuti format label Cityscapes yang telah disesuaikan dengan kondisi jalan Surabaya:

| ID | Kelas | Keterangan |
|---|---|---|
| 0 | road | Permukaan jalan aspal |
| 1 | sidewalk | Trotoar/kaki lima |
| 2 | building | Bangunan di sekitar jalan |
| 3 | wall | Tembok/dinding |
| 4 | fence | Pagar |
| 5 | pole | Tiang (lampu, rambu) |
| 6 | traffic light | Lampu lalu lintas |
| 7 | traffic sign | Rambu lalu lintas |
| 8 | vegetation | Pepohonan/tanaman |
| 9 | terrain | Medan/tanah |
| 10 | sky | Langit |
| 11 | person | Pejalan kaki |
| 12 | rider | Pengendara sepeda/motor |
| 13 | car | Mobil |
| 14 | truck | Truk |
| 15 | bus | Bus |
| 16 | train | Kereta |
| 17 | motorcycle | Sepeda motor |
| 18 | bicycle | Sepeda |

---

## 🔧 Konfigurasi & Eksperimen

### Generate Konfigurasi Eksperimen

```bash
# Generate konfigurasi untuk eksperimen tertentu (misal, exp ID 40 = Final HRDA)
python run_experiments.py --exp 40
```

### Validasi Dataset

```bash
python test_dataset.py
```

### Validasi Konfigurasi

```bash
python validate_config.py
```

---

## 🛠️ Troubleshooting

| Masalah | Solusi |
|---|---|
| `No such file or directory: data/cityscapes_indo` | Pastikan dataset sudah ditempatkan dengan struktur yang benar |
| Error saat load annotation | Jalankan `convert_annotations.py` untuk konversi format label |
| CUDA Out of Memory | Kurangi `batch_size` ke 1, atau `crop_size` ke 512×512 |
| Hasil segmentasi buruk | Cek kualitas anotasi dataset target, tambah iterasi training |

### Konfigurasi untuk GPU Terbatas

Edit file konfigurasi dan sesuaikan:

```python
data = dict(
    samples_per_gpu=1,    # Kurangi batch size
    workers_per_gpu=2,
)
# Di konfigurasi model:
hr_crop_size = [256, 256]  # Kurangi ukuran detail crop
```

---

## 🧠 Metode Utama

### HRDA (High-Resolution Domain Adaptation)

HRDA menggabungkan prediksi dari dua skala untuk mengatasi trade-off antara resolusi tinggi dan konteks global:

1. **Multi-Scale Processing**: Gambar diproses dalam dua skala — full resolution (detail) dan downsampled 0.5× (context)
2. **Scale Attention**: Model belajar bobot attention untuk setiap kelas secara terpisah (*class-wise attention*)
3. **Detail Loss (λd=0.1)**: Supervisi tambahan pada prediction skala tinggi untuk mempertajam detail

### DACS (Domain Adaptation via Cross-domain mixed Sampling)

Metode UDA yang digunakan untuk *self-training* dengan pseudo-label:
- Menghasilkan pseudo-label pada domain target menggunakan model yang sedang dilatih
- Menggabungkan patch gambar sumber dan target (*copy-paste mixing*)
- Menerapkan Rare Class Sampling (RCS) untuk mengatasi class imbalance

### DAFormer Decoder

Decoder berbasis transformer dengan Separable ASPP yang efisien secara komputasi, dioptimalkan untuk UDA tasks.

---

## 📚 Referensi

- **HRDA** — Hoyer, L., Dai, D., & Van Gool, L. (2022). *HRDA: Context-Aware High-Resolution Domain-Adaptive Semantic Segmentation*. ECCV 2022. [[Paper]](https://arxiv.org/abs/2204.13132) [[Code]](https://github.com/lhoyer/HRDA)
- **DAFormer** — Hoyer, L., Dai, D., & Van Gool, L. (2022). *DAFormer: Improving Network Architectures and Training Strategies for Domain-Adaptive Semantic Segmentation*. CVPR 2022.
- **DACS** — Tranheden, W., et al. (2021). *DACS: Domain Adaptation via Cross-domain Mixed Sampling*. WACV 2021.
- **SegFormer** — Xie, E., et al. (2021). *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*. NeurIPS 2021.

---

## 📄 Lisensi

Kode ini dikembangkan di atas [HRDA](https://github.com/lhoyer/HRDA) dan [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) yang berlisensi **Apache License 2.0**. Silakan lihat file `LICENSE` dan folder `resources/license_*` untuk detail setiap komponen.

</div>
