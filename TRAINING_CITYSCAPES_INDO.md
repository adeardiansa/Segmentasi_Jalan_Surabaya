# Panduan Training dengan Indonesian Cityscapes Dataset

## Struktur Dataset yang Diperlukan
```
data/cityscapes_indo/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
├── gtFine/
│   ├── train/
│   ├── val/
│   └── test/
├── train.txt
├── val.txt
├── test.txt
└── category_mapping.json
```

## Mempersiapkan Annotations
Sebelum training, pastikan file-file label sudah dalam format yang benar:
- Format: PNG dengan trainId
- Nama file harus sesuai dengan gambar, misal: `image.png` → `image_gtFine_labelTrainIds.png`

Jika label Anda belum dalam format ini, jalankan script konversi:

```bash
python tools/convert_datasets/cityscapes.py
```

## Training dengan Dataset Indo

### Opsi 1: Basic Training (SegFormer)
```bash
python tools/train.py configs/cityscapes_indo_segformer.py --work-dir work_dirs/cityscapes_indo_basic
```

### Opsi 2: Training dengan HRDA (High-Resolution Domain Adaptation)
Jika ingin menggunakan HRDA dengan dataset Indo:

```bash
python tools/train.py configs/hrda/cityscapesindo_hrda.py --work-dir work_dirs/cityscapes_indo_hrda
```

### Opsi 3: Training dengan Multiple GPUs
```bash
./tools/dist_train.sh configs/cityscapes_indo_segformer.py <GPU_COUNT> --work-dir work_dirs/cityscapes_indo
```

Contoh dengan 2 GPU:
```bash
./tools/dist_train.sh configs/cityscapes_indo_segformer.py 2 --work-dir work_dirs/cityscapes_indo
```

## Testing/Inference

### Test di Validation Set
```bash
python tools/test.py configs/cityscapes_indo_segformer.py work_dirs/cityscapes_indo_basic/latest.pth --eval mIoU
```

### Inference pada Gambar Baru
```python
from mmseg.apis import inference_segmentor, init_segmentor
import cv2
import matplotlib.pyplot as plt

config_file = 'configs/cityscapes_indo_segformer.py'
checkpoint_file = 'work_dirs/cityscapes_indo_basic/latest.pth'
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

img = 'path/to/your/image.jpg'
result = inference_segmentor(model, img)
visualize_result = model.show_result(img, result, out_file='result.png')
```

## Konfigurasi Penting

### Data Augmentation
Sesuaikan pipeline di `configs/_base_/datasets/cityscapes_indo_512x512.py`:
- `crop_size`: Ukuran crop default 512x512
- `RandomFlip`: Probability 0.5
- `PhotoMetricDistortion`: Untuk color augmentation

### Training Parameters (di konfigurasi training file)
- `lr`: Learning rate 6e-5
- `max_iters`: 160,000 iterations
- `batch_size`: 2 samples per GPU
- `workers_per_gpu`: 4

### Menyesuaikan untuk Resource Terbatas

Jika GPU memory terbatas, edit konfigurasi:
```python
data = dict(
    samples_per_gpu=1,  # Kurangi batch size
    workers_per_gpu=2,
)
crop_size = (256, 256)  # Kurangi input size
```

## Monitoring Training
Lihat progress di tensorboard:
```bash
tensorboard --logdir work_dirs/cityscapes_indo_basic
```

## Troubleshooting

### Error: "No such file or directory: data/cityscapes_indo"
- Pastikan struktur folder sudah benar
- Pastikan Anda menjalankan command dari root directory project

### Error saat load annotation
- Cek format label file (harus PNG dengan trainId)
- Cek nama file: harus match dengan image filename

### Out of Memory
- Kurangi batch size
- Kurangi crop size
- Gunakan gradient accumulation
