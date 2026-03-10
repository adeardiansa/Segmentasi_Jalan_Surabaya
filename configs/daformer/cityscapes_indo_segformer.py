# ---------------------------------------------------------------
# Training configuration untuk Indonesian Cityscapes Dataset
# ---------------------------------------------------------------
# dataset settings
_base_ = [
    '../_base_/models/segformer_b5.py',
    '../_base_/datasets/cityscapes_indo_512x512.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/adamw.py',
    '../_base_/schedules/poly10warm.py'
]

# Data settings - reduce batch size for CPU training
data = dict(
    samples_per_gpu=1,  # Reduced from 2 for CPU
    workers_per_gpu=0,  # Set to 0 for Windows/CPU compatibility
    persistent_workers=False,  # Disable persistent workers for CPU
)

# Training settings - optimizer and learning rate are from base configs
# You can override them here if needed

# Runtime settings
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU', save_best='mIoU')

# GPU settings - use CPU if GPU not available
gpu_ids = []  # Empty list means use CPU

# Disable code archive generation to speed up startup
disable_code_archive = True
