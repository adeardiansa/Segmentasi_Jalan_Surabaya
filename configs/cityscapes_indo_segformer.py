# Base config
_base_ = [
    './_base_/models/segformer_r101.py',
    './_base_/datasets/cityscapes_indo_512x512.py',
    './_base_/default_runtime.py',
    './_base_/schedules/adamw.py',
    './_base_/schedules/poly10warm.py'
]

# Model settings
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 2, 2),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        in_channels=3
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

# Dataset settings
data = dict(
    samples_per_gpu=1,  # Keep batch size as 1 to avoid dimension issues
    workers_per_gpu=0,  # Set to 0 to avoid multiprocessing issues
    persistent_workers=False,  # Disable persistent workers
    train=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes_indo/',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train'),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes_indo/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val'),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes_indo/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val')
)

# Runtime settings
runner = dict(type='IterBasedRunner', max_iters=10000)  # Mengurangi iterasi untuk percobaan awal
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU')

# Optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optimizer_config = dict()

# Learning policy
lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)