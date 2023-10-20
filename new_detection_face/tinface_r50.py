# 1. data
dataset_type = 'WIDERFaceDataset'
data_root = 'data/WIDERFace/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
size_divisor = 32

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_train/train.txt',
        img_prefix=data_root + 'WIDER_train/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile', to_float32=True),
            dict(typename='LoadAnnotations', with_bbox=True),
            dict(typename='RandomSquareCrop',
                 crop_choice=[0.3, 0.45, 0.6, 0.8, 1.0]),
            dict(
                typename='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(typename='RandomFlip', flip_ratio=0.5),
            dict(typename='Resize', img_scale=(640, 640), keep_ratio=False),
            dict(typename='Normalize', **img_norm_cfg),
            dict(typename='DefaultFormatBundle'),
            dict(typename='Collect', keys=['img', 'gt_bboxes',
                                           'gt_labels', 'gt_bboxes_ignore']),
        ]),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + 'WIDER_val/val.txt',
        img_prefix=data_root + 'WIDER_val/',
        min_size=1,
        offset=0,
        pipeline=[
            dict(typename='LoadImageFromFile'),
            dict(
                typename='MultiScaleFlipAug',
                img_scale=(1100, 1650),
                flip=False,
                transforms=[
                    dict(typename='Resize', keep_ratio=True),
                    dict(typename='RandomFlip', flip_ratio=0.0),
                    dict(typename='Normalize', **img_norm_cfg),
                    dict(typename='Pad', size_divisor=32, pad_val=0),
                    dict(typename='ImageToTensor', keys=['img']),
                    dict(typename='Collect', keys=['img'])
                ])
        ]),
)

# 2. model
num_classes = 1
strides = [4, 8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 3
ratios = [1.3]
num_anchors = scales_per_octave * len(ratios)

model = dict(
    typename='SingleStageDetector',
    backbone=dict(
        typename='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        norm_cfg=dict(typename='BN'),
        norm_eval=False,
        dcn=None,
        style='pytorch'),
    neck=[
        dict(
            typename='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs='on_input',
            num_outs=6,
            norm_cfg=dict(typename='BN'),
            upsample_cfg=dict(mode='bilinear')),
        dict(
            typename='Inception',
            in_channel=256,
            num_levels=6,
            norm_cfg=dict(typename='BN'),
            share=True)
    ],
    head=dict(
        typename='IoUAwareRetinaHead',
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=dict(typename='BN'),
        use_sigmoid=use_sigmoid))
