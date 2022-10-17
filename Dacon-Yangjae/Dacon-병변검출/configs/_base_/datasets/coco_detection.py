albu_train_transforms = [
    #dict(
    #    type='ShiftScaleRotate',
    #    shift_limit=0.0625,
    #    scale_limit=0.0,
    #    rotate_limit=0,
    #    interpolation=1,
    #    p=1),
    
    #dict(
    #    type='RandomCrop',
    #    crop_size=400
    #    crop_type='absolute',
    #    p=1),
    
    dict(type='Cutout', num_holes=4, max_h_size=60, max_w_size=60, fill_value=0, p=0.5),
    dict(type='Emboss', p=0.5),
    
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.5],
        contrast_limit=[0.1, 0.5],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='RGBShift',
                r_shift_limit=10,
                g_shift_limit=10,
                b_shift_limit=10,
                p=1),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=1)
        ],
        p=0.5),
    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
    dict(type='ChannelShuffle', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.5),
]


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1300, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
#     dict(type='Albu', transforms=albu_train_transforms,
#         bbox_params=dict(
#             type='BboxParams',
#             format='pascal_voc',
#             label_fields=['gt_labels'],
#             min_visibility=0.0,
#             filter_lost_elements=True),
#         keymap={
#             'img': 'image',
#             'gt_bboxes': 'bboxes',
#         },
#         update_pad_shape=False,
#         skip_img_without_anno=True),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    #dict(type='MixUp'),
    #dict(type='Mosaic'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]



test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(576, 576),
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            #dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]



data = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
