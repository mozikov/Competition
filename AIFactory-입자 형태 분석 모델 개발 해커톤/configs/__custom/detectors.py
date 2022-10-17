# _base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
# _base_ = '../detectors/detectors_htc_r101_20e_coco.py'
_base_ = '../detectors/detectors_htc_r50_1x_coco.py'




import time
now = time.localtime()
save = '{}년_{}월_{}일_{}시_{}분_{}초_detectors'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
work_dir = 'work_dirs/{}'.format(save)


runner = dict(type='EpochBasedRunner', max_epochs=100)
auto_scale_lr = dict(enable=False, base_batch_size=4)
evaluation = dict(interval=1, metric=['bbox', 'segm'])
load_from = 'checkpoint/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'



optimizer = dict(type='AdamW', lr=0.001)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-7)




albu_train_transforms = [
    dict(
       type='ShiftScaleRotate',
       shift_limit=0.1,
       scale_limit=0.1,
       rotate_limit=20,
       interpolation=1,
       p=1),
    
#     dict(
#        type='RandomCrop',
#        crop_size=900,
#        crop_type='absolute',
#        p=1),
    
    dict(type='Cutout', num_holes=4, max_h_size=50, max_w_size=50, fill_value=0, p=0.5),
    dict(type='Emboss', p=0.5),
    
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.7],
        p=0.5),
    dict(
        type='OneOf',
        transforms=[
#             dict(
#                 type='RGBShift',
#                 r_shift_limit=10,
#                 g_shift_limit=10,
#                 b_shift_limit=10,
#                 p=1),
            dict(
                type='HueSaturationValue',
                hue_shift_limit=0,
                sat_shift_limit=20,
                val_shift_limit=20,
                p=1)
        ],
        p=0.5),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
#     dict(type='ChannelShuffle', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0)
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
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    
    
    
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    
    
    
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# 데이터 폴더 설정
# dataset_type = 'COCODataset'
dataset_type = 'CocoDataset'
data_root = 'datasets/lg/'
classes = ('Normal',)

# 데이터 설정
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=12,
    train=dict(
      type=dataset_type,
      img_prefix=data_root + "train/",
      classes = classes,
      ann_file=data_root + "train_custom.json",
      pipeline=train_pipeline)
,
    val=dict(
      type=dataset_type,
      img_prefix=data_root + "train/",
      classes = classes,
      ann_file=data_root + "train_custom.json",
      pipeline=test_pipeline)
,
    test=dict(
      type=dataset_type,
      img_prefix=data_root + "test/",
      classes = classes,
      ann_file=data_root + "label_test_custom.json",
      pipeline=test_pipeline)
)