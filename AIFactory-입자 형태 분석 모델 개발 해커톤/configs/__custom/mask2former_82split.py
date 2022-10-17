_base_ = '../mask2former/mask2former_r101_lsj_8x2_50e_coco.py'




import time
now = time.localtime()

cont=1
sat=0

save = '{}년_{}월_{}일_{}시_{}분_{}초_mask2former101_82split'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
work_dir = 'work_dirs/mask2former/{}'.format(save)


runner = dict(type='EpochBasedRunner', max_epochs=30)
# auto_scale_lr = dict(enable=False, base_batch_size=4)
# evaluation = dict(interval=1, metric=['segm'])

evaluation = dict(
    interval=31,
#     dynamic_intervals=dynamic_intervals,
    metric=['segm'])

load_from = 'checkpoint/mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth'



# optimizer = dict(type='AdamW', lr=0.0001)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-6)








albu_train_transforms = [
    dict(
       type='ShiftScaleRotate',
       shift_limit=0.1,
       scale_limit=0.1,
       rotate_limit=360,
       interpolation=0,
       border_mode=0,
       p=1),
    
#     dict(
#        type='RandomCrop',
#        crop_size=900,
#        crop_type='absolute',
#        p=1),
    
    dict(type='Cutout', num_holes=4, max_h_size=200, max_w_size=200, fill_value=0, p=0.5),
    dict(type='Emboss', p=0.5),
    
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.2],
#         contrast_limit=[0.1, 0.7],
#         p=0.5),
    
    dict(
        type='ColorJitter',
        brightness=0.2, 
        contrast=cont,
        saturation=sat, 
        hue=0.0,
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
                sat_shift_limit=0,
                val_shift_limit=0,
                p=1)
        ],
        p=0.5),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
#     dict(type='ChannelShuffle', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0)
        ],
        p=0.5),
]


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    
    dict(type='Resize',
        img_scale=(1024, 800), #[(1024, value) for value in min_values],
        ratio_range=(0.9, 1.1),
        multiscale_mode='range',
        keep_ratio=True),
    
    
    
    dict(type='RandomFlip', 
         direction=['horizontal', 'vertical', 'diagonal'],
         flip_ratio=0.5),
    
    
    
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
dataset_type = 'CocoDataset'
data_root = 'datasets/lg/'
classes = ('Normal',)

# 데이터 설정
data = dict(
    samples_per_gpu=2,
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
      ann_file=data_root + "val_8:2split.json",
      pipeline=test_pipeline)
,
    test=dict(
      type=dataset_type,
      img_prefix=data_root + "test/",
      classes = classes,
      ann_file=data_root + "label_test_custom.json",
      pipeline=test_pipeline)
)