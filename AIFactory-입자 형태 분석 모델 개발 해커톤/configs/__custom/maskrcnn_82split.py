# _base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'
#_base_ = '../mask_rcnn/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco.py'

# 모델 class수 변경
# model = dict(
#   roi_head = dict(
#     bbox_head = dict(
#       num_classes = 1
#     ),
#     mask_head = dict(
#       num_classes = 1
#     )
#   )
# )

log_level = 'INFO'
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
resume_from = None
dist_params = dict(backend='nccl')

import time
now = time.localtime()

bri = 0.5
cont = 1
sat = 0
ms = 0
scale = 0

save = '{}년_{}월_{}일_{}시_{}분_{}초_MaskRCNN'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
work_dir = 'work_dirs/maskrcnn/{}'.format(save)


runner = dict(type='EpochBasedRunner', max_epochs=30)
auto_scale_lr = dict(enable=False, base_batch_size=4)
evaluation = dict(interval=1, metric=['segm'])
load_from = 'checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'
#load_From = 'checkpoint/mask_rcnn_x101_64x4d_fpn_mstrain-poly_3x_coco_20210526_120447-c376f129.pth'



#optimizer = dict(type='AdamW', lr=0.0001)
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
   policy='CosineAnnealing',
   warmup='linear',
   warmup_iters=500,
   warmup_ratio=1.0 / 10,
   min_lr_ratio=1e-5)

# lr_config = dict(
#      policy='step',
#      warmup='linear',
#      warmup_iters=500,
#      warmup_ratio=0.001,
#      step=[10, 15, 20, 25])





albu_train_transforms = [
    dict(
       type='ShiftScaleRotate',
       shift_limit=scale,
       scale_limit=scale,
       rotate_limit=360,
       interpolation=0,
       border_mode=0,
       p=0.5),
    
#     dict(
#        type='HorizontalFlip',
#        p=0.5),
    
#     dict(
#        type='VerticalFlip',
#        p=0.5),
    
#     dict(
#        type='RandomCrop',
#        width=800,
#        height=700,
#        p=0.5),
    
    dict(type='Cutout', num_holes=4, max_h_size=200, max_w_size=200, fill_value=0, p=0.5),
    #dict(type='Emboss', p=0.5),
    
#     dict(
#         type='RandomBrightnessContrast',
#         brightness_limit=[0.1, 0.2],
#         contrast_limit=[0.1, 0.7],
#         p=0.5),
    
    dict(
        type='ColorJitter',
        brightness=bri, 
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
#    dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
#    dict(type='ChannelShuffle', p=0.5),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, 50.0), p=1.0),
        ],
        p=0.5),
]


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    
    dict(type='Resize',
        img_scale=(1024, 800), #[(1024, value) for value in min_values],
        ratio_range=(1 - ms, 1 + ms),
        multiscale_mode='range',
        keep_ratio=True),
    
#    dict(type='Resize',
#         img_scale=[(2000, 2000), (1800, 1800)],
#         multiscale_mode='value',
#         keep_ratio=True),
    
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
    
    
    
#     dict(type='CopyPaste', max_num_pasted=100),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#        img_scale=[(1424, 1224), (1224, 1024), (1024, 800)],
        
        img_scale=[(1300, 1016), (1200, 938), (1024, 800)], #best
        
#        img_scale=(1300, 800),
        flip=True,
        flip_direction=['horizontal', 'vertical', 'diagonal'],
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
    samples_per_gpu=2,
    workers_per_gpu=12,
    train=dict(
      #type='MultiImageMixDataset',
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



