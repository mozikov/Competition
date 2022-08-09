#_base_ = '../detectors/detectors_htc_r50_1x_coco.py'
# _base_ = '../detectors/detectors_htc_r101_20e_coco.py'

import time
now = time.localtime()

bri = 0.2
cont = 0.8
sat = 0.2
ms = 0.3
scale = 0.1
cut = 400
prob = 0.6
hole = 2
epoch = 100
blur = 7
trainall = False
optim = 'sgd'
submit = True
det50 = True


if det50:
    model_ = 'detectors50'
    load_from = 'checkpoint/detectors_htc_r50_1x_coco-329b1453.pth'
    _base_ = '../detectors/detectors_htc_r50_1x_coco.py'

else:
    model_ = 'detectors101'
    load_from = 'checkpoint/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'
    _base_ = '../detectors/detectors_htc_r101_20e_coco.py'

save = '{}년_{}월_{}일_{}시_{}분_{}초_{}_trainall{}_bri{}_cont{}_sat{}_mstrain-{}_scale{}_cut{}_hole{}_prob{}_epoch{}_blur{}_optim{}_[(1280, 1024)]'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, model_, trainall, bri, cont, sat, ms, scale, cut, hole, prob, epoch, blur, optim)
work_dir = 'work_dirs/detectors/{}'.format(save)


runner = dict(type='EpochBasedRunner', max_epochs=epoch)
auto_scale_lr = dict(enable=False, base_batch_size=4)

if trainall:
    interval=epoch+1
else:
    interval=1
    
    
evaluation = dict(interval=interval, metric=['segm'])

# load_from = 'checkpoint/detectors_htc_r50_1x_coco-329b1453.pth'
# load_from = 'checkpoint/detectors_htc_r101_20e_coco_20210419_203638-348d533b.pth'




if optim == 'sgd':
    optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
elif optim == 'adamw':
    optimizer = dict(type='AdamW', lr=0.001)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
  policy='CosineAnnealing',
  warmup='linear',
  warmup_iters=500,
  warmup_ratio=1.0 / 10,
  min_lr_ratio=1e-6)

# lr_config = dict(
#      policy='step',
#      warmup='linear',
#      warmup_iters=500,
#      warmup_ratio=0.001,
#      step=[10, 20, 30, 40])





albu_train_transforms = [
    dict(
       type='ShiftScaleRotate',
       shift_limit=scale,
       scale_limit=scale,
       rotate_limit=360,
       interpolation=0,
       border_mode=0,
       p=prob),
    
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
    
    dict(type='Cutout', num_holes=hole, max_h_size=cut, max_w_size=cut, fill_value=0, p=prob),
    
#     dict(type='CoarseDropout', max_holes=4, max_height=200, max_width=200, 
#                                min_holes=2, min_height=100, min_width=100, fill_value=0, p=0.5),
    
    dict(type='Emboss', p=prob),
    
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
        p=prob),
    
    
#     dict(
#         type='OneOf',
#         transforms=[
#              dict(
#                  type='RGBShift',
#                  r_shift_limit=10,
#                  g_shift_limit=10,
#                  b_shift_limit=10,
#                  p=1),
#             dict(
#                 type='HueSaturationValue',
#                 hue_shift_limit=0,
#                 sat_shift_limit=0,
#                 val_shift_limit=0,
#                 p=1)
#         ],
#         p=prob),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=prob),
#     dict(type='ChannelShuffle', p=prob),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=blur, p=1.0),
            dict(type='MedianBlur', blur_limit=blur, p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, blur*10), p=1.0),
        ],
        p=prob),
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
#         img_scale=(1248, 1024), #[(1024, value) for value in min_values],
        img_scale=(1280, 1024), #[(1024, value) for value in min_values],
        ratio_range=(1 - ms, 1),
        multiscale_mode='range',
        keep_ratio=True),
    
    

#     dict(
#         type='AutoAugment',
#         policies=[[
#             dict(
#                 type='Resize',
#                 img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
#                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
#                            (736, 1333), (768, 1333), (800, 1333)],
#                 multiscale_mode='value',
#                 keep_ratio=True)
#         ],
#                   [
#                       dict(
#                           type='Resize',
#                           img_scale=[(400, 1333), (500, 1333), (600, 1333)],
#                           multiscale_mode='value',
#                           keep_ratio=True),
#                       dict(
#                           type='RandomCrop',
#                           crop_type='absolute_range',
#                           crop_size=(384, 600),
#                           allow_negative_crop=True),
#                       dict(
#                           type='Resize',
#                           img_scale=[(480, 1333), (512, 1333), (544, 1333),
#                                      (576, 1333), (608, 1333), (640, 1333),
#                                      (672, 1333), (704, 1333), (736, 1333),
#                                      (768, 1333), (800, 1333)],
#                           multiscale_mode='value',
#                           override=True,
#                           keep_ratio=True)
#                   ]]),
    
    
    

    dict(type='RandomFlip', 
         direction=['horizontal', 'vertical', 'diagonal'],
         flip_ratio=prob),
    
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
    
    
    
#     dict(type='CopyPaste', max_num_pasted=10),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

#0.6402177648
if submit:
    img_scale = [(1500, 1178), (1400, 1094), (1300, 1016), (1200, 938), (1280, 1024), (1024, 800)]
    flip_direction = ['horizontal', 'vertical', 'diagonal']


# if submit:
#     img_scale = [(1500, 1178), (1400, 1094), (1300, 1016), (1280, 1024), (1200, 940), (1100, 860), (1024, 800)]
#     flip_direction = ['horizontal', 'vertical', 'diagonal']
else:
    img_scale=(1280, 1024)
    flip_direction = ['horizontal']

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
#        img_scale=[(1424, 1224), (1224, 1024), (1024, 800)],
        
        #img_scale=[(1300, 1016), (1200, 938), (1024, 800)], #best
#         img_scale=[(1024, 800), (800, 640), (736, 576)],
        
        img_scale=img_scale,
        flip=True,
        flip_direction=flip_direction,
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


if trainall:
    split = 'train_custom'
else:
    split = 'train_8:2split'
    
# 데이터 설정
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
    train=dict(
#       type='MultiImageMixDataset',
      type=dataset_type,
      img_prefix=data_root + "train/",
      classes = classes,
      ann_file=data_root + "{}.json".format(split),
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



