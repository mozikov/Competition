_base_ = '../simple_copy_paste/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco.py'

import time
now = time.localtime()

copy_num = 20

save = '{}년_{}월_{}일_{}시_{}분_{}초_copypaste_copynum{}_82split_noresizecrop'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec, copy_num)
work_dir = 'work_dirs/copypaste/{}'.format(save)


runner = dict(type='EpochBasedRunner', max_epochs=30)
auto_scale_lr = dict(enable=False, base_batch_size=4)
evaluation = dict(interval=1, metric=['segm'])

load_from = 'checkpoint/mask_rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco_20220324_201229-80ee90b7.pth'









# dataset settings
dataset_type = 'CocoDataset'
data_root = 'datasets/lg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
image_size = (1024, 1024)

file_client_args = dict(backend='disk')


albu_train_transforms = [
#     dict(
#        type='ShiftScaleRotate',
# #        shift_limit=0.1,
# #        scale_limit=0.1,
#        rotate_limit=0,
#        interpolation=1,
#        p=0.5),
    
    dict(
       type='HorizontalFlip',
       p=0.5),
    
#     dict(
#        type='VerticalFlip',
#        p=0.5),
    
#     dict(
#        type='RandomCrop',
#        crop_size=900,
#        crop_type='absolute',
#        p=1),
    
#     dict(type='Cutout', num_holes=4, max_h_size=50, max_w_size=50, fill_value=0, p=0.5),
    dict(type='Emboss', p=0.5),
    
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.2],
        contrast_limit=[0.1, 0.2],
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
                val_shift_limit=0,
                p=1)
        ],
        p=0.5),
#     dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.5),
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










# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
load_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(
    #    type='Resize',
    #    img_scale=image_size,
    #    ratio_range=(0.8, 1.25),
    #    multiscale_mode='range',
    #    keep_ratio=True),
    #dict(
    #    type='RandomCrop',
    #    crop_type='absolute_range',
    #    crop_size=image_size,
    #    recompute_bbox=True,
    #    allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size=image_size),
]

train_pipeline = [
    
    
#     dict(type='CopyPaste', max_num_pasted=copy_num),
    
    
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
    
    
#     dict(type='CopyPaste', max_num_pasted=copy_num),
    
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
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

classes = ('Normal',)

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=1,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + "train_8:2split.json",
            img_prefix=data_root + 'train/',
            classes = classes,
            pipeline=load_pipeline),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "val_8:2split.json",
        img_prefix=data_root + 'train/',
        classes = classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "label_test_custom.json",
        img_prefix=data_root + 'test/',
        classes = classes,
        pipeline=test_pipeline))



evaluation = dict(interval=1, metric=['segm'])



# optimizer assumes batch_size = (32 GPUs) x (2 samples per GPU)
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.00004)
optimizer_config = dict(grad_clip=None)

# lr steps at [0.9, 0.95, 0.975] of the maximum iterations
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[10, 20])
checkpoint_config = dict(interval=6000)
# The model is trained by 270k iterations with batch_size 64,
# which is roughly equivalent to 144 epochs.
# runner = dict(type='IterBasedRunner', max_iters=270000)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (32 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=64)




