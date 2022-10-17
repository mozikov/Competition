checkpoint_config = dict(interval=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'checkpoint/detectors_htc_r50_1x_coco-329b1453.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
auto_scale_lr = dict(enable=False, base_batch_size=4)
submit = False
rpn_iou = 0.7
conf = 0.2
rcnn_iou = 0.5
mask_thr = 0.5
model = dict(
    type='HybridTaskCascade',
    backbone=dict(
        type='DetectoRS_ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        conv_cfg=dict(type='ConvAWS'),
        sac=dict(type='SAC', use_deform=True),
        stage_with_sac=(False, True, True, True),
        output_img=True),
    neck=dict(
        type='RFP',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        rfp_steps=2,
        aspp_out_channels=64,
        aspp_dilations=(1, 3, 6, 1),
        rfp_backbone=dict(
            rfp_inplanes=256,
            type='DetectoRS_ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            conv_cfg=dict(type='ConvAWS'),
            sac=dict(type='SAC', use_deform=True),
            stage_with_sac=(False, True, True, True),
            pretrained='torchvision://resnet50',
            style='pytorch')),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='HybridTaskCascadeRoIHead',
        interleaved=True,
        mask_info_flow=True,
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=1,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ],
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=[
            dict(
                type='HTCMaskHead',
                with_conv_res=False,
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)),
            dict(
                type='HTCMaskHead',
                num_convs=4,
                in_channels=256,
                conv_out_channels=256,
                num_classes=1,
                loss_mask=dict(
                    type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.2,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
now = (2022, 7, 29, 10, 32, 20, 4, 210, 0)
bri = 0.2
cont = 0.8
sat = 0.2
ms = 0.3
scale = 0.1
cut = 400
prob = 0.6
hole = 2
epoch = 30
blur = 5
trainall = True
optim = 'sgd'
save = '2022년_7월_29일_10시_32분_20초_detectors50_trainallTrue_bri0.2_cont0.8_sat0.2_mstrain-0.3_scale0.1_cut400_hole2_prob0.6_epoch30_blur5'
work_dir = 'work_dirs/detectors/2022년_7월_29일_10시_32분_20초_detectors50_trainallTrue_bri0.2_cont0.8_sat0.2_mstrain-0.3_scale0.1_cut400_hole2_prob0.6_epoch30_blur5'
runner = dict(type='EpochBasedRunner', max_epochs=30)
interval = 31
evaluation = dict(interval=31, metric=['segm'])
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    min_lr_ratio=1e-05)
albu_train_transforms = [
    dict(
        type='ShiftScaleRotate',
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=360,
        interpolation=0,
        border_mode=0,
        p=0.6),
    dict(
        type='Cutout',
        num_holes=2,
        max_h_size=400,
        max_w_size=400,
        fill_value=0,
        p=0.6),
    dict(type='Emboss', p=0.6),
    dict(
        type='ColorJitter',
        brightness=0.2,
        contrast=0.8,
        saturation=0.2,
        hue=0.0,
        p=0.6),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='HueSaturationValue',
                hue_shift_limit=0,
                sat_shift_limit=0,
                val_shift_limit=0,
                p=1)
        ],
        p=0.6),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=5, p=1.0),
            dict(type='MedianBlur', blur_limit=5, p=1.0),
            dict(type='GaussNoise', var_limit=(10.0, 50), p=1.0)
        ],
        p=0.6)
]
dataset_type = 'CocoDataset'
data_root = 'datasets/lg/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
min_values = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864,
              896, 928, 960, 992)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=(1280, 1024),
        ratio_range=(0.7, 1),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomFlip',
        direction=['horizontal', 'vertical', 'diagonal'],
        flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=360,
                interpolation=0,
                border_mode=0,
                p=0.6),
            dict(
                type='Cutout',
                num_holes=2,
                max_h_size=400,
                max_w_size=400,
                fill_value=0,
                p=0.6),
            dict(type='Emboss', p=0.6),
            dict(
                type='ColorJitter',
                brightness=0.2,
                contrast=0.8,
                saturation=0.2,
                hue=0.0,
                p=0.6),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='HueSaturationValue',
                        hue_shift_limit=0,
                        sat_shift_limit=0,
                        val_shift_limit=0,
                        p=1)
                ],
                p=0.6),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=5, p=1.0),
                    dict(type='MedianBlur', blur_limit=5, p=1.0),
                    dict(type='GaussNoise', var_limit=(10.0, 50), p=1.0)
                ],
                p=0.6)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
img_scale = (1300, 800)
flip_direction = ['horizontal']
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1300, 800),
        flip=True,
        flip_direction=['horizontal'],
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('Normal', )
split = 'train_custom'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
    train=dict(
        type='CocoDataset',
        img_prefix='datasets/lg/train/',
        classes=('Normal', ),
        ann_file='datasets/lg/train_custom.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Resize',
                img_scale=(1280, 1024),
                ratio_range=(0.7, 1),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomFlip',
                direction=['horizontal', 'vertical', 'diagonal'],
                flip_ratio=0.5),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='ShiftScaleRotate',
                        shift_limit=0.1,
                        scale_limit=0.1,
                        rotate_limit=360,
                        interpolation=0,
                        border_mode=0,
                        p=0.6),
                    dict(
                        type='Cutout',
                        num_holes=2,
                        max_h_size=400,
                        max_w_size=400,
                        fill_value=0,
                        p=0.6),
                    dict(type='Emboss', p=0.6),
                    dict(
                        type='ColorJitter',
                        brightness=0.2,
                        contrast=0.8,
                        saturation=0.2,
                        hue=0.0,
                        p=0.6),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='HueSaturationValue',
                                hue_shift_limit=0,
                                sat_shift_limit=0,
                                val_shift_limit=0,
                                p=1)
                        ],
                        p=0.6),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=5, p=1.0),
                            dict(type='MedianBlur', blur_limit=5, p=1.0),
                            dict(
                                type='GaussNoise', var_limit=(10.0, 50), p=1.0)
                        ],
                        p=0.6)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        img_prefix='datasets/lg/train/',
        classes=('Normal', ),
        ann_file='datasets/lg/val_8:2split.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1300, 800),
                flip=True,
                flip_direction=['horizontal'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        img_prefix='datasets/lg/test/',
        classes=('Normal', ),
        ann_file='datasets/lg/label_test_custom.json',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1300, 800),
                flip=True,
                flip_direction=['horizontal'],
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
auto_resume = False
gpu_ids = [0]
