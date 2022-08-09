# The new config inherits a base config to highlight the necessary modification

_base_ = '../swin/cascade_rcnn_swin-t-p4-w7_fpn_1x_coco.py'


# Modify dataset related settings
dataset_type = 'COCODataset'
#dataset_type = 'MultiImageMixDataset'

#classes = ('balloon',)
classes = ('01_ulcer', '02_mass', '04_lymph', '05_bleeding')


data = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    train=dict(
        img_prefix='data/dacon/train_all/',
        classes=classes,
        ann_file='data/dacon/train_all/train_annotations.json'),
    val=dict(
        img_prefix='data/dacon/val/',
        classes=classes,
        ann_file='data/dacon/val/val_annotations4.json'),
    test=dict(
        img_prefix='data/dacon/test_image/',
        classes=classes,
        ann_file='data/dacon/test_annotations.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance

#load_from = 'work_dirs/Swin_train_allimages_from25to/epoch_11.pth'
work_dir = 'work_dirs/Swin_train_allimages_allaug_last'