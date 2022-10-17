# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))


# 데이터 폴더 설정
# dataset_type = 'COCODataset'
dataset_type = 'CocoDataset'
classes = ('Normal',)

# 데이터 설정
data = dict(
    train=dict(
      type=dataset_type,
      img_prefix='datasets/lg/train/',
      classes = classes,
      ann_file='datasets/lg/train_custom.json'),
    val=dict(
      type=dataset_type,
      img_prefix='datasets/lg/train/',
      classes = classes,
      ann_file='datasets/lg/train_custom.json'),
    test=dict(
      type=dataset_type,
      img_prefix='datasets/lg/train/',
      classes = classes,
      ann_file='datasets/lg/train_custom.json'))

# # log 저장 위치
# checkpoint_config = dict(insterval=1,out_dir='work_dirs/lg_mask/')

# # 평가 방법
# evaluation = dict(interval=1, metric=['bbox', 'segm'])

# # 사전 가중치 사용
# # load_from = 'checkpoint/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'

# # epoch 설정 
# runner = dict(type='EpochBasedRunner', max_epochs=5)

# # batch size 설정
# auto_scale_lr = dict(enable=False, base_batch_size=16)