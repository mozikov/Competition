from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import pandas as pd
import os
import gc
import sys
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import cv2

import scipy as sp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial

import argparse
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW

import datetime
import wandb

import glob

from skimage.morphology import remove_small_objects


# !pip install /kaggle-ink-4th-place/input/files-whl/whlfiles/einops-0.6.1-py3-none-any.whl
# !pip install /kaggle-ink-4th-place/input/files-whl/whlfiles/iopath-0.1.10-py3-none-any.whl
# !pip install /kaggle-ink-4th-place/input/files-whl/whlfiles/fvcore-0.1.5.post20221221-py3-none-any.whl


threshold = 0.47
img_size = 256
in_chans = 22
resnet_depth = 152
weights_path_movinet = '/kaggle-ink-4th-place/input/inchans16-movineta5-longepochs-21252428'
weights_path_resnet152 = 'submitted_weights/2023-06-05-12-22-clip-50-200'
weights_path_resnet200 = 'submitted_weights/2023-06-08-12-22-clip50-200-ls03-resnet200'
weights_path_resnext = 'submitted_weights/2023-06-07-12-22-clip50-200-ls03-resnext101'
weights_path_resnet_tmp = 'weights-tmp'


# LB Best 0.79 4th -> '/kaggle-ink-4th-place/input/2023-06-05-12-22-clip-50-200' & '/kaggle-ink-4th-place/input/2023-06-07-12-22-clip50-200-ls03-resnext101' & '/kaggle-ink-4th-place/input/2023-06-08-12-22-clip50-200-ls03-resnet200' stride 5 no remove
# LB Best 0.79 -> '/kaggle-ink-4th-place/input/2023-06-05-12-22-clip-50-200' & '/kaggle-ink-4th-place/input/2023-06-07-12-22-clip50-200-ls03-resnext101'
# LB Best 0.78 -> '/kaggle-ink-4th-place/input/2023-06-05-12-22-clip-50-200' & thr 0.5, no remove
# LB Best 0.78 -> '/kaggle-ink-4th-place/input/2023-05-30-randomcrop12-22-randompaste-cutout2-50e' & threshold 0.6, remove 0.0003
# resnet solo LB Best 0.77 -> '/kaggle-ink-4th-place/input/2023-05-30-randomcrop12-22-randompaste-cutout2-50e' & threshold 0.6, no remove
# resnet solo LB Best 0.76 -> '/kaggle-ink-4th-place/input/2023-05-28-randomcrop14-18-inchans18-fromstart' (name before modification -> '/kaggle-ink-4th-place/input/2023-05-28-randomcrop16-18-inchans18-fromstart')
# resnet solo LB Best 0.74 -> /kaggle-ink-4th-place/input/inchans16-resnet152-stride3
# movinet LB best 0.63 -> inchans16-movineta5-50epoch
# movinet LB ensemble best 0.72 -> inchans16-movineta5-longepochs-21252428
# resnet LB best 0.71 -> weights-3dcnn-inchans16-resnetdepth152-4fold6281
flip_test = False
rot_test = False

ensemble = True
resnet152_weight = 1
resnet200_weight = 1
resnext_weight = 1
movinet_weight = 1
remove_small_objects_degree_rate = 0.0001  #0.0025
stride_rate = 5

tta_temporal = False
tta_num = 3

clipping = True
clip_min = 50
clip_max = 200
CELOSS = True
remove_small_object = True
batct_size = 16

model_movinet = False
model_resnet152 = True
model_resnet200 = True
model_resnext = True

short_test = False
denoise = False
iter_num = 50



# sys.path.append('/kaggle-ink-4th-place/input/pretrainedmodels/pretrainedmodels-0.7.4')
# sys.path.append('/kaggle-ink-4th-place/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
# sys.path.append('/kaggle-ink-4th-place/input/timm-pytorch-image-models/pytorch-image-models-master')
# sys.path.append('/kaggle-ink-4th-place/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
# sys.path.append('/kaggle-ink-4th-place/input/segmentation/segmentation_models_pytorch_my')
# sys.path.append('/kaggle-ink-4th-place/input/resnet3d')
# sys.path.append('/kaggle-ink-4th-place/input/resnet')
# sys.path.append('/kaggle-ink-4th-place/input/movinet/MoViNet-pytorch')

# import segmentation_models_pytorch as smp
# from resnet3d import *
from resnet import *

sys.path.append('Efficient-3DCNNs')
from models.resnext import resnext101, resnext152

import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

IS_DEBUG = False
mode = 'train' if IS_DEBUG else 'test'
TH = 0.4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    # comp_dataset_path = f'{comp_dir_path}vesuvius-challenge-ink-detection/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'

    exp_name = 'vesuvius_2d_slide_exp002'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b7'
    #     backbone = 'se_resnext50_32x4d'

    in_chans = in_chans  # 65
    # ============== training cfg =============
    size = img_size
    tile_size = img_size
    stride = tile_size // stride_rate

    batch_size = batct_size  # 32
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 15

    warmup_factor = 10
    lr = 1e-4 / warmup_factor

    # ============== fold =============
    valid_id = 2

    objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 0

    seed = 42

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3),
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        #         A.Resize(size, size),
        A.Normalize(
            mean=[0] * in_chans,
            std=[1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


# ref.: https://www.kaggle.com/stainsby/fast-tested-rle
def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    # pixels = (pixels >= thr).astype(int)

    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def read_image(fragment_id):
    images = []
    masks = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)
    #     idxs = [14, 15] + list(idxs)

    for i in tqdm(idxs):
        image = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/surface_volume/{i:02}.tif", 0)
        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)
        #         image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        images.append(image)

    mask = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/mask.png", 0)
    pad0 = (CFG.tile_size - mask.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - mask.shape[1] % CFG.tile_size)
    #     mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    images = np.stack(images, axis=2)

    return images, mask


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug


class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform

    def __len__(self):
        # return len(self.xyxys)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.labels[idx]

        if clipping:
            image = np.clip(image, clip_min, clip_max)

        image_horizontal_flip = A.Compose([
            A.HorizontalFlip(p=1),
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0] * in_chans,
                std=[1] * in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ])(image=image.copy())['image'].unsqueeze(0)

        tta_total = []
        if tta_temporal:
            for i in range(tta_num):
                left = (in_chans // 2) - ((in_chans - 2 * i) // 2)
                right = (in_chans // 2) + ((in_chans - 2 * i) // 2)

                image_tmp = np.zeros_like(image)
                image_tmp[..., 0: (in_chans - 2 * i)] = image[..., left: right]
                data_tmp = self.transform(image=image_tmp)
                image_tmp = data_tmp['image'].unsqueeze(0)
                tta_total.append(image_tmp)

            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

            return image, image_horizontal_flip, tta_total, mask

        else:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

            return image, image_horizontal_flip, [], mask


def make_test_dataset(fragment_id):
    test_images, mask = read_image(fragment_id)

    x1_list = list(range(0, test_images.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, test_images.shape[0] - CFG.tile_size + 1, CFG.stride))

    test_images_list = []
    test_masks_list = []
    xyxys = []
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size

            test_images_list.append(test_images[y1:y2, x1:x2])
            test_masks_list.append(mask[y1:y2, x1:x2])
            xyxys.append((x1, y1, x2, y2))
    xyxys = np.stack(xyxys)

    test_dataset = CustomDataset(test_images_list, CFG, test_masks_list,
                                 transform=get_transforms(data='valid', cfg=CFG))

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.batch_size,
                             shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    return test_loader, xyxys


class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        self.encoder = smp.UnetPlusPlus(
            encoder_name=cfg.backbone,
            encoder_weights=weight,
            in_channels=cfg.in_chans,
            classes=cfg.target_size,
            activation=None,
        )

    def forward(self, image):
        output = self.encoder(image)
        output = output.squeeze(-1)
        return output


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i - 1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        if CELOSS:
            self.logit = nn.Conv2d(encoder_dims[0], 2, 1, 1, 0)
        else:
            self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)

        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps) - 1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i - 1], f_up], dim=1)
            f_down = self.convs[i - 1](f)
            feature_maps[i - 1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self, depth):
        super().__init__()
        #         self.encoder = generate_model(model_depth=18, n_input_channels=1)

        self.depth = depth

        self.encoder = generate_model(model_depth=self.depth,
                                      n_input_channels=1,
                                      shortcut_type='B',
                                      conv1_t_size=7,
                                      conv1_t_stride=1,
                                      widen_factor=1.0,
                                      n_classes=1039,
                                      no_max_pool=True)

        #         self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)
        self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))


class SegModel_resnext101(nn.Module):
    def __init__(self):
        super().__init__()
        # original kaggle-ink-4th-place code
        # self.encoder = generate_model(model_depth=args.resnet_depth, n_input_channels=1)

        # original paper code
        self.encoder = resnext101(sample_size=112,
                                  sample_duration=16,
                                  shortcut_type='B',
                                  cardinality=32,
                                  num_classes=600)

        # original paper code
        self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask


class SegModel_resnext152(nn.Module):
    def __init__(self):
        super().__init__()
        # original kaggle-ink-4th-place code
        # self.encoder = generate_model(model_depth=args.resnet_depth, n_input_channels=1)

        # original paper code
        self.encoder = resnext152(sample_size=112,
                                  sample_duration=16,
                                  shortcut_type='B',
                                  cardinality=32,
                                  num_classes=600)

        # original paper code
        self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask


def build_model(cfg, weight="imagenet"):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    # model = CustomModel(cfg, weight)
    model = SegModel()
    return model


# from movinets import MoViNet
# from movinets.config import _C

# model = MoViNet(_C.MODEL.MoViNetA5, causal=False, pretrained=False)
# model.conv1.conv_1.conv3d = nn.Conv3d(1, 24, kernel_size=(1, 3, 3), stride=(1, 2, 2), bias=False)
# layers = list(model.children())


# class Decoder(nn.Module):
#     def __init__(self, encoder_dims, upscale):
#         super().__init__()
#         self.convs = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(encoder_dims[i] + encoder_dims[i - 1], encoder_dims[i - 1], 3, 1, 1, bias=False),
#                 nn.BatchNorm2d(encoder_dims[i - 1]),
#                 nn.ReLU(inplace=True)
#             ) for i in range(1, len(encoder_dims))])

#         self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
#         self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

#     def forward(self, feature_maps):
#         for i in range(len(feature_maps) - 1, 0, -1):
#             f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
#             f = torch.cat([feature_maps[i - 1], f_up], dim=1)
#             f_down = self.convs[i - 1](f)
#             feature_maps[i - 1] = f_down

#         x = self.logit(feature_maps[0])
#         mask = self.up(x)
#         return mask

# class MoviNetA5_modified(nn.Module):
#     def __init__(self, layers):
#         super().__init__()
#
#         # A3
#         #         block2_num = 4
#         #         block3_num = 6
#         #         block4_num = 5
#         #         block5_num = 8
#         #         block6_num = 10
#
#         # A5
#         block2_num = 6
#         block3_num = 11
#         block4_num = 13
#         block5_num = 11
#         block6_num = 18
#
#         self.layers = layers
#
#         self.block0 = self.layers[0]
#
#         self.block2 = self.layers[1][:block2_num]
#         self.block3 = self.layers[1][block2_num: block2_num + block3_num]
#         self.block4 = self.layers[1][block2_num + block3_num: block2_num + block3_num + block4_num]
#         self.block5 = self.layers[1][
#                       block2_num + block3_num + block4_num: block2_num + block3_num + block4_num + block5_num]
#         self.block6 = self.layers[1][
#                       block2_num + block3_num + block4_num + block5_num: block2_num + block3_num + block4_num + block5_num + block6_num]
#
#         #         self.decoder = Decoder(encoder_dims=[16, 48, 88, 168], upscale=4)
#         self.decoder = Decoder(encoder_dims=[24, 64, 120, 224], upscale=4)
#
#     def forward(self, x):
#         x = self.block0(x)
#
#         x2 = self.block2(x)
#         x3 = self.block3(x2)
#         x4 = self.block4(x3)
#         x5 = self.block5(x4)
#         x6 = self.block6(x5)
#
#         x_total = [x2, x3, x5, x6]
#
#         x_total = [torch.mean(f, dim=2) for f in x_total]
#
#         result = self.decoder(x_total)
#
#         return result

# model = MoviNetA5_modified(layers)
# model = model.to(device)


class EnsembleModel:
    def __init__(self, use_tta=False):
        self.models = []
        self.use_tta = use_tta

    def __call__(self, x):

        if ensemble:
            total = []
            for i in range(len(self.models)):
                model = self.models[i]
                if i <= 3:
                    if CELOSS:
                        pred = (torch.softmax(model(x), 1))[:, 1, ...].to('cpu').detach().numpy() * resnext_weight
                    else:
                        pred = (torch.sigmoid(model(x))).to('cpu').detach().numpy() * resnext_weight


                elif i > 3 and i <= 7:

                    if CELOSS:
                        pred = (torch.softmax(model(x), 1))[:, 1, ...].to('cpu').detach().numpy() * resnet152_weight
                    else:
                        pred = (torch.sigmoid(model(x))).to('cpu').detach().numpy() * resnet152_weight


                else:

                    if CELOSS:
                        pred = (torch.softmax(model(x), 1))[:, 1, ...].to('cpu').detach().numpy() * resnet200_weight
                    else:
                        pred = (torch.sigmoid(model(x))).to('cpu').detach().numpy() * resnet200_weight

                total.append(pred)
            outputs = total


        else:
            if CELOSS:
                outputs = [torch.softmax(model(x), 1)[:, 1, ...].to('cpu').detach().numpy() for model in self.models]
            else:
                outputs = [torch.sigmoid(model(x)).to('cpu').detach().numpy() for model in self.models]

        avg_preds = np.mean(outputs, axis=0)
        return avg_preds

    def add_model(self, model):
        self.models.append(model)


movinet_weights_list = glob.glob(os.path.join(weights_path_movinet, '*.pth'))
resnet152_weights_list = glob.glob(os.path.join(weights_path_resnet152, '*.pth'))
resnet200_weights_list = glob.glob(os.path.join(weights_path_resnet200, '*.pth'))
resnext_weights_list = glob.glob(os.path.join(weights_path_resnext, '*.pth'))


def build_ensemble_model():
    model = EnsembleModel()

    if model_movinet:

        for weights in movinet_weights_list:
            _model = MoviNetA5_modified(layers)
            _model.to(device)

            state = torch.load(weights)['model']

            _model.load_state_dict(state)
            _model.eval()

            model.add_model(_model)

    if model_resnext:
        for weights in resnext_weights_list:
            _model = SegModel_resnext101()
            _model.encoder.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3),
                                             bias=False)
            _model.to(device)

            state = torch.load(weights)['model']
            _model.load_state_dict(state)
            _model.eval()

            model.add_model(_model)

    if model_resnet152:
        for weights in resnet152_weights_list:
            _model = SegModel(152)
            _model.to(device)

            state = torch.load(weights)['model']
            _model.load_state_dict(state)
            _model.eval()

            model.add_model(_model)

    if model_resnet200:
        for weights in resnet200_weights_list:
            _model = SegModel(200)
            _model.to(device)

            state = torch.load(weights)['model']
            _model.load_state_dict(state)
            _model.eval()

            model.add_model(_model)

    #     for i in range(4):

    #         _model = SegModel()
    #         _model.to(device)

    #         if i == 0:
    #             weights = '/kaggle-ink-4th-place/input/3fold-balance-fold1-tmp/Unet_fold1_epoch20_score0.6237474369132259.pth'
    #         elif i == 1:
    #             weights = '/kaggle-ink-4th-place/input/weights-3dcnn-inchans16-resnetdepth152-4fold6281/Unet_fold2_best.pth'
    #         elif i == 2:
    #             weights = '/kaggle-ink-4th-place/input/weights-3dcnn-inchans16-resnetdepth152-4fold6281/Unet_fold3_best.pth'
    #         else:
    #             weights = '/kaggle-ink-4th-place/input/weights-3dcnn-inchans16-resnetdepth152-4fold6281/Unet_fold4_best.pth'

    return model


def TTA(x:torch.Tensor,model:nn.Module):
    #x.shape=(batch,c,h,w)
    shape=x.shape
    x=[x,*[torch.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)]]
    x=torch.cat(x,dim=0)
    x=model(x)
    x=torch.from_numpy(x)
    x=x.reshape(4,shape[0],1,*shape[-2:])
    x=[torch.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x=torch.stack(x,dim=0)
    return x.mean(0).numpy()


if mode == 'test':
    fragment_ids = sorted(os.listdir(CFG.comp_dataset_path + mode))
else:
    fragment_ids = [3]

model = build_ensemble_model()



# import cupy as cp
#
# xp = cp
# # xp = np
#
# delta_lookup = {
#     "xx": xp.array([[1, -2, 1]], dtype=float),
#     "yy": xp.array([[1], [-2], [1]], dtype=float),
#     "xy": xp.array([[1, -1], [-1, 1]], dtype=float),
# }
#
#
# def operate_derivative(img_shape, pair):
#     assert len(img_shape) == 2
#     delta = delta_lookup[pair]
#     fft = xp.fft.fftn(delta, img_shape)
#     return fft * xp.conj(fft)
#
#
# def soft_threshold(vector, threshold):
#     return xp.sign(vector) * xp.maximum(xp.abs(vector) - threshold, 0)
#
#
# def back_diff(input_image, dim):
#     assert dim in (0, 1)
#     r, n = xp.shape(input_image)
#     size = xp.array((r, n))
#     position = xp.zeros(2, dtype=int)
#     temp1 = xp.zeros((r + 1, n + 1), dtype=float)
#     temp2 = xp.zeros((r + 1, n + 1), dtype=float)
#
#     temp1[position[0]:size[0], position[1]:size[1]] = input_image
#     temp2[position[0]:size[0], position[1]:size[1]] = input_image
#
#     size[dim] += 1
#     position[dim] += 1
#     temp2[position[0]:size[0], position[1]:size[1]] = input_image
#     temp1 -= temp2
#     size[dim] -= 1
#     return temp1[0:size[0], 0:size[1]]
#
#
# def forward_diff(input_image, dim):
#     assert dim in (0, 1)
#     r, n = xp.shape(input_image)
#     size = xp.array((r, n))
#     position = xp.zeros(2, dtype=int)
#     temp1 = xp.zeros((r + 1, n + 1), dtype=float)
#     temp2 = xp.zeros((r + 1, n + 1), dtype=float)
#
#     size[dim] += 1
#     position[dim] += 1
#
#     temp1[position[0]:size[0], position[1]:size[1]] = input_image
#     temp2[position[0]:size[0], position[1]:size[1]] = input_image
#
#     size[dim] -= 1
#     temp2[0:size[0], 0:size[1]] = input_image
#     temp1 -= temp2
#     size[dim] += 1
#     return -temp1[position[0]:size[0], position[1]:size[1]]
#
#
# def iter_deriv(input_image, b, scale, mu, dim1, dim2):
#     g = back_diff(forward_diff(input_image, dim1), dim2)
#     d = soft_threshold(g + b, 1 / mu)
#     b = b + (g - d)
#     L = scale * back_diff(forward_diff(d - b, dim2), dim1)
#     return L, b
#
#
# def iter_xx(*args):
#     return iter_deriv(*args, dim1=1, dim2=1)
#
#
# def iter_yy(*args):
#     return iter_deriv(*args, dim1=0, dim2=0)
#
#
# def iter_xy(*args):
#     return iter_deriv(*args, dim1=0, dim2=1)
#
#
# def iter_sparse(input_image, bsparse, scale, mu):
#     d = soft_threshold(input_image + bsparse, 1 / mu)
#     bsparse = bsparse + (input_image - d)
#     Lsparse = scale * (d - bsparse)
#     return Lsparse, bsparse
#
#
# def denoise_image(input_image, iter_num=100, fidelity=150, sparsity_scale=10, continuity_scale=0.5, mu=1):
#     image_size = xp.shape(input_image)
#     # print("Initialize denoising")
#     norm_array = (
#             operate_derivative(image_size, "xx") +
#             operate_derivative(image_size, "yy") +
#             2 * operate_derivative(image_size, "xy")
#     )
#     norm_array += (fidelity / mu) + sparsity_scale ** 2
#     b_arrays = {
#         "xx": xp.zeros(image_size, dtype=float),
#         "yy": xp.zeros(image_size, dtype=float),
#         "xy": xp.zeros(image_size, dtype=float),
#         "L1": xp.zeros(image_size, dtype=float),
#     }
#     g_update = xp.multiply(fidelity / mu, input_image)
#     for i in tqdm(range(iter_num), total=iter_num):
#         # print(f"Starting iteration {i+1}")
#         g_update = xp.fft.fftn(g_update)
#         if i == 0:
#             g = xp.fft.ifftn(g_update / (fidelity / mu)).real
#         else:
#             g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
#         g_update = xp.multiply((fidelity / mu), input_image)
#
#         # print("XX update")
#         L, b_arrays["xx"] = iter_xx(g, b_arrays["xx"], continuity_scale, mu)
#         g_update += L
#
#         # print("YY update")
#         L, b_arrays["yy"] = iter_yy(g, b_arrays["yy"], continuity_scale, mu)
#         g_update += L
#
#         # print("XY update")
#         L, b_arrays["xy"] = iter_xy(g, b_arrays["xy"], 2 * continuity_scale, mu)
#         g_update += L
#
#         # print("L1 update")
#         L, b_arrays["L1"] = iter_sparse(g, b_arrays["L1"], sparsity_scale, mu)
#         g_update += L
#
#     g_update = xp.fft.fftn(g_update)
#     g = xp.fft.ifftn(xp.divide(g_update, norm_array)).real
#
#     g[g < 0] = 0
#     g -= g.min()
#     g /= g.max()
#     return g

results = []
for fragment_id in fragment_ids:

    test_loader, xyxys = make_test_dataset(fragment_id)

    binary_mask = cv2.imread(CFG.comp_dataset_path + f"{mode}/{fragment_id}/mask.png", 0)
    binary_mask = (binary_mask / 255).astype(int)

    ori_h = binary_mask.shape[0]
    ori_w = binary_mask.shape[1]
    # mask = mask / 255

    pad0 = (CFG.tile_size - binary_mask.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - binary_mask.shape[1] % CFG.tile_size)

    #     binary_mask = np.pad(binary_mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask_pred = np.zeros(binary_mask.shape)
    mask_count = np.zeros(binary_mask.shape)

    for step, (images, image_horizontal_flip, images_tta, mask) in tqdm(enumerate(test_loader), total=len(test_loader)):

        if short_test:
            if step == 1:
                break

        if mask.max() == 0:
            continue

        images = images.to(device)
        image_horizontal_flip = image_horizontal_flip.to(device)

        if tta_temporal:
            images_tta = [i.to(device) for i in images_tta]

        batch_size = images.size(0)

        with torch.no_grad():

            if flip_test:
                y_preds_org = model(images)
                y_preds_horizontal_flip = model(image_horizontal_flip)  # (batch_size, 1, 16, 256, 256)
                y_preds_horizontal_flip = [cv2.flip(i[0], 1)[None, ...] for i in y_preds_horizontal_flip]
                y_preds_horizontal_flip = np.stack(y_preds_horizontal_flip)

                y_preds = (y_preds_org + y_preds_horizontal_flip) / 2

            elif rot_test:
                y_preds = TTA(images, model)

            else:

                if tta_temporal:
                    y_preds_tta = []
                    for k in range(len(images_tta)):
                        y_preds_tta.append(model(images_tta[k]))

                    y_preds_tta = np.concatenate(y_preds_tta, 1).mean(1)[:, None, :, :]
                    y_preds = (y_preds_tta) / len(images_tta)

                else:
                    y_preds = model(images)

        start_idx = step * CFG.batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(xyxys[start_idx:end_idx]):
            if CELOSS:
                mask_pred[y1:y2, x1:x2] += y_preds[i]  # .squeeze(0)
            else:
                mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    mask_pred /= (mask_count + (1e-7))

    if denoise:
        mask_pred = xp.array(mask_pred)
        mask_pred = denoise_image(mask_pred, iter_num=iter_num)
        mask_pred = mask_pred.get()

    mask_pred = mask_pred[:ori_h, :ori_w]
    binary_mask = binary_mask[:ori_h, :ori_w]

    remove_small_objects_degree = int((mask_pred.shape[0] * mask_pred.shape[1]) * remove_small_objects_degree_rate)

    if remove_small_object:
        mask_pred = remove_small_objects(mask_pred >= threshold, remove_small_objects_degree)
    else:
        mask_pred = remove_small_objects(mask_pred >= threshold, 1)

    mask_pred = mask_pred.astype(int)
    mask_pred *= binary_mask

    inklabels_rle = rle(mask_pred)

    results.append((fragment_id, inklabels_rle))

    del mask_pred, mask_count
    del test_loader

    gc.collect()
    torch.cuda.empty_cache()


sub = pd.DataFrame(results, columns=['Id', 'Predicted'])

sub

sample_sub = pd.read_csv(CFG.comp_dataset_path + 'sample_submission.csv')
sample_sub = pd.merge(sample_sub[['Id']], sub, on='Id', how='left')

sample_sub

sample_sub.to_csv("submission.csv", index=False)

