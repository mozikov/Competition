#!/usr/bin/env python
# coding: utf-8

# In[1]:


import segmentation_models_pytorch as smp

import torch

import argparse
import logging
import os
import sys
import datetime
import glob
import cv2
import matplotlib.pyplot as plt
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import albumentations as A
import math

    
import torch.cuda.amp as amp
import torch.nn as nn

import zipfile





device = 'cuda'

model_name = 'efficientnet-b7'
model = smp.unetplusplus.UnetPlusPlus(encoder_name="{}".format(model_name),
                      encoder_weights='imagenet',
                      in_channels=3,
                      classes=3).to(device)


test_path = sorted(glob.glob('datasets/test/lr/*'))





total = np.zeros(shape=(5, 18, 2048, 2048, 3))


for i in range(5):
    
    model.load_state_dict(torch.load('checkpoint/Fold{}.pth'.format(i+1)))
    model.eval()
    
    fold = []
    for j in range(len(test_path)):

        print(j+1)

        lr_path = test_path[j]
        img = cv2.imread(lr_path)[..., ::-1]
        img = cv2.resize(img, (2048, 2048))


        img = (img / 255)

        patch_size = 768
        stride = int(patch_size / 3)
        batch_size = 1

        crop = []
        position = []
        batch_count = 0

        result_img = np.zeros(shape=(img.shape[0], img.shape[1], 3))
        voting_mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))

        
        for top in range(0, img.shape[0], stride):
            for left in range(0, img.shape[1], stride):
                piece = np.ones([patch_size, patch_size, 3], np.float32)
                temp = img[top : top + patch_size, left : left + patch_size, :]
                piece[:temp.shape[0], :temp.shape[1], :] = temp
                crop.append(piece)
                position.append([top, left])
                batch_count += 1
                if batch_count == batch_size:
                    input_ = np.array(crop)            

                    input_ = torch.from_numpy(input_.transpose(0, 3, 1, 2)).to(device)

                    with torch.no_grad():
                        pred = model(input_)

                    pred = (pred + input_)
                    pred = pred.cpu().detach().numpy().transpose(0, 2, 3, 1).astype('float')

                    for num, (top, left) in enumerate(position):
                        piece = pred[num]
                        piece = np.clip(piece, 0, 1)

                        h, w, c = result_img[(top):(top) + (patch_size), (left):(left) + (patch_size)].shape

                        result_img[(top):(top) + (patch_size), (left):(left) + (patch_size)] += piece[:h, :w]

                        voting_mask[(top):(top) + (patch_size), (left):(left) + (patch_size)] += 1

                    batch_count = 0
                    crop = []
                    position = []

               
        image_file = result_img / voting_mask

        fold += [image_file]
    
    total[i] = np.array(fold)


total = total.mean(0)
total = np.clip(total, 0, 1)


for i in range(len(total)):
    image = total[i]
    
    cv2.imwrite('{}.png'.format(i + 20000), cv2.cvtColor((image*255).astype('uint8'), cv2.COLOR_BGR2RGB))


submission = zipfile.ZipFile("submission.zip", 'w')
for i in range(18):
    
    path = '{}.png'.format(i+20000)
    submission.write(path)
submission.close()
