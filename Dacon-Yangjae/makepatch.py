#!/usr/bin/env python
# coding: utf-8

# In[2]:


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




lr_path = np.array(sorted(glob.glob('datasets/train/lr/*')))
hr_path = np.array(sorted(glob.glob('datasets/train/hr/*')))

test_path = sorted(glob.glob('datasets/test/lr/*'))


from sklearn.model_selection import StratifiedKFold, KFold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state = 357)

temp = np.ones_like(lr_path)
for fold, (train_index, val_index) in enumerate(kf.split(lr_path, temp)):

    if fold != target_fold - 1:
        continue

    lr_path_train, lr_path_val = lr_path[train_index], lr_path[val_index]
    hr_path_train, hr_path_val = hr_path[train_index], hr_path[val_index]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


def cut_img(path):

    lr_path = path[0]
    hr_path = path[1]
    

    
    
    patch_size = 768
    stride = int(patch_size / 2)
    
    num = 0
        
    lr_img = cv2.imread(lr_path)[..., ::-1]
    lr_img = cv2.resize(lr_img, None, fx=4.0, fy=4.0)

    hr_img = cv2.imread(hr_path)[..., ::-1]

    print(lr_path)
    
    for top in range(0, lr_img.shape[0], stride):
        for left in range(0, lr_img.shape[1], stride):
            
            lr_name = lr_path.split('/')[-1].split('.')[0]
            hr_name = hr_path.split('/')[-1].split('.')[0]
            
            piece_lr = np.zeros([patch_size, patch_size, 3], np.uint8)
            temp_lr = lr_img[top : top+patch_size, left : left+patch_size, :]
            piece_lr[:temp_lr.shape[0], :temp_lr.shape[1], :] = temp_lr

            piece_hr = np.zeros([patch_size, patch_size, 3], np.uint8)
            temp_hr = hr_img[top : top+patch_size, left : left+patch_size, :]
            piece_hr[:temp_hr.shape[0], :temp_hr.shape[1], :] = temp_hr

            os.makedirs('datasets/patchdata/Fold1/lr', exist_ok=True)
            os.makedirs('datasets/patchdata/Fold1/hr', exist_ok=True)

            lr_name = lr_name + '_{}'.format(num)
            hr_name = hr_name + '_{}'.format(num)
            
            np.save('datasets/patchdata/Fold1/lr/{}.npy'.format(lr_name), lr_img)
            np.save('datasets/patchdata/Fold1/hr/{}.npy'.format(hr_name), hr_img)

            num+=1


# In[ ]:





# In[ ]:





# In[14]:


temp = []

for i in range(len(lr_path_train)):
    temp += [[lr_path_train.tolist()[i], hr_path_train.tolist()[i]]]


# In[ ]:





# In[ ]:





# In[15]:


from multiprocessing import Pool

with Pool(64) as p:
    print(p.map(cut_img, temp))


# In[ ]:





# In[ ]:




