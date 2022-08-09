#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import glob
import json

from skimage import draw

import matplotlib.pyplot as plt

import os
from cv2 import cv2

from PIL import Image

import random

import tifffile as tiff

import pandas as pd

import cv2
import numpy as np
# import openslide
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from tifffile import memmap

import albumentations as A

from skimage import morphology as morph

import skimage
from skimage.color import rgb2gray, rgb2hed, hed2rgb, rgb2hsv, hsv2rgb
from skimage.exposure import rescale_intensity
from skimage.util import dtype

import torch
import torch.nn.functional as F

import torchvision
from torchvision import transforms

import logging

import warnings
warnings.filterwarnings("ignore")

from SwinUnet.networks.vision_transformer import SwinUnet as ViT_seg

from scipy.ndimage import measurements
from skimage.morphology import remove_small_objects

import segmentation_models_pytorch as smp

import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm

from metrics.stats_utils import get_pq, get_multi_pq_info, get_multi_r2

from scipy.ndimage import filters, measurements
from scipy.ndimage.morphology import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_cdt,
    distance_transform_edt,
)

from skimage.segmentation import watershed

import sys
# sys.path.append('segmenter/')

# from segmenter.segm.model.factory import create_segmenter


from my_misc import *
from my_misc import __proc_np_hv


import pickle


def process_segmentation(output_binary, output_hv, output_classification):

    np_map = torch.softmax(output_binary, 1)[0].cpu().detach().permute(1, 2, 0)[..., 1].unsqueeze(2).numpy()
    np_map = cv2.resize(np_map.astype('float32'), (0, 0), fx=2.0, fy=2.0)

    hv_map = output_hv[0].cpu().detach().permute(1, 2, 0).numpy().astype('float32')
    hv_map = cv2.resize(hv_map, (0, 0), fx=2.0, fy=2.0) 

    tp_map = torch.argmax(torch.softmax(output_classification, 1), 1).cpu().numpy()[0]
    tp_map = cv2.resize(tp_map, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST).astype('float32')

    inst_map = __proc_np_hv(np_map[..., None], hv_map)
    inst_dict = get_instance_info(inst_map, tp_map)

    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([
        [v['type']] * 3 for v in inst_dict.values()
    ])
    type_map = overlay_prediction_contours(
        type_map, inst_dict,
        line_thickness=-1,
        inst_colours=inst_type_colours)

    pred_map = np.dstack([inst_map, type_map])

    pred_map = cv2.resize(pred_map, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    
    return pred_map





def process_segmentation_2branch(output_binary, output_classification):

    tp_map = torch.argmax(torch.softmax(output_classification, 1), 1).cpu().numpy()[0]

    temp = torch.argmax(torch.softmax(output_binary, 1), 1).cpu()[0]
    inst_map = measurements.label(temp)[0]
    inst_map = remove_small_objects(inst_map, 10)
    
    inst_dict = get_instance_info(inst_map, tp_map)

    type_map = np.zeros_like(inst_map)
    inst_type_colours = np.array([
        [v['type']] * 3 for v in inst_dict.values()
    ])
    type_map = overlay_prediction_contours(
        type_map, inst_dict,
        line_thickness=-1,
        inst_colours=inst_type_colours)

    pred_map = np.dstack([inst_map, type_map])

    
    return pred_map



class ConicDataset(Dataset):

    def __init__(self, image_path, mask_path, counts, source, transform, copypaste):

        self.image_path = image_path
        self.mask_path = mask_path
        self.counts = counts
        self.source = source

        self.transform = transform
        self.copypaste = copypaste

    def __len__(self):
        return (len(self.image_path))

    def __getitem__(self, idx):

        image = self.image_path[idx].astype('uint8')
        original = image.copy()

        mask = self.mask_path[idx]
        
        mask_instance = mask[..., 0]
        mask_class = mask[..., 1]
        
        counts = self.counts[idx]
        source = self.source[idx]
        
        class1_copy_num = 18
        class4_copy_num = 3
        class5_copy_num = 24
        
        if self.copypaste:
            copypaste_result = copypaste_random_instance(image, mask_instance, mask_class, 
                                                         class1_copy_num, class4_copy_num, class5_copy_num)
            
            image = copypaste_result['pasted_image']
            mask_instance = copypaste_result['pasted_mask_instance']
            mask_class = copypaste_result['pasted_mask_class']
                                      
                        
        if self.transform:
            

            image = A.OneOf([
                             A.GaussianBlur(blur_limit=(1, 3), p=1),
                             A.MedianBlur(blur_limit=3, p=1),
                             A.GaussNoise (var_limit=(10.0, 50.0), p=1)
                             ], p=0.5)(image=image)['image']
            
            image = A.ColorJitter(brightness=0.3, contrast=0.6, saturation=0.2, hue=0.2, p=0.5)(image=image)['image']
            
            output = A.Compose([
                #A.ShiftScaleRotate(p=1),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Cutout(num_holes=4, max_h_size=40, max_w_size=40, p=0.5),
                A.RandomRotate90(90, p=0.5)],
                additional_targets={'mask1' : 'image',
                                    'mask2' : 'image'})(image=image.astype('uint8'), 
                                                        mask1=mask_class.astype('uint8'),
                                                        mask2=mask_instance.astype('uint8'))            
            
            image = output['image']
            
            
            mask = output['mask1']
            mask_instance = output['mask2'].astype('uint8')
            
            hv_map = gen_targets(mask_instance, (256, 256))['hv_map']


            image = transforms.ToTensor()(image)
            #image = cropping_center_torch(image, crop_shape=(224, 224), batch=False)
            
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask_instance = torch.from_numpy(mask_instance).unsqueeze(0)
            #mask = cropping_center_torch(mask, crop_shape=(224, 224), batch=False)
            
            hv_map = transforms.ToTensor()(hv_map)
            #hv_map = cropping_center_torch(hv_map, crop_shape=(224, 224), batch=False)
        
        
    
            return {"original" : original, "image": image, "mask": mask, "mask_instance" : mask_instance, 'hv_map' : hv_map, "counts" : counts, "source" : source}

        

        else:

            image_hflip = A.HorizontalFlip(p=1)(image=image)['image']
            image_hflip = transforms.ToTensor()(image_hflip)
            #image_hflip = cropping_center_torch(image_hflip, crop_shape=(224, 224), batch=False)
            
            image_vflip = A.VerticalFlip(p=1)(image=image)['image']
            image_vflip = transforms.ToTensor()(image_vflip)
            #image_vflip = cropping_center_torch(image_vflip, crop_shape=(224, 224), batch=False)
            
            image = transforms.ToTensor()(image)
            #image = cropping_center_torch(image, crop_shape=(224, 224), batch=False)
            
            mask = torch.from_numpy(mask_class).unsqueeze(0)
            #mask = cropping_center_torch(mask, crop_shape=(224, 224), batch=False)
            
            hv_map = gen_targets(mask_instance, (256, 256))['hv_map']
            hv_map = transforms.ToTensor()(hv_map)
            #hv_map = cropping_center_torch(hv_map, crop_shape=(224, 224), batch=False)
            
            return {"image": image, "mask": mask, 'hv_map' : hv_map, "image_hflip" : image_hflip, "image_vflip" : image_vflip, "counts" : counts, 'mask_instance' : mask_instance, "source" : source}  
        
        
        
        
        
        

from datetime import datetime

year = datetime.today().year
month = datetime.today().month
day = datetime.today().day  

today = str(year) + str(month) + str(day)        
        
        
        
        
        
gamma = 1

run_info = '-'.format(today, gamma)

if not os.path.exists('checkpoint/{}'.format(run_info)):
    os.mkdir('checkpoint/{}'.format(run_info))

log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoint/{}/log.txt'.format(run_info))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)




data_root = "patchdata/patchsize256stride256" #! Change this according to the root path where the data is located
#data_root = "data"

images_path = "%s/images.npy" % data_root # images array Nx256x256x3
labels_path = "%s/labels.npy" % data_root # labels array Nx256x256x3
hv_path = "%s/hv_maps.npy" % data_root # labels array Nx256x256x3
counts_path = "%s/counts.csv" % data_root # csv of counts per nuclear type for each patch
info_path = "%s/patch_info.csv" % data_root # csv indicating which image from Lizard each patch comes from



images = np.load(images_path)
labels = np.load(labels_path)
hv_maps = np.load(hv_path)
counts = pd.read_csv(counts_path)
patch_info = pd.read_csv(info_path)

info = pd.read_csv(f'{data_root}/patch_info.csv')
file_names = np.squeeze(info.to_numpy()).tolist()




img_sources = [v.split('-')[0] for v in file_names]
img_sources = np.unique(img_sources)

cohort_sources = [v.split('_')[0] for v in img_sources]
_, cohort_sources = np.unique(cohort_sources, return_inverse=True)

temp = np.array([file_name.split('-')[0] for file_name in file_names])
temp2 = np.array([i.split('_')[0] for i in temp])


# source 별

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 357) 

for fold, (train_index, test_index) in enumerate(kf.split(img_sources, cohort_sources)): 
    
    log.info("Fold {}".format(fold+1))
    log.info("\n")
    
    X_train, X_test = img_sources[train_index], img_sources[test_index] 
    y_train, y_test = cohort_sources[train_index], cohort_sources[test_index]
    
    train_idx = np.where(np.isin(temp, X_train))[0]
    test_idx = np.where(np.isin(temp, X_test))[0]
    
    train_images = images[train_idx]
    train_masks = labels[train_idx]
    train_counts = counts.iloc[train_idx].values
    train_info = info.iloc[train_idx].values
    train_source = [i.split('_')[0] for i in train_info[..., 0]]
    
    
    test_images = images[test_idx]
    test_masks = labels[test_idx]
    test_counts = counts.iloc[test_idx].values
    test_info = info.iloc[test_idx].values
    test_source = [i.split('_')[0] for i in test_info[..., 0]]    
    

    source_name = np.unique(np.array(train_source))

    for source in source_name:

        log.info(source)
        
        insufficient_class_alldict = {"class1" : {}, "class2" : {}, "class3" : {}, "class4" : {}, "class5" : {}, "class6" : {}}
        insufficient_class = [1, 2, 3, 4, 5, 6]


        total_num_class1 = 1
        total_num_class2 = 1
        total_num_class3 = 1
        total_num_class4 = 1
        total_num_class5 = 1
        total_num_class6 = 1


        source_idx = np.where(np.array(train_source) == source)[0]

        target_image = train_images[source_idx]
        target_mask_instance = train_masks[source_idx]
        target_mask_class = train_masks[source_idx]

        for i in range(target_image.shape[0]):

            image = target_image[i].copy()
            mask_instance = target_mask_instance[i][..., 0].copy()
            mask_class = target_mask_class[i][..., 1].copy()


            for class_ in insufficient_class:

        #         print("Class : {}".format(class_))
        #         print('\n')

                class_mask = (mask_class == class_)

                class_instance = mask_instance * class_mask

                for idx, num in enumerate(np.unique(class_instance)[1:]):

                    instance_mask = mask_instance == num
                    instance = mask_instance * instance_mask
                    instance_bool = instance.astype('bool')

                    instance_image = image * instance_bool[..., None].repeat(3, 2)
                    instance_mask_instance = mask_instance * instance_bool
                    instance_mask_class = mask_class * instance_bool


                    x_min = np.where(np.any(instance_mask_instance, 0))[0][0]
                    x_max = np.where(np.any(instance_mask_instance, 0))[0][-1]

                    y_min = np.where(np.any(instance_mask_instance, 1))[0][0]
                    y_max = np.where(np.any(instance_mask_instance, 1))[0][-1]        

                    width = (x_max - x_min)
                    height = (y_max - y_min)

                    if (width == 0) | (height == 0):
                        continue

                    image_final = instance_image[y_min : y_max, x_min : x_max]
                    instance_final = instance_mask_instance[y_min : y_max, x_min : x_max]
                    class_final = instance_mask_class[y_min : y_max, x_min : x_max]

                    total_final = np.zeros(shape=(image_final.shape[0], image_final.shape[1], 5))
                    total_final[..., [0, 1, 2]] = image_final
                    total_final[..., 3] = instance_final
                    total_final[..., 4] = class_final

                    #insufficient_class_alldict.update({"class{}".format(class_) : instance_image[y_min : y_max, x_min : x_max]})

                    if class_ == 1:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class1) : total_final})
                        total_num_class1 += 1

                    if class_ == 2:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class2) : total_final})
                        total_num_class2 += 1                

                    if class_ == 3:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class3) : total_final})
                        total_num_class3 += 1                

                    if class_ == 4:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class4) : total_final})
                        total_num_class4 += 1                

                    if class_ == 5:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class5) : total_final})
                        total_num_class5 += 1                

                    if class_ == 6:
                        insufficient_class_alldict['class{}'.format(class_)].update({"{}".format(total_num_class6) : total_final})
                        total_num_class6 += 1            



        with open('insufficient_class_alldict/insufficient_class_alldict_{}_Fold5_{}.pickle'.format(source, fold+1), 'wb') as temp_:
            pickle.dump(insufficient_class_alldict, temp_)
