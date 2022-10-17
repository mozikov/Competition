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

from util import *
# from datasets import *
# from models import *

import augments
from sklearn.model_selection import StratifiedKFold, KFold

   


parser = argparse.ArgumentParser()
parser.add_argument('--fold', default=1, type=int)
parser.add_argument('--crop_size', default=768, type=int)
args = parser.parse_args()


class CFG:
    
    def __init__(self):
    
        self.model_name = 'unet++'
        self.encoder_name = 'efficientnet-b7'
        self.batch_size = 4
        self.stride_rate = 1
        self.num_epochs = 200
        self.seed = 357
        self.loss = 'L1' # or 'L2'
        self.num_workers = 8
        self.n_splits = 5
        self.lr = 1e-4

        self.device = 'cuda'


        self.cutblur = False
        self.augs = ["blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup", "cutblur"]


        self.prob = [0, 0, 0, 0, 0, 0, 1.0]
        self.alpha = [0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7]
        self.aux_prob, self.aux_alpha = 1.0, 1.2
        self.mix_p = None

        self.lr_path = np.array(sorted(glob.glob('datasets/train/lr/*.png')))
        self.hr_path = np.array(sorted(glob.glob('datasets/train/hr/*.png')))
        self.test_path = sorted(glob.glob('datasets/test/lr/*'))

        self.augmentation = A.Compose([
    #                                A.ColorJitter(brightness=0, contrast=0.1, saturation=0.1, hue=0, p=0.5),
    #                                 A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, interpolation=0, border_mode=0, p=0.5),
                                    #A.HorizontalFlip(p=0.5),
                                    #A.VerticalFlip(p=0.5),
                                    #A.RandomRotate90(p=0.5),
                                    A.CoarseDropout(max_holes=4, max_height=200, max_width=200,
                                                    min_holes=2, min_height=100, min_width=100, p=0.5)],
    #                                 A.RandomCrop(height=crop_size, width=crop_size, p=1)],
                                    #A.ElasticTransform(p=0.5)],
                                    additional_targets={'target_image':'image'})


class Trainer():
    def __init__(self, opt, model, train_loader, test_loader, criterion, optimizer, scheduler, fold, run_info, lr_path_val, hr_path_val, logger):
        self.opt = opt
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.logger = logger
        self.fold = fold
        self.lr_path_val = lr_path_val
        self.hr_path_val = hr_path_val
        self.run_info = run_info
        self.loss_fn = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_psnr = 0

    def fit(self):
        opt = self.opt

        for epoch in range(0, opt.num_epochs):

            self.logger.info("Current Lr : {}".format(self.scheduler.get_last_lr()[0]))
            self.scheduler.step()

            self.model.train()
            for iteration, batch in enumerate(self.train_loader):

                real_a = batch['lr_image'].to(opt.device).type(torch.float32)
                real_b = batch['hr_image'].to(opt.device).type(torch.float32)

                if opt.cutblur:
                
                    real_b, real_a, mask, aug = augments.apply_augment(
                        real_b, real_a,
                        opt.augs, opt.prob, opt.alpha,
                        opt.aux_alpha, opt.aux_alpha, opt.mix_p
                    )

                # generate fake image
                #         with amp.autocast():
                fake_b = self.model(real_a)

                loss = self.loss_fn(fake_b + real_a, real_b)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

            self.summary_and_save(epoch)

    def summary_and_save(self, epoch):
        psnr = self.evaluate()

        if psnr >= self.best_psnr:
            self.best_psnr = psnr
            self.save(epoch, self.best_psnr)


    @torch.no_grad()
    def evaluate(self):
        opt = self.opt
        self.model.eval()

        psnr_total = []
        for i in range(len(self.lr_path_val)):

            lr_path = self.lr_path_val[i]
            hr_path = self.hr_path_val[i]

            img = cv2.imread(lr_path)[..., ::-1]
            gt = cv2.imread(hr_path)[..., ::-1]

            img = cv2.resize(img, (2048, 2048))

            img = (img / 255)

            patch_size = args.crop_size
            stride = int(patch_size / opt.stride_rate)
            batch_size = 1

            crop = []
            position = []
            batch_count = 0

            result_img = np.zeros(shape=(img.shape[0], img.shape[1], 3))
            voting_mask = np.zeros(shape=(img.shape[0], img.shape[1], 3))

            self.model.eval()
            for top in range(0, img.shape[0], stride):
                for left in range(0, img.shape[1], stride):
                    piece = np.ones([patch_size, patch_size, 3], np.float32)
                    temp = img[top: top + patch_size, left: left + patch_size, :]
                    piece[:temp.shape[0], :temp.shape[1], :] = temp
                    crop.append(piece)
                    position.append([top, left])
                    batch_count += 1
                    if batch_count == batch_size:
                        input_ = np.array(crop)

                        input_ = torch.from_numpy(input_.transpose(0, 3, 1, 2)).to(opt.device)

                        with torch.no_grad():
                            pred = self.model(input_)

                        pred = (pred + input_)
                        pred = pred.cpu().detach().numpy().transpose(0, 2, 3, 1).astype('float')

                        for num, (top, left) in enumerate(position):
                            piece = pred[num]

                            h, w, c = result_img[(top):(top) + (patch_size), (left):(left) + (patch_size)].shape

                            result_img[(top):(top) + (patch_size), (left):(left) + (patch_size)] += piece[:h, :w]

                            voting_mask[(top):(top) + (patch_size), (left):(left) + (patch_size)] += 1

                        batch_count = 0
                        crop = []
                        position = []

            image_file = result_img / voting_mask

            psnr = psnr_score((image_file * 255).astype('float'),
                              gt.astype('float'), 255)

            psnr_total += [psnr]


        psnr_mean = np.array(psnr_total).mean()

        return psnr_mean



    def save(self, epoch, psnr):

        path = os.path.join('checkpoint/{}/Fold{}'.format(self.run_info, self.fold))
        os.makedirs(path, exist_ok=True)

        save_path = os.path.join(path, "Epoch {} PSNR {:.4f}.pth".format(epoch+1, psnr))
        torch.save(self.model.state_dict(), save_path)



cfg = CFG()
today = get_today()
kf = KFold(n_splits=cfg.n_splits, shuffle=True, random_state = cfg.seed)
for fold, (train_index, val_index) in enumerate(kf.split(cfg.lr_path, np.ones_like(cfg.lr_path))):

    if fold != args.fold - 1:
        continue

        
    run_info = '{}_patch{}_{}_{}_{}Fold_{}Loss_baseaug_onlycutout'.format(today, args.crop_size, cfg.model_name, cfg.encoder_name, cfg.n_splits, cfg.loss)

    if not os.path.exists('checkpoint/{}/Fold{}'.format(run_info, args.fold)):
        os.makedirs('checkpoint/{}/Fold{}'.format(run_info, args.fold), exist_ok=True)

    log = logging.getLogger('')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler('checkpoint/{}/Fold{}/log.txt'.format(run_info, args.fold))
    streamHandler = logging.StreamHandler()
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)

    log.addHandler(fileHandler)
    log.addHandler(streamHandler)

    log.info(vars(cfg))
    log.info('\n')

    lr_path_train, lr_path_val = cfg.lr_path[train_index], cfg.lr_path[val_index]
    hr_path_train, hr_path_val = cfg.hr_path[train_index], cfg.hr_path[val_index]


    lr_path_train = sorted(glob.glob('datasets/patchdata_{}/Fold{}/lr/*'.format(args.crop_size, args.fold)))
    hr_path_train = sorted(glob.glob('datasets/patchdata_{}/Fold{}/hr/*'.format(args.crop_size, args.fold)))

    train_dataset = Patch_Dataset(lr_path_train, hr_path_train, 'train', cfg.augmentation)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    test_dataset = Patch_Dataset(lr_path_val, hr_path_val, 'test', cfg.augmentation)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)


    if cfg.model_name == 'SRCNN':
        model = SRCNN().to(device)

    elif cfg.model_name == 'unet':
        model = smp.unet.Unet(encoder_name="{}".format(cfg.encoder_name),
                              encoder_weights='imagenet',
                              in_channels=3,
                              classes=3).to(cfg.device)   
        
    elif cfg.model_name == 'unet++':
        model = smp.unetplusplus.UnetPlusPlus(encoder_name="{}".format(cfg.encoder_name),
                              encoder_weights='imagenet',
                              in_channels=3,
                              classes=3).to(cfg.device)

        
#     path = glob.glob('checkpoint/2022105_patch768_unet++_resnext101_32x8d_5Fold_L1Loss_baseaug_onlycutout/Fold{}/*.pth'.format(args.fold))
#     idx = np.argmax(np.array([float(i.split('/')[-1].split('PSNR')[-1].lstrip()[:7]) for i in path]))
#     best_checkpoint = path[idx]
#     model.load_state_dict(torch.load(best_checkpoint))
    

    optimizer = optim.RAdam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(cfg.num_epochs / 2), gamma=0.1)
    
    if cfg.loss == 'L1':
        criterion = nn.L1Loss().to(cfg.device)
    elif cfg.loss == 'L2':
        criterion = nn.MSELoss().to(cfg.device)




    cutmix = False
    beta = 1

    log.info("Train : {}".format(len(train_dataset)))
    log.info("Val : {}".format(len(test_dataset)))


    scaler = amp.GradScaler()
    best_psnr = 0
    

    trainer = Trainer(cfg, 
                      model, 
                      train_dataloader, 
                      test_dataloader, 
                      criterion, 
                      optimizer, 
                      scheduler, 
                      args.fold, 
                      run_info, 
                      lr_path_val, 
                      hr_path_val, 
                      log)
    trainer.fit()

