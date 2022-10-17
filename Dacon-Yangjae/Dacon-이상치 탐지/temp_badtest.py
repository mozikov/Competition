#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import os

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

from glob import glob
import pandas as pd
import numpy as np 
from tqdm import tqdm
import cv2

import os
import timm
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from sklearn.metrics import f1_score, accuracy_score
import time

import logging

import albumentations as A

import torch.nn.functional as F

import torch_optimizer as optim


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2



def focalloss(pred, mask, gamma):

    prob = torch.softmax(pred, 1)
    mask_onehot = F.one_hot(mask, num_classes=88)

    true_prob = prob[mask_onehot.type(torch.bool)]

    celoss = -torch.log(true_prob)

    weight = 0.25 * ((1-true_prob) ** gamma)
    return (weight * celoss).mean()



import yaml
with open('config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

    
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed_everything(config['seed'])



device = torch.device('cuda')


# In[4]:


train_png = sorted(glob('../../jun09173/train/*.png'))
test_png = sorted(glob('../../jun09173/test/*.png'))


# In[5]:


train_y = pd.read_csv("train_df.csv")

train_labels = train_y["label"]

label_unique = sorted(np.unique(train_labels))
label_unique = {key:value for key,value in zip(label_unique, range(len(label_unique)))}

train_labels = [label_unique[k] for k in train_labels]






def img_load(path):
    img = cv2.imread(path)[:,:,::-1]
    img = cv2.resize(img, (256, 256))
    return img







# In[16]:

image_size = config['image_size']

all_images = np.load('train_imgs_{}.npy'.format(image_size))
all_labels = np.array(train_labels)


# In[44]:

class CustomDataset(Dataset):
    def __init__(self, image_path, label_path, mode, aug):
        self.image_path = image_path
        self.label_path = label_path
        self.mode = mode
        self.aug = aug

    def __len__(self):
        return len(self.image_path)
    
    def __getitem__(self, idx):

        # image
        image = self.image_path[idx]
        label = np.array([self.label_path[idx]])
        
        if self.mode == 'train':
            
            image = self.aug(image=image)['image']
        
            image = transforms.ToTensor()(image)
            label = torch.from_numpy(label)
            
        else:
            
            image = A.Normalize(p=1)(image=image)['image']
            
            image = transforms.ToTensor()(image)
            label = torch.from_numpy(label)
        
        
        return {
            'img' : image,
            'label' : label
        }


# In[47]:

model_name = config['model_name']

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('{}'.format(model_name), pretrained=True, num_classes=88)
#         self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=88)
        
    def forward(self, x):
        x = self.model(x)
        return x



from datetime import datetime

year = datetime.today().year
month = datetime.today().month
day = datetime.today().day  

today = str(year) + str(month) + str(day)  


  



import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--idx', default=1, type=int)
args = parser.parse_args()




from sklearn.model_selection import StratifiedKFold
n_splits = 5
seed = config['seed']
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = seed)

FOCAL = config['focal']

for fold, (train_index, test_index) in enumerate(kf.split(all_images, all_labels)): 
    
    
    if (fold+1) != args.idx:
        continue    
        
    
    compensate = config['compensate']
    UPSAMPLE = config['UPSAMPLE']
    
    run_info = '{}_base_size{}_fliprotate_schedulerCosine_shiftscale_allaug++_seed{}_UPSAMPLE{}{}_{}_LS{}_cutmix{}_focal_{}_mixup_{}_normalize'.format(today, image_size, seed, UPSAMPLE, compensate, model_name, config['LS'], config['cutmix'], FOCAL, config['mixup'])

    os.makedirs('checkpoint/{}/fold{}'.format(run_info, args.idx), exist_ok=True)

    log = logging.getLogger('staining_log')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fileHandler = logging.FileHandler('checkpoint/{}/fold{}/log.txt'.format(run_info, args.idx))
    streamHandler = logging.StreamHandler()
    fileHandler.setFormatter(formatter)
    streamHandler.setFormatter(formatter)
    #
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)
    
    log.info("Config : {}".format(config))

    
    X_train, X_test = all_images[train_index], all_images[test_index] 
    y_train, y_test = all_labels[train_index], all_labels[test_index]
    
    
    
    ##############      Only Bad Test      ################
    count = np.bincount(y_test)
    fewer_index = count<10
    fewer_label = np.where(fewer_index)[0]

    final_idx = np.isin(y_test, fewer_label)

    #X_test = X_test[final_idx]
    #y_test = y_test[final_idx]

    
    if UPSAMPLE:
        a = np.bincount(y_train)
        insufficient_class_idx = a < 15
        idx = np.where(insufficient_class_idx)[0]


        

        total_image_upsample = []
        total_label_upsample = []

        for i in range(len(idx)):

            target_idx = y_train == idx[i]

            target_image = X_train[target_idx]
            target_label = y_train[target_idx]

            num = target_image.shape[0]

            compen_num = round((compensate - num) / num)

            new_image = target_image.repeat(compen_num, 0)
            new_label = target_label.repeat(compen_num, 0)

            for j in range(new_image.shape[0]):

                total_image_upsample += [new_image[j]]
                total_label_upsample += [new_label[j]]



        a = np.bincount(y_train)
        insufficient_class_idx = a >= 15
        idx = np.where(insufficient_class_idx)[0]


        total_image_original = []
        total_label_original = []

        for i in range(len(idx)):

            target_idx = y_train == idx[i]

            target_image = X_train[target_idx]
            target_label = y_train[target_idx]

            new_image = target_image#.repeat(compen_num, 0)
            new_label = target_label#.repeat(compen_num, 0)

            for j in range(new_image.shape[0]):

                total_image_original += [new_image[j]]
                total_label_original += [new_label[j]]



        X_train = np.array(total_image_upsample + total_image_original)
        y_train = np.array(total_label_upsample + total_label_original)



    all_aug = A.Compose([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0, p=0.5),

#                         A.Sharpen(p=0.7),
#                         A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.3),
#                         A.FancyPCA(alpha=0.1, p=0.3),
#                         A.Emboss(p=0.5),

#                         A.CLAHE(clip_limit=5, p=0.5),
#                         A.Posterize(p=0.5),
                         
                         
                         
                         
                         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=360, interpolation=0, border_mode=0, p=0.5),

                         A.OneOf([
                         A.GaussianBlur(blur_limit=(1, 3), p=1),
                         A.MedianBlur(blur_limit=3, p=1),
                         A.GaussNoise (var_limit=(10.0, 50.0), p=1)
                         ], p=0.5),

                         A.VerticalFlip(p=0.5),
                         A.HorizontalFlip(p=0.5),
                         A.RandomRotate90(90, p=0.5),
                         
#                          A.CLAHE(clip_limit=2, p=0.5),
#                          A.Sharpen(p=0.5),
                         

                         A.CoarseDropout(max_holes=4, max_height=20, max_width=20, 
                                         min_holes=2, min_height=20, min_width=20, p=0.5),
                         
                         A.Normalize(p=1)
                         
                         ])

    log.info("Augmentation : {}".format(all_aug))
    
    batch_size = config['batch_size']
    
    train_dataset = CustomDataset(X_train, y_train, 'train', all_aug)
    test_dataset = CustomDataset(X_test, y_test, 'test', None)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=8, shuffle=False)
    
    device = 'cuda'
    model = Network().to(device)
    
    optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#     optimizer = optim.Lamb(model.parameters(), lr=0.001)




    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 300)

#     weight = torch.Tensor([2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1,
#        2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2,
#        2, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2,
#        2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2]).to(device)
    
    weight = torch.Tensor([3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1,
       3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3,
       3, 1, 3, 3, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3,
       3, 1, 1, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3]).to(device)
    
    criterion = nn.CrossEntropyLoss()

    if config['LS']:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



    log.info('Criterion : {}'.format(criterion))
    
    from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
    num_epochs = config['num_epochs']
    total_steps = int(len(train_dataset)*num_epochs/(batch_size))
    warmup_steps = config['warmup_steps']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)  
    

    
    
    def accuracy_function(real, pred):    
        real = real.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        score = f1_score(real, pred, average='macro')
        return score


    log.info("Training Set : {}".format(len(train_dataset)))
    log.info("Test Set : {}".format(len(test_dataset)))
    log.info('\n')


    ######################################################### TRAIN #########################################################
    import torch.cuda.amp as amp  
    scaler = amp.GradScaler()

    cutmix = config['cutmix']
    beta = 1
    best_f1 = 0
    
    for epoch in range(num_epochs):
        
#         scheduler.step()
        
        log.info("Epoch : {}".format(epoch+1))

        log.info('\n')
        log.info("Current Learning Rate : {:.8f}".format(scheduler.get_lr()[0]))
        
        train_loss_sum = []
        train_pred_label = []
        train_true_label = []
        F1_score = []

        model.train()
        for idx, batch in enumerate(train_dataloader):
            
            scheduler.step()

            image = batch['img'].to(device)
            label = batch['label'].squeeze(1).to(device)
            
            
            with amp.autocast():
            
                if cutmix and random.random() > 0.6:

                    lam = np.random.beta(beta, beta)
                    rand_index = torch.randperm(image.size()[0]).cuda()
                    bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)

                    target_a = label
                    target_b = label[rand_index]

                    image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
                    output = model(image)     
                    loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
                    
                    if FOCAL:
                        loss = focalloss(output, target_a, 5) * lam + focalloss(output, target_b, 5) * (1. - lam)

                
                elif config['mixup'] and random.random() > 0.5:
                    x, y, y_2, lam = mixup_data(image, label)
                    output = model(x)
                    loss = mixup_criterion(criterion, output, y, y_2, lam)

                else:
                    output = model(image)

                    loss = criterion(output, label)
                    if FOCAL:
                        loss = focalloss(output, label, 5)
                
        #         loss = focalloss(output, label, 2)
    #             loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)


    
    
            f1 = accuracy_function(label, output)
    #         loss = torch.tensor([1 - f1], requires_grad=True).to(device)


            optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

            scaler.scale(loss).backward()
            scaler.step(optimizer)    
            scaler.update() 

    #         train_loss_sum += [loss.detach().cpu().tolist()]
            train_true_label += label.cpu().numpy().tolist()
            train_pred_label += torch.argmax(torch.softmax(output, 1), 1).tolist()
            F1_score += [f1]

            true_number = np.array(train_true_label) == np.array(train_pred_label)
    #         train_loss = sum(train_loss_sum) / len(train_loss_sum)
            Accuracy = sum(true_number) / len(true_number)
            F1_Score = sum(F1_score) / len(F1_score)

    
        F1_Score = f1_score(np.array(train_true_label), np.array(train_pred_label), average='macro')



        log.info("Train Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, F1 : {:.4f}".format(epoch+1, num_epochs, idx+1, len(train_dataloader), Accuracy, F1_Score))
        log.info("\n")


    #     torch.save(model.state_dict(), 'checkpoint/{}/Epoch {}.pth'.format(run_info, epoch+1))            



        test_loss_sum = []
        test_pred_label = []
        test_true_label = []
        F1_score = []
        
        model.eval()  
        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):

                image = batch['img'].to(device)
                label = batch['label'].squeeze(1).to(device)
                output = model(image)

                loss = criterion(output, label)
                f1 = accuracy_function(label, output)


                test_loss_sum += [loss.detach().cpu().tolist()]
                test_true_label += label.cpu().numpy().tolist()
                test_pred_label += torch.argmax(torch.softmax(output, 1), 1).tolist()

                true_number = np.array(test_true_label) == np.array(test_pred_label)
                test_loss = sum(test_loss_sum) / len(test_loss_sum)
                Accuracy = sum(true_number) / len(true_number)

                F1_score += [f1]
    #         F1_Score = sum(F1_score) / len(F1_score)
            F1_Score = f1_score(np.array(test_true_label), np.array(test_pred_label), average='macro')


            log.info("TEST Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, F1 : {:.4f}, Loss : {:.3f}".format(epoch+1, num_epochs, i+1, len(test_dataloader), Accuracy, F1_Score, test_loss))
            log.info("\n")

            os.makedirs('checkpoint/{}/fold{}'.format(run_info, fold+1), exist_ok=True)

            if F1_Score > best_f1:
                best_f1 = F1_Score
                torch.save(model.state_dict(), 'checkpoint/{}/fold{}/Epoch {} ACC {:.3f} F1 : {:.4f} TEST Loss{:.3f}.pth'.format(run_info, fold+1, epoch+1, Accuracy, F1_Score, test_loss))

    #         if F1_Score > 0.8:
    #             torch.save(model.state_dict(), 'checkpoint/{}/fold{}/Epoch {} ACC {:.3f} F1 : {:.4f} TEST Loss{:.3f}.pth'.format(run_info, fold+1, epoch+1, Accuracy, F1_Score, test_loss))




