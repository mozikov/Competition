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



import paddleseg
from paddleseg.cvlibs import manager, Config

import argparse



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



parser = argparse.ArgumentParser()
parser.add_argument('--fold', type=int, default=1)
# parser.add_argument('--copy_paste', type=str, default=True)
# parser.add_argument('--copy_level', type=str, default='random')
# parser.add_argument('--nooverlap', type=str, default=True)
args = parser.parse_args()


class ConicDataset(Dataset):

    def __init__(self, image_path, mask_path, counts, source, transform, copypaste, copy_level):

        self.image_path = image_path
        self.mask_path = mask_path
        self.counts = counts
        self.source = source

        self.transform = transform
        self.copypaste = copypaste
        self.copy_level = copy_level
        
        
        with open('insufficient_class_alldict/insufficient_class_alldict_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict = pickle.load(temp)  
        
        with open('insufficient_class_alldict/insufficient_class_alldict_consep_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict_consep = pickle.load(temp)      
            
        with open('insufficient_class_alldict/insufficient_class_alldict_crag_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict_crag = pickle.load(temp) 
            
        with open('insufficient_class_alldict/insufficient_class_alldict_dpath_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict_dpath = pickle.load(temp) 
            
        with open('insufficient_class_alldict/insufficient_class_alldict_glas_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict_glas = pickle.load(temp)       
            
        with open('insufficient_class_alldict/insufficient_class_alldict_pannuke_Fold5_{}.pickle'.format(args.fold),'rb') as temp:
            self.insufficient_class_alldict_pannuke = pickle.load(temp)             
            
   

    def __len__(self):
        return (len(self.image_path))

    def __getitem__(self, idx):

        image = self.image_path[idx].astype('uint8')
        original = image.copy()

        mask = self.mask_path[idx]
        
        mask_instance = mask[..., 0]
        mask_class = mask[..., 1]
        mask_class_original = mask_class.copy()
        
        counts = self.counts[idx]
        source = self.source[idx]
        
#         default   
        class1_rate = 0.4
        class2_rate = 0.05
        class3_rate = 0.05
        class4_rate = 0.05
        class5_rate = 0.4
        class6_rate = 0.05

        class1_copy_num = np.random.randint(50) 
        class2_copy_num = np.random.randint(1) 
        class3_copy_num = np.random.randint(2) 
        class4_copy_num = np.random.randint(9) 
        class5_copy_num = np.random.randint(50) 
        class6_copy_num = np.random.randint(2)
        
#         class1_rate = 0.4
#         class2_rate = 0
#         class3_rate = 0
#         class4_rate = 0.2
#         class5_rate = 0.4
#         class6_rate = 0
        
        resize_rate = 0
        
        self.nooverlap = True
        
        
        if self.copypaste:
            
            if self.copy_level == 'random':
                dict_ = self.insufficient_class_alldict
                
                if random.random() > 0.5:
                    
                    print('Copy Paste !')
                    
                    copypaste_result = copypaste_candidate_instance(image, mask_instance, mask_class, 
                                                                 class1_rate, class2_rate, class3_rate, 
                                                                 class4_rate, class5_rate, class6_rate, dict_, 
                                                                 resize_rate, self.nooverlap)
                    
#                     copypaste_result = copypaste_random_instance(image, mask_instance, mask_class, 
#                                                                  class1_copy_num, class2_copy_num, class3_copy_num, 
#                                                                  class4_copy_num, class5_copy_num, class6_copy_num, dict_, 
#                                                                  resize_rate, self.nooverlap)

#                     copypaste_result = copypaste_candidate_instance_fixedboxes(image, mask_instance, mask_class, 
#                                                                  class1_rate, class2_rate, class3_rate, 
#                                                                  class4_rate, class5_rate, class6_rate, dict_, 
#                                                                  resize_rate, self.nooverlap)                  
    
    
                    image = copypaste_result['pasted_image']
                    mask_instance = copypaste_result['pasted_mask_instance']
                    mask_class = copypaste_result['pasted_mask_class']
              
            
            elif self.copy_level == 'source':
                
                if source == 'consep':
                    dict_ = self.insufficient_class_alldict_consep
                    
                if source == 'crag':
                    dict_ = self.insufficient_class_alldict_crag            
                    
                if source == 'dpath':
                    dict_ = self.insufficient_class_alldict_dpath          
                    
                if source == 'glas':
                    dict_ = self.insufficient_class_alldict_glas
                    
                if source == 'pannuke':
                    dict_ = self.insufficient_class_alldict_pannuke                    
                
                if random.random() > 0.5:
                    
                    print('Copy Paste !')
                    
                    copypaste_result = copypaste_candidate_instance(image, mask_instance, mask_class, 
                                                                 class1_rate, class2_rate, class3_rate, 
                                                                 class4_rate, class5_rate, class6_rate, dict_, 
                                                                 resize_rate, self.nooverlap)

#                     copypaste_result = copypaste_random_instance(image, mask_instance, mask_class, 
#                                                                  class1_copy_num, class2_copy_num, class3_copy_num, 
#                                                                  class4_copy_num, class5_copy_num, class6_copy_num, dict_, 
#                                                                  resize_rate, self.nooverlap)                    
                    
#                     copypaste_result = copypaste_candidate_instance_fixedboxes(image, mask_instance, mask_class, 
#                                                                  class1_rate, class2_rate, class3_rate, 
#                                                                  class4_rate, class5_rate, class6_rate, dict_, 
#                                                                  resize_rate, self.nooverlap)
    
                    image = copypaste_result['pasted_image']
                    mask_instance = copypaste_result['pasted_mask_instance']
                    mask_class = copypaste_result['pasted_mask_class']
                    
            else:
                pass
                    
                    
                    
                    
                                      
                        
        if self.transform:
            
            image = A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2, p=0.5)(image=image)['image']

            image = A.OneOf([
                             A.GaussianBlur(blur_limit=(1, 3), p=1),
                             A.MedianBlur(blur_limit=3, p=1)
#                              A.GaussNoise (var_limit=(10.0, 50.0), p=1)
                             ], p=0.5)(image=image)['image']
            
            
            output = A.Compose([
#                 A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=360, interpolation=0, border_mode=4, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
#                 A.CoarseDropout(max_holes=4, max_height=40, max_width=40, 
#                                 min_holes=2, min_height=20, min_width=20, p=0.5)],
                A.RandomRotate90(90, p=0.5)],
                additional_targets={'mask1' : 'image',
                                    'mask2' : 'image'})(image=image.astype('uint8'), 
                                                        mask1=mask_class.astype('uint8'),
                                                        mask2=mask_instance.astype('uint8'))            
            
            
            
            image = output['image']
            mask = output['mask1']
            mask_instance = output['mask2']            
            
            mask_instance = remap_label(mask_instance)
            hv_map = gen_targets(mask_instance, (256, 256))['hv_map']


            image = transforms.ToTensor()(image)
            #image = cropping_center_torch(image, crop_shape=(224, 224), batch=False)
            
            mask = torch.from_numpy(mask).unsqueeze(0)
            mask_instance = torch.from_numpy(mask_instance).unsqueeze(0)
            #mask = cropping_center_torch(mask, crop_shape=(224, 224), batch=False)
            
            hv_map = transforms.ToTensor()(hv_map)
            #hv_map = cropping_center_torch(hv_map, crop_shape=(224, 224), batch=False)
        
        
    
            return {"original" : original, "image": image, "mask": mask, "mask_instance" : mask_instance, 'hv_map' : hv_map, "counts" : counts, "source" : source, "mask_class_original" : mask_class_original}

        

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
            
            return {"original" : original, "image": image, "mask": mask, 'hv_map' : hv_map, "image_hflip" : image_hflip, "image_vflip" : image_vflip, "counts" : counts, 'mask_instance' : mask_instance, "source" : source, "mask_class_original" : mask_class_original}  
        
        
        
        

from datetime import datetime

year = datetime.today().year
month = datetime.today().month
day = datetime.today().day  

today = str(year) + str(month) + str(day)        



gamma = 2
copy_level = 'random'
copy_paste = True


run_info = '{}_Unet++_Likebaseline_copypaste_candidate_allclass_focalloss{}_efficientnet-b7_5FOLD_Copypaste{}_{}_BaseAug_0.4ratio'.format(today, gamma, copy_paste, copy_level)


os.makedirs('checkpoint/{}'.format(run_info), exist_ok=True) 
os.makedirs('checkpoint/{}/Fold{}'.format(run_info, args.fold), exist_ok=True)   
    
log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler = logging.FileHandler('checkpoint/{}/Fold{}/log.txt'.format(run_info, args.fold))
streamHandler = logging.StreamHandler()
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)
#
log.addHandler(fileHandler)
log.addHandler(streamHandler)







log.info("Fold : {}".format(args.fold))
log.info("Copy Paste: {}".format(copy_paste))
log.info("Copy Level: {}".format(copy_level))
log.info("Loss : {}".format("Focal 2"))







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

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 357)


for fold, (train_index, test_index) in enumerate(kf.split(img_sources, cohort_sources)): 
    
    if fold != (args.fold - 1):
        continue
    
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
    

    train_dataset = ConicDataset(train_images, train_masks, train_counts, train_source, transform=True, copypaste=copy_paste, copy_level = copy_level)
    train_dataloader = DataLoader(train_dataset, batch_size=8, num_workers=12, shuffle=True)

    test_dataset = ConicDataset(test_images, test_masks, test_counts, test_source, transform=False, copypaste=False, copy_level = None)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=12, shuffle=False)


    device='cuda'


#     from SwinUnet.networks.vision_transformer import SwinUnet_output3_base_256 as ViT_seg
#     from SwinUnet.networks.vision_transformer import SwinUnet_output3_large_256 as ViT_seg
#     model = ViT_seg(img_size = 256, num_classes_output1 = 7, num_classes_output2 = 2, num_classes_output3 = 2).to(device)
#     model.load_from('SwinUnet/swin_tiny_patch4_window7_224.pth')


    import sys
    sys.path.append('segmentation_models.pytorch')
    import segmentation_models_pytorch
    model = segmentation_models_pytorch.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b7",       
                                                                encoder_weights="imagenet",    
                                                                in_channels = 3,                 
                                                                classes = 7,
                                                                classes2 = 2,
                                                                classes3 = 2,
                                                                num_decoder = 3).to(device)
#     from net_desc import HoVerNetConic
#     model = HoVerNetConic(num_types=7, freeze=False, pretrained_backbone='hovernet-conic.pth').to(device)
    
    
#     model.load_state_dict(torch.load('checkpoint/2022219_Unet++_Likebaseline_copypaste_candidate_random_allclass_focalloss2_efficientnet-b7_5FOLD/Fold{}/Fold {} Epoch100.pth'.format(my_fold+1, my_fold+1)))
    
    # model = nn.DataParallel(model, device_ids = [0, 1, 2, 3])
    # model.to(device)


    # model.load_state_dict(torch.load('checkpoint/202215_Unet++_Likebaseline_NOcopypaste_focalloss2_base/Epoch 110 Loss : 0.8731, Dice : 0.7679, PQ : 0.6454, DQ : 0.6367, SQ :0.8257, Multi_PQ+ : 0.5305.pth'))

    # model.load_state_dict(torch.load('checkpoint/202216_Unet++_Likebaseline_copypaste_candidate_random_allclass_focalloss2_scale0/Epoch 376 Loss : 0.8816, Dice : 0.7686, PQ : 0.6448, DQ : 0.6391, SQ :0.8276, Multi_PQ+ : 0.5331.pth'))




    # sys.path.append('ocr-pytorch')
    # from model_ocr import OCR
    # from vovnet import vovnet39, vovnet57
    # model = OCR(7, 2, 2, vovnet57(pretrained=True)).to(device)


    # sys.path.append('semantic-segmentation')
    # from semseg.models import *
    # model = eval('SegFormer')(
    #     backbone='MiT-B5',
    #     num_classes=7,
    #     num_classes2=2,
    #     num_classes3=2
    # )
    # model.backbone.load_state_dict(torch.load('mit_b5.pth', map_location=device), strict=False)
    # model.to(device)



    log.info("Model Parameters : {:.2f}M".format(sum([p.numel() for p in model.parameters()]) / 1000000))
    log.info("\n")





    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 40)



    ######### SWA #########
    # from torch.optim.swa_utils import AveragedModel, SWALR
    # model = AveragedModel(model)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    # scheduler = SWALR(optimizer, swa_lr = 1e-5, anneal_epochs=200)



    criterion_CE = nn.CrossEntropyLoss().to(device)
    criterion_BCE = nn.BCEWithLogitsLoss().to(device)
    criterion_MSE = nn.MSELoss().to(device)


    focalloss = focalloss().to(device)
    dice_loss = dice_loss().to(device)
    msge_loss = msge_loss().to(device)



    import torch.cuda.amp as amp  
    scaler = amp.GradScaler()





    target = ['train', 'test']

    for target_ in target:

        if target_ == 'train':

            imgs = train_images
            labs = train_masks
        else:
            imgs = test_images
            labs = test_masks

        total = []

        for i in range(imgs.shape[0]):
            image = imgs[i]
            label = labs[i]

            nuclei_counts_perclass = []
            # get the counts per class
            for nuc_val in range(1, 7):
                patch_class_crop_tmp = label[..., 1] == nuc_val
                patch_inst_crop_tmp = label[..., 0] * patch_class_crop_tmp
                nr_nuclei = len(np.unique(patch_inst_crop_tmp).tolist()[1:])
                nuclei_counts_perclass.append(nr_nuclei)

            total.append(nuclei_counts_perclass)



        log.info("{} 1 neutrophil : {}".format(target_, np.array(total)[:, 0].sum()))
        log.info("{} 2 epithelial : {}".format(target_, np.array(total)[:, 1].sum()))
        log.info("{} 3 lymphocyte : {}".format(target_, np.array(total)[:, 2].sum()))
        log.info("{} 4 plasma : {}".format(target_, np.array(total)[:, 3].sum()))
        log.info("{} 5 eosinophil : {}".format(target_, np.array(total)[:, 4].sum()))
        log.info("{} 6 connective : {}".format(target_, np.array(total)[:, 5].sum()))
        log.info('\n')





    weight = torch.FloatTensor([1, 1.2, 1, 1, 1.2, 1.2, 1]).to(device)



    def focalloss(pred, mask, gamma):

        prob = torch.softmax(pred, 1)
        mask_onehot = F.one_hot(mask, num_classes=7)

        true_prob = prob[mask_onehot[:, 0, ...].permute(0, 3, 1, 2).type(torch.bool)]

        celoss = -torch.log(true_prob)

        weight = (1-true_prob) ** gamma
        return (weight * celoss).mean()











    log.info("Train : {}".format(len(train_dataset)))
    log.info("Test : {}".format(len(test_dataset)))
    log.info('\n')




    num_epochs = 100
    cutmix = False
    TTA = False
    copypaste = False
    mosaic = False
    multiscale = False

    beta = 1

    pred_final = np.zeros(shape=(len(test_dataset), images.shape[1], images.shape[2], 2))
    gt_final = np.zeros(shape=(len(test_dataset), images.shape[1], images.shape[2], 2))


    for epoch in range(1, num_epochs):

        loss_total = []
        dice_total = []


        loss_binary_ce = []
        loss_binary_dice = []
        loss_ce_ce = []
        loss_ce_dice = []
        loss_hv_mse = []
        loss_hv_msge = []


        scheduler.step()
        log.info("Current Learning Rate : {}".format(scheduler.get_lr()[0]))

        

        model.train()
        for iteration, batch in enumerate(train_dataloader):  

            image = batch['image'].to(device)
            mask = batch['mask'].type(torch.LongTensor).to(device)
            mask_instance = batch['mask_instance'].type(torch.LongTensor).to(device)
            hv_map = batch['hv_map'].to(device)


            image_copy = image.clone()
            mask_class_copy = mask.clone()
            mask_instance_copy = mask_instance.clone()   
            hv_map_copy = hv_map.clone()




            if cutmix and random.random() > 0.5:

                #mask = mask.unsqueeze(1)
                lam = np.random.beta(beta, beta)
                rand_index = torch.randperm(image.size()[0]).cuda()
                bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
                image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask[:, :, bbx1:bbx2, bby1:bby2] = mask[rand_index, :, bbx1:bbx2, bby1:bby2]
                mask_instance[:, :, bbx1:bbx2, bby1:bby2] = mask_instance[rand_index, :, bbx1:bbx2, bby1:bby2]
                hv_map[:, :, bbx1:bbx2, bby1:bby2] = hv_map[rand_index, :, bbx1:bbx2, bby1:bby2]
                #mask = mask.squeeze(1)



            if multiscale:

                if random.random() > 0.5:
                    scale_size_list = np.array([128 * 3, 128 * 4, 128 * 5])
                    scale_size = np.random.choice(scale_size_list)

                    image = F.interpolate(image, size=(scale_size, scale_size)).to(device)
                    mask = F.interpolate(mask.type(torch.FloatTensor), size=(scale_size, scale_size)).type(torch.LongTensor).to(device)
                    mask_instance = F.interpolate(mask_instance.type(torch.FloatTensor), size=(scale_size, scale_size)).type(torch.LongTensor).to(device)
                    hv_map = F.interpolate(hv_map.type(torch.FloatTensor), size=(scale_size, scale_size)).to(device)            





            if mosaic and random.random() > 0.5:

                if image.shape[0] != 8:
                    continue

                for m in range(int(image.shape[0])):

                    idx = np.arange(image.shape[0])
                    np.random.shuffle(idx)
                    random_idx = idx[:4]  

                    selected_image = image_copy[random_idx]
                    selected_mask_class = mask_class_copy[random_idx]
                    selected_mask_instance = mask_instance_copy[random_idx]
                    selected_hv_map = hv_map_copy[random_idx]

                    output_img, output_mask_class, output_mask_instance, output_hv_map = mosaic_aug(selected_image, selected_mask_class, selected_mask_instance, selected_hv_map)

                    output_img = output_img.to(device)
                    output_mask = output_mask_class.to(device)
                    output_mask_instance = output_mask_instance.to(device)
                    output_hv_map = output_hv_map.to(device)

                    image[m] = output_img[0]
                    mask[m] = output_mask
                    mask_instance[m] = output_mask_instance
                    hv_map[m] = output_hv_map





            with amp.autocast():

                output_classification, output_binary, output_hv = model(image)
#                 output = model(image)
#                 output_classification, output_binary, output_hv = output['tp'], output['np'], output['hv']
        
                target_onehot = F.one_hot(mask.cpu(), num_classes=7)[:, 0, :, :, :].permute(0, 3, 1, 2).type(torch.FloatTensor).cuda()
                #target_onehot = torch.from_numpy(mask.cpu().numpy().astype('bool').astype('float'))

                #classification_celoss = criterion_CE(output_classification, mask[:, 0, :, :])

                classification_celoss = focalloss(output_classification, mask, gamma)
                classification_diceloss = dice_loss(F.one_hot(torch.argmax(torch.softmax(output_classification, 1), 1), num_classes=7).permute(0, 3, 1, 2), target_onehot)

                binary_celoss = criterion_CE(output_binary, mask[:, 0, :, :].type(torch.bool).type(torch.LongTensor).cuda())
                binary_diceloss = dice_loss(torch.argmax(torch.softmax(output_binary, 1), 1).unsqueeze(0), mask[:, 0, :, :].type(torch.bool).type(torch.FloatTensor).unsqueeze(0).cuda())

                hv_mseloss = criterion_MSE(output_hv, hv_map)

                true_np_onehot = F.one_hot(mask.cpu().type(torch.bool).type(torch.int64)[:, 0, ...], num_classes=2)[..., 1]
                hv_msgeloss =  msge_loss(hv_map.permute(0, 2, 3, 1).cpu().type(torch.FloatTensor), 
                                         output_hv.permute(0, 2, 3, 1).cpu().type(torch.FloatTensor), 
                                         true_np_onehot.cpu())[0]          


                loss = (classification_celoss + binary_celoss) + \
                       (classification_diceloss + binary_diceloss) + \
                       (hv_mseloss) + \
                       (hv_msgeloss)



                optimizer.zero_grad() 

                scaler.scale(loss).backward()
                scaler.step(optimizer)    
                scaler.update()            



            #####################################################################
            #####################        Post Process        ####################
            #####################################################################            
            probs = post_process(output_classification, 30)
            probs_onehot = F.one_hot(torch.from_numpy(probs.astype('int')), num_classes=7).permute(0, 3, 1, 2)



            loss_total += [loss.item()]
            target_onehot_fordice = F.one_hot(mask.cpu(), num_classes=7)[:, 0, :, :, :].permute(0, 3, 1, 2).type(torch.FloatTensor)
            dice_total += [dice_coef(probs_onehot, target_onehot_fordice).mean()]



            loss_binary_ce += [binary_celoss.item()]
            loss_binary_dice += [binary_diceloss.item()]
            loss_ce_ce += [classification_celoss.item()]
            loss_ce_dice += [classification_diceloss.item()]
            loss_hv_mse += [hv_mseloss.item()]
            loss_hv_msge += [hv_msgeloss.item()]   



        log.info('Train ---- Epoch : {}/{}, Batch : {}/{}, \
                 Loss : {:.4f}, \
                 Loss_binary_ce : {:.4f}, \
                 Loss_binary_dice : {:.4f}, \
                 Loss_ce_ce : {:.4f}, \
                 Loss_ce_dice : {:.4f}, \
                 Loss_hv_mse : {:.4f}, \
                 Loss_hv_msge : {:.4f}, \
                 Dice : {:.4f}'.format(epoch, num_epochs, iteration+1, len(train_dataloader),
                                                                 sum(loss_total) / len(loss_total),
                                                                 sum(loss_binary_ce) / len(loss_binary_ce),
                                                                 sum(loss_binary_dice) / len(loss_binary_dice),
                                                                 sum(loss_ce_ce) / len(loss_ce_ce),
                                                                 sum(loss_ce_dice) / len(loss_ce_dice),
                                                                 sum(loss_hv_mse) / len(loss_hv_mse),    
                                                                 sum(loss_hv_msge) / len(loss_hv_msge),    
                                                                 sum(dice_total) / len(dice_total)))

        log.info('\n')



        if (epoch % 1 == 0) & (epoch > 35):

            loss_total = []
            dice_total = [] 

            loss_binary_ce = []
            loss_binary_dice = []
            loss_ce_ce = []
            loss_ce_dice = []
            loss_hv_mse = []
            loss_hv_msge = []        


            model.eval()
            for iteration, batch in enumerate(test_dataloader):

                image = batch['image'].to(device)
                mask = batch['mask'].type(torch.LongTensor).to(device) 
                hv_map = batch['hv_map'].to(device)


                with amp.autocast():        

                    if TTA:
                        output_classification, output_binary, output_hv = model(image)

                        image_hflip = batch['image_hflip'].to(device)
                        image_vflip = batch['image_vflip'].to(device)



                        output_classification_hflip, output_binary_hflip, output_hv_hflip = model(image_hflip)
                        output_classification_hflip, output_binary_hflip, output_hv_hflip = (output_classification_hflip.cpu().detach()[0].permute(1, 2, 0).numpy()), \
                                                                                            (output_binary_hflip.cpu().detach()[0].permute(1, 2, 0).numpy()), \
                                                                                            (output_hv_hflip.cpu().detach()[0].permute(1, 2, 0).numpy())

                        output_classification_hflip, output_binary_hflip, output_hv_hflip = A.HorizontalFlip(p=1)(image=output_classification_hflip)['image'], \
                                                                                            A.HorizontalFlip(p=1)(image=output_binary_hflip)['image'], \
                                                                                            A.HorizontalFlip(p=1)(image=output_hv_hflip)['image']

                        output_classification_hflip, output_binary_hflip, output_hv_hflip = transforms.ToTensor()(output_classification_hflip).unsqueeze(0).to(device), \
                                                                                            transforms.ToTensor()(output_binary_hflip).unsqueeze(0).to(device), \
                                                                                            transforms.ToTensor()(output_hv_hflip).unsqueeze(0).to(device)


                        output_classification_vflip, output_binary_vflip, output_hv_vflip = model(image_vflip)
                        output_classification_vflip, output_binary_vflip, output_hv_vflip = (output_classification_vflip.cpu().detach()[0].permute(1, 2, 0).numpy()), \
                                                                                            (output_binary_vflip.cpu().detach()[0].permute(1, 2, 0).numpy()),\
                                                                                            (output_hv_vflip.cpu().detach()[0].permute(1, 2, 0).numpy())

                        output_classification_vflip, output_binary_vflip, output_hv_vflip = A.VerticalFlip(p=1)(image=output_classification_vflip)['image'], \
                                                                                            A.VerticalFlip(p=1)(image=output_binary_vflip)['image'], \
                                                                                            A.VerticalFlip(p=1)(image=output_hv_vflip)['image']

                        output_classification_vflip, output_binary_vflip, output_hv_vflip = transforms.ToTensor()(output_classification_vflip).unsqueeze(0).to(device), \
                                                                                            transforms.ToTensor()(output_binary_vflip).unsqueeze(0).to(device), \
                                                                                            transforms.ToTensor()(output_hv_vflip).unsqueeze(0).to(device)

                        output_classification = (output_classification + output_classification_hflip + output_classification_vflip) / 3
                        output_binary = (output_binary + output_binary_hflip + output_binary_vflip) / 3
                        output_hv = (output_hv + output_hv_hflip + output_hv_vflip) / 3

                    else:
                        output_classification, output_binary, output_hv = model(image)
#                         output = model(image)
#                         output_classification, output_binary, output_hv = output['tp'], output['np'], output['hv']



                    target_onehot = F.one_hot(mask.cpu(), num_classes=7)[:, 0, :, :, :].permute(0, 3, 1, 2).type(torch.FloatTensor).cuda()
                    #target_onehot = torch.from_numpy(mask.cpu().numpy().astype('bool').astype('float'))

                    #classification_celoss = criterion_CE(output_classification, mask[:, 0, :, :])

                    classification_celoss = focalloss(output_classification, mask, gamma)
                    classification_diceloss = dice_loss(F.one_hot(torch.argmax(torch.softmax(output_classification, 1), 1), num_classes=7).permute(0, 3, 1, 2), target_onehot)

                    binary_celoss = criterion_CE(output_binary, mask[:, 0, :, :].type(torch.bool).type(torch.LongTensor).cuda())
                    binary_diceloss = dice_loss(torch.argmax(torch.softmax(output_binary, 1), 1).unsqueeze(0), mask[:, 0, :, :].type(torch.bool).type(torch.FloatTensor).unsqueeze(0).cuda())


                    true_np_onehot = F.one_hot(mask.cpu().type(torch.bool).type(torch.int64)[:, 0, ...], num_classes=2)[..., 1]
                    hv_mseloss = criterion_MSE(output_hv, hv_map)
                    hv_msgeloss =  msge_loss(hv_map.permute(0, 2, 3, 1).cpu().type(torch.FloatTensor), 
                                             output_hv.permute(0, 2, 3, 1).cpu().type(torch.FloatTensor), 
                                             true_np_onehot.cpu())[0]          

                    loss = (classification_celoss + binary_celoss) + \
                           (classification_diceloss + binary_diceloss) + \
                           (hv_mseloss) + \
                           (hv_msgeloss)






                #####################################################################
                #####################        Post Process        ####################
                #####################################################################            
                probs = post_process(output_classification, 30)
                probs_onehot = F.one_hot(torch.from_numpy(probs.astype('int')), num_classes=7).permute(0, 3, 1, 2)



                loss_total += [loss.item()]
                target_onehot_fordice = F.one_hot(mask.cpu(), num_classes=7)[:, 0, :, :, :].permute(0, 3, 1, 2).type(torch.FloatTensor)
                dice_total += [dice_coef(probs_onehot, target_onehot_fordice).mean()]



                loss_binary_ce += [binary_celoss.item()]
                loss_binary_dice += [binary_diceloss.item()]
                loss_ce_ce += [classification_celoss.item()]
                loss_ce_dice += [classification_diceloss.item()]
                loss_hv_mse += [hv_mseloss.item()]
                loss_hv_msge += [hv_msgeloss.item()]





                ######## Just 2 Branch Post Processing ########
                #pred_map = process_segmentation_2branch(output_binary, output_classification)

                ######## Original HoverNet Post Processing ########
                pred_map = process_segmentation(output_binary, output_hv, output_classification)



                gt_instance_map = batch['mask_instance']


                pred_final[iteration, :, :, :] = pred_map

                gt_final[iteration, :, :, 0] = gt_instance_map
                gt_final[iteration, :, :, 1] = mask[0].cpu().numpy()


            result = compute_pq('seg_class', pred_final, gt_final)

            log.info('Test ---- Epoch : {}/{}, Batch : {}/{}, Loss : {:.4f}, \
                     Loss_binary_ce : {:.4f}, \
                     Loss_binary_dice : {:.4f}, \
                     Loss_ce_ce : {:.4f}, \
                     Loss_ce_dice : {:.4f}, \
                     Loss_hv_mse : {:.4f}, \
                     Loss_hv_msge : {:.4f}, \
                     Dice : {:.4f}, PQ : {:.4f}, DQ : {:.4f}, SQ :{:.4f}, Multi_PQ+ : {:.4f}'
                     .format(epoch, num_epochs, iteration+1, len(test_dataloader),
                             sum(loss_total) / len(loss_total),
                             sum(loss_binary_ce) / len(loss_binary_ce),
                             sum(loss_binary_dice) / len(loss_binary_dice),
                             sum(loss_ce_ce) / len(loss_ce_ce),
                             sum(loss_ce_dice) / len(loss_ce_dice),
                             sum(loss_hv_mse) / len(loss_hv_mse),    
                             sum(loss_hv_msge) / len(loss_hv_msge),                 
                             sum(dice_total) / len(dice_total),
                             result['pq'][0],
                             result['DQ'][0].mean(),
                             result['SQ'][0].mean(),
                             result['multi_pq+'][0].mean()))

            log.info(result)

            log.info('\n')


            
            torch.save(model.state_dict(), 'checkpoint/{}/Fold{}/Epoch {} Loss : {:.4f}, Dice : {:.4f}, PQ : {:.4f}, DQ : {:.4f}, SQ :{:.4f}, Multi_PQ+ : {:.4f}.pth'.format(run_info, fold+1, epoch, 
                                                                                               sum(loss_total) / len(loss_total),
                                                                                               sum(dice_total) / len(dice_total),
                                                                                               result['pq'][0],
                                                                                               result['DQ'][0].mean(),
                                                                                               result['SQ'][0].mean(),
                                                                                               result['multi_pq+'][0].mean()))

