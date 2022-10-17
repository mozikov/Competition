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

from typing import Tuple, Union

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
import tqdm

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

import segmentation_models_pytorch as smp

#from tiatoolbox.utils.misc import get_bounding_box

import pickle

def cropping_center(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input array
        crop_shape: dimensions of cropped array
    
    Returns:
        x: cropped array
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        x = x[h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def cropping_center_torch(x, crop_shape, batch=False):
    """Crop an input image at the centre.

    Args:
        x: input torch tensor
        crop_shape: dimensions of cropped torch tensor
    
    Returns:
        x: cropped torch tensor
    
    """
    orig_shape = x.shape
    if not batch:
        h0 = int((orig_shape[1] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[2] - crop_shape[1]) * 0.5)
        x = x[:, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    else:
        h0 = int((orig_shape[2] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[3] - crop_shape[1]) * 0.5)
        x = x[:, :, h0 : h0 + crop_shape[0], w0 : w0 + crop_shape[1]]
    return x


def dice_coef(pred, target, smooth = 1e-5):
    
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    # dice coefficient
    dice = ((2.0 * intersection) + smooth) / (union + smooth)
    
    return dice



# def dice_loss(pred, target, smooth=1e-5):
    
#     # dice loss
#     dice = dice_coef(pred, target).mean()    
#     dice_loss = 1.0 - dice
    
#     return dice_loss


def dice_loss_w(pred, target, weight):
    
    # dice loss
    dice = dice_coef(pred, target)
    dice_loss = 1.0 - dice
    dice_loss = dice_loss * weight
    
    return dice_loss.mean()



# In[ ]:

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





def post_process(pred, min_size=30):
    
    prediction_map = torch.softmax(pred, 1)
    prediction_map = np.argmax(prediction_map.detach().cpu(), axis=1)
    
    def remove(prediction_map, class_):

        pred = (prediction_map == class_).numpy().astype(float)
        pred = measurements.label(pred)[0]
        pred = remove_small_objects(pred, min_size=min_size)
        pred = np.where((pred != 0) == True, class_, 0)

        return pred
    

    pred_final = np.zeros(shape=(pred.shape[0], pred.shape[2], pred.shape[3])).astype('float')
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred_final[i] += remove(prediction_map[i], j)
    
    
    return pred_final




def compute_pq(mode, pred, true):

    mode = mode
    pred_path = pred
    true_path = true

    seg_metrics_names = ["pq", "multi_pq+"]
    reg_metrics_names = ["r2"]

    # do initial checks
    if mode not in ["regression", "seg_class"]:
        raise ValueError("`mode` must be either `regression` or `seg_class`")

    all_metrics = {}
    if mode == "seg_class":
        # check to make sure input is a single numpy array
        pred_format = 'npy'#pred_path.split(".")[-1]
        true_format = 'npy'#true_path.split(".")[-1]
        if pred_format != "npy" or true_format != "npy":
            raise ValueError("pred and true must be in npy format.")

        # initialise empty placeholder lists
        pq_list = []
        mpq_info_list = []
        # load the prediction and ground truth arrays
        pred_array = pred
        true_array = true

        nr_patches = pred_array.shape[0]

        for patch_idx in range(nr_patches):
            # get a single patch
            pred = pred_array[patch_idx]
            true = true_array[patch_idx]

            # instance segmentation map
            pred_inst = pred[..., 0]
            true_inst = true[..., 0]
            # classification map
            pred_class = pred[..., 1]
            true_class = true[..., 1]

            # ===============================================================

            for idx, metric in enumerate(seg_metrics_names):
                if metric == "pq":
                    # get binary panoptic quality
                    pq = get_pq(true_inst, pred_inst)
                    pq = pq[0][2]
                    pq_list.append(pq)
                elif metric == "multi_pq+":
                    # get the multiclass pq stats info from single image
                    mpq_info_single = get_multi_pq_info(true, pred)
                    mpq_info = []
                    # aggregate the stat info per class
                    for single_class_pq in mpq_info_single:
                        tp = single_class_pq[0]
                        fp = single_class_pq[1]
                        fn = single_class_pq[2]
                        sum_iou = single_class_pq[3]
                        mpq_info.append([tp, fp, fn, sum_iou])
                    mpq_info_list.append(mpq_info)
                else:
                    raise ValueError("%s is not supported!" % metric)

        pq_metrics = np.array(pq_list)
        pq_metrics_avg = np.mean(pq_metrics, axis=-1)  # average over all images
        if "multi_pq+" in seg_metrics_names:
            mpq_info_metrics = np.array(mpq_info_list, dtype="float")
            # sum over all the images
            total_mpq_info_metrics = np.sum(mpq_info_metrics, axis=0)

        for idx, metric in enumerate(seg_metrics_names):
            if metric == "multi_pq+":
                dq_list = []
                sq_list = []
                mpq_list = []
                # for each class, get the multiclass PQ
                for cat_idx in range(total_mpq_info_metrics.shape[0]):
                    total_tp = total_mpq_info_metrics[cat_idx][0]
                    total_fp = total_mpq_info_metrics[cat_idx][1]
                    total_fn = total_mpq_info_metrics[cat_idx][2]
                    total_sum_iou = total_mpq_info_metrics[cat_idx][3]

                    # get the F1-score i.e DQ
                    dq = total_tp / (
                        (total_tp + 0.5 * total_fp + 0.5 * total_fn) + 1.0e-6
                    )
                    # get the SQ, when not paired, it has 0 IoU so does not impact
                    sq = total_sum_iou / (total_tp + 1.0e-6)
                    
                    dq_list.append(dq)
                    sq_list.append(sq)
                    mpq_list.append(dq * sq)
                    
                dq_metrics = np.array(dq_list)
                sq_metircs = np.array(sq_list)
                mpq_metrics = np.array(mpq_list)
                #all_metrics[metric] = [np.mean(mpq_metrics)]
                all_metrics['DQ'] = [dq_metrics]
                all_metrics['SQ'] = [sq_metircs]
                all_metrics[metric] = [mpq_metrics]
                
            else:
                all_metrics[metric] = [pq_metrics_avg]

    else:
        # first check to make sure ground truth and prediction is in csv format
        if not os.path.isfile(true_path) or not os.path.isfile(pred_path):
            raise ValueError("pred and true must be in csv format.")

        pred_format = pred_path.split(".")[-1]
        true_format = true_path.split(".")[-1]
        if pred_format != "csv" or true_format != "csv":
            raise ValueError("pred and true must be in csv format.")

        pred_csv = pd.read_csv(pred_path)
        true_csv = pd.read_csv(true_path)

        for idx, metric in enumerate(reg_metrics_names):
            if metric == "r2":
                # calculate multiclass coefficient of determination
                r2 = get_multi_r2(true_csv, pred_csv)
                all_metrics["multi_r2"] = [r2]
            else:
                raise ValueError("%s is not supported!" % metric)
                
    return all_metrics


def remap_label(pred, by_size=False):
    """Rename all instance id so that the id is contiguous i.e [0, 1, 2, 3] 
    not [0, 2, 4, 6]. The ordering of instances (which one comes first) 
    is preserved unless by_size=True, then the instances will be reordered
    so that bigger nucler has smaller ID.

    Args:
        pred (ndarray): the 2d array contain instances where each instances is marked
            by non-zero integer.
        by_size (bool): renaming such that larger nuclei have a smaller id (on-top).

    Returns:
        new_pred (ndarray): Array with continguous ordering of instances.
        
    """
    pred_id = list(np.unique(pred))
    pred_id.remove(0)
    if len(pred_id) == 0:
        return pred  # no label
    if by_size:
        pred_size = []
        for inst_id in pred_id:
            size = (pred == inst_id).sum()
            pred_size.append(size)
        # sort the id by size in descending order
        pair_list = zip(pred_id, pred_size)
        pair_list = sorted(pair_list, key=lambda x: x[1], reverse=True)
        pred_id, pred_size = zip(*pair_list)

    new_pred = np.zeros(pred.shape, np.int32)
    for idx, inst_id in enumerate(pred_id):
        new_pred[pred == inst_id] = idx + 1
    return new_pred



def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]


def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    
    crop_ann = fixed_ann
    # re-cropping with fixed instance id map
#     crop_ann = cropping_center(fixed_ann, crop_shape)

    # TODO: deal with 1 label warning
    crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)
    y_map = np.zeros(orig_ann.shape[:2], dtype=np.float32)

    inst_list = list(np.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(fixed_ann == inst_id, np.uint8)
        inst_box = get_bounding_box(inst_map)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
#         inst_box[0] -= 2
#         inst_box[2] -= 2
#         inst_box[1] += 2
#         inst_box[3] += 2
            

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if (inst_map.shape[0] < 2) | (inst_map.shape[1] < 2):
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map))
          

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        

        inst_x_range = np.arange(1, inst_map.shape[1] + 1)
        inst_y_range = np.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = np.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.astype("float32")
        inst_y = inst_y.astype("float32")

        # normalize min into -1 scale
        if np.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -np.amin(inst_x[inst_x < 0])
        if np.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -np.amin(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if np.max(inst_x) > 0:
            inst_x[inst_x > 0] /= np.amax(inst_x[inst_x > 0])
        if np.max(inst_y) > 0:
            inst_y[inst_y > 0] /= np.amax(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = np.dstack([x_map, y_map])
    return hv_map


####
def gen_targets(ann, crop_shape, **kwargs):
    """Generate the targets for the network."""
    hv_map = gen_instance_hv_map(ann, crop_shape)
    np_map = ann.copy()
    np_map[np_map > 0] = 1

#     hv_map = cropping_center(hv_map, crop_shape)
#     np_map = cropping_center(np_map, crop_shape)

    target_dict = {
        "hv_map": hv_map,
        "np_map": np_map,
    }

    return target_dict


def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    
    """
    current_max_id = np.amax(ann)
    inst_list = list(np.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = np.array(ann == inst_id, np.uint8)
        remapped_ids = measurements.label(inst_map)[0]
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = np.amax(ann)
    return ann





# def msge_loss(true, pred, focus):

#     def get_sobel_kernel(size):
#         """Get sobel kernel with a given size."""
#         assert size % 2 == 1, "Must be odd, get size=%d" % size

#         h_range = torch.arange(
#             -size // 2 + 1,
#             size // 2 + 1,
#             dtype=torch.float32,
#             device="cpu",
#             requires_grad=False,
#         )
#         v_range = torch.arange(
#             -size // 2 + 1,
#             size // 2 + 1,
#             dtype=torch.float32,
#             device="cpu",
#             requires_grad=False,
#         )
#         h, v = torch.meshgrid(h_range, v_range)
#         kernel_h = h / (h * h + v * v + 1.0e-15)
#         kernel_v = v / (h * h + v * v + 1.0e-15)
#         return kernel_h, kernel_v

#     ####
#     def get_gradient_hv(hv):
#         """For calculating gradient."""
#         kernel_h, kernel_v = get_sobel_kernel(5)
#         kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
#         kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

#         h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
#         v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

#         # can only apply in NCHW mode
#         h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
#         v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
#         dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
#         dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
#         return dhv

#     focus = (focus[..., None]).float()  # assume input NHW
#     focus = torch.cat([focus, focus], axis=-1)
#     true_grad = get_gradient_hv(true)
#     pred_grad = get_gradient_hv(pred)
#     loss = pred_grad - true_grad
    
#     loss = focus * (loss * loss)
#     # artificial reduce_mean with focused region
#     loss = loss.sum() / (focus.sum() + 1.0e-8)
#     return loss, true_grad, pred_grad



def __proc_np_hv(pred, output_hv):

    pred = np.array(pred, dtype=np.float32)

    blb_raw = pred[..., 0]
    h_dir_raw = output_hv[..., 0]
    v_dir_raw = output_hv[..., 1]

    # processing
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)

    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)
    overall[overall < 0] = 0

    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    overall = np.array(overall >= 0.4, dtype=np.int32)

    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)
    
    return proced_pred





class Unetplusplus_my2(nn.Module):
    def __init__(self):
        super(Unetplusplus_my2, self).__init__()

        self.unetplusplus1 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=7)
        
        self.unetplusplus2 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=2)
            

    def forward(self, x):
        logits1 = self.unetplusplus1(x)
        logits2 = self.unetplusplus2(x)
        return logits1, logits2

    

class Unetplusplus_my3(nn.Module):
    def __init__(self):
        super(Unetplusplus_my3, self).__init__()

        self.unetplusplus1 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=7)
        
        self.unetplusplus2 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=2)
        
        self.unetplusplus3 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=2)        

    def forward(self, x):
        logits1 = self.unetplusplus1(x)
        logits2 = self.unetplusplus2(x)
        logits3 = self.unetplusplus3(x)
        return logits1, logits2, logits3
    
    
    
    
class Unetplusplus_my3_resnet(nn.Module):
    def __init__(self):
        super(Unetplusplus_my3_resnet, self).__init__()

        self.unetplusplus1 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=7)
        
        self.unetplusplus2 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=2)
        
        self.unetplusplus3 = smp.unetplusplus.UnetPlusPlus(encoder_name="efficientnet-b0",       
                                      encoder_weights="imagenet",    
                                      in_channels=3,                 
                                      classes=2)        

    def forward(self, x):
        logits1 = self.unetplusplus1(x)
        logits2 = self.unetplusplus2(x)
        logits3 = self.unetplusplus3(x)
        return logits1, logits2, logits3    
    
    
    
    
    
from tiatoolbox.utils.misc import get_bounding_box as get_bounding_box_toatoolbox
    
def get_instance_info(pred_inst, pred_type=None):
    """To collect instance information and store it within a dictionary.

    Args:
        pred_inst (np.ndarray): An image of shape (heigh, width) which
            contains the probabilities of a pixel being a nuclei.
        pred_type (np.ndarray): An image of shape (heigh, width, 1) which
            contains the probabilities of a pixel being a certain type of nuclei.

    Returns:
        inst_info_dict (dict): A dictionary containing a mapping of each instance
                within `pred_inst` instance information. It has following form

                inst_info = {
                        box: number[],
                        centroids: number[],
                        contour: number[][],
                        type: number,
                        prob: number,
                }
                inst_info_dict = {[inst_uid: number] : inst_info}

                and `inst_uid` is an integer corresponds to the instance
                having the same pixel value within `pred_inst`.

    """
    inst_id_list = np.unique(pred_inst)[1:]  # exclude background
    inst_info_dict = {}
    for inst_id in inst_id_list:
        inst_map = pred_inst == inst_id
        inst_box = get_bounding_box_toatoolbox(inst_map)
        inst_box_tl = inst_box[:2]
        inst_map = inst_map[inst_box[1] : inst_box[3], inst_box[0] : inst_box[2]]
        inst_map = inst_map.astype(np.uint8)
        inst_moment = cv2.moments(inst_map)
        inst_contour = cv2.findContours(
            inst_map, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # * opencv protocol format may break
        inst_contour = inst_contour[0][0].astype(np.int32)
        inst_contour = np.squeeze(inst_contour)

        # < 3 points does not make a contour, so skip, likely artifact too
        # as the contours obtained via approximation => too small
        if inst_contour.shape[0] < 3:  # pragma: no cover
            continue
        # ! check for trickery shape
        if len(inst_contour.shape) != 2:  # pragma: no cover
            continue

        inst_centroid = [
            (inst_moment["m10"] / inst_moment["m00"]),
            (inst_moment["m01"] / inst_moment["m00"]),
        ]
        inst_centroid = np.array(inst_centroid)
        inst_contour += inst_box_tl[None]
        inst_centroid += inst_box_tl  # X
        inst_info_dict[inst_id] = {  # inst_id should start at 1
            "box": inst_box,
            "centroid": inst_centroid,
            "contour": inst_contour,
            "prob": None,
            "type": None,
        }

    if pred_type is not None:
        # * Get class of each instance id, stored at index id-1
        for inst_id in list(inst_info_dict.keys()):
            cmin, rmin, cmax, rmax = inst_info_dict[inst_id]["box"]
            inst_map_crop = pred_inst[rmin:rmax, cmin:cmax]
            inst_type_crop = pred_type[rmin:rmax, cmin:cmax]

            inst_map_crop = inst_map_crop == inst_id
            inst_type = inst_type_crop[inst_map_crop]

            (type_list, type_pixels) = np.unique(inst_type, return_counts=True)
            type_list = list(zip(type_list, type_pixels))
            type_list = sorted(type_list, key=lambda x: x[1], reverse=True)

            inst_type = type_list[0][0]

            # ! pick the 2nd most dominant if it exists
            if inst_type == 0 and len(type_list) > 1:  # pragma: no cover
                inst_type = type_list[1][0]

            type_dict = {v[0]: v[1] for v in type_list}
            type_prob = type_dict[inst_type] / (np.sum(inst_map_crop) + 1.0e-6)

            inst_info_dict[inst_id]["type"] = int(inst_type)
            inst_info_dict[inst_id]["prob"] = float(type_prob)

    return inst_info_dict    










def overlay_prediction_contours(
    canvas: np.ndarray,
    inst_dict: dict,
    draw_dot: bool = False,
    type_colours: dict = None,
    inst_colours: Union[np.ndarray, Tuple[int]] = (255, 255, 0),
    line_thickness: int = 2,
):
    """Overlaying instance contours on image.

    Internally, colours from `type_colours` are prioritized over
    `inst_colours`. However, if `inst_colours` is `None` and `type_colours`
    is not provided, random colour is generated for each instance.

    Args:
        canvas (ndarray): Image to draw predictions on.
        inst_dict (dict): Dictionary of instances. It is expected to be
            in the following format:
            {instance_id: {type: int, contour: List[List[int]], centroid:List[float]}.
        draw_dot (bool): To draw a dot for each centroid or not.
        type_colours (dict): A dict of {type_id : (type_name, colour)},
            `type_id` is from 0-N and `colour` is a tuple of (R, G, B).
        inst_colours (tuple, np.ndarray): A colour to assign for all instances,
            or a list of colours to assigned for each instance in `inst_dict`. By
            default, all instances will have RGB colour `(255, 255, 0).
        line_thickness: line thickness of contours.

    Returns:
        (np.ndarray) The overlaid image.

    """
    overlay = np.copy((canvas))

    if inst_colours is None:
        inst_colours = random_colors(len(inst_dict))
        inst_colours = np.array(inst_colours) * 255
        inst_colours = inst_colours.astype(np.uint8)
    elif isinstance(inst_colours, tuple):
        inst_colours = np.array([inst_colours] * len(inst_dict))
    elif not isinstance(inst_colours, np.ndarray):
        raise ValueError(
            f"`inst_colours` must be np.ndarray or tuple: {type(inst_colours)}"
        )
    inst_colours = inst_colours.astype(np.uint8)

    for idx, [_, inst_info] in enumerate(inst_dict.items()):
        inst_contour = inst_info["contour"]
        if "type" in inst_info and type_colours is not None:
            inst_colour = type_colours[inst_info["type"]][1]
        else:
            inst_colour = (inst_colours[idx]).tolist()
        cv2.drawContours(
            overlay, [np.array(inst_contour)], -1, inst_colour, line_thickness
        )

        if draw_dot:
            inst_centroid = inst_info["centroid"]
            inst_centroid = tuple([int(v) for v in inst_centroid])
            overlay = cv2.circle(overlay, inst_centroid, 3, (255, 0, 0), -1)
    return overlay










def copypaste_multiimage(copy_image, copy_mask_instance, copy_mask_class, paste_image, paste_mask_instance, paste_mask_class, copypaste_num1, copypaste_num4, copypaste_num5):
    
    image = (copy_image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()
    mask_class = copy_mask_class[0].cpu().numpy().copy()
    mask_instance = copy_mask_instance[0].cpu().numpy().copy()

    image_paste = (paste_image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8').copy()
    mask_class_paste = paste_mask_class[0].cpu().numpy().copy()
    mask_instance_paste = paste_mask_instance[0].cpu().numpy().copy()
    
#     image_instance_bool = np.zeros_like(image).astype('bool')
#     mask_instance_bool = np.zeros_like(mask_class).astype('bool')

    class_ = [1, 4, 5]

    for c in class_:

        temp = measurements.label(mask_class == c)[0]

        if c == 1:
            copypaste_num = copypaste_num1
        elif c == 4:
            copypaste_num = copypaste_num4     
        elif c == 5:
            copypaste_num = copypaste_num5    
    
    
        for i in np.unique(temp)[1:]:

            # How many times you copy the instance cell
            num = 0
            while num < copypaste_num:
                num += 1

                x_coord = np.where(np.any(temp == i, 0))[0]
                y_coord = np.where(np.any(temp == i, 1))[0]

                x_min = x_coord[0]
                x_max = x_coord[-1]

                y_min = y_coord[0]
                y_max = y_coord[-1]

                width = x_max - x_min
                height = y_max - y_min

                output_mask = temp[y_min : y_max, x_min : x_max]
                output_mask_bool = output_mask.astype('bool')


                x_random = np.random.randint(image.shape[1] - width)
                y_random = np.random.randint(image.shape[1] - height)        


                image_copy_bool = np.zeros_like(image).astype('bool')
                image_copy_bool[y_min : y_max, x_min : x_max] = output_mask_bool[..., None].repeat(3, 2)


                image_paste_bool = np.zeros_like(image).astype('bool')
                image_paste_bool[y_random : y_random + height, x_random : x_random + width] = output_mask_bool[..., None].repeat(3, 2)
         


                image_paste[image_paste_bool] = image[image_copy_bool]
                #image_instance_bool[image_paste_bool] = image[image_copy_bool]

                mask_class_paste[image_paste_bool[..., 0]] = mask_class[image_copy_bool[..., 0]]
                #mask_instance_bool[image_paste_bool[..., 0]] = mask_class[image_copy_bool[..., 0]]

                mask_instance_paste[image_paste_bool[..., 0]] = mask_instance[image_copy_bool[..., 0]]
                

                
    image_paste = transforms.ToTensor()(image_paste)
    mask_class_paste = torch.from_numpy(mask_class_paste).unsqueeze(0)
    mask_instance_paste = torch.from_numpy(mask_instance_paste).unsqueeze(0)
    
    return (copy_image, copy_mask_instance, copy_mask_class, image_paste, mask_instance_paste, mask_class_paste)


def copypaste_random_instance(image, mask_instance, mask_class, class1_copy_num, class2_copy_num, class3_copy_num, class4_copy_num, class5_copy_num, class6_copy_num, dict_, resize_rate=0.2, nooverlap=False):

    target_image = image.copy()
    target_mask_instance = mask_instance.copy()
    target_mask_class = mask_class.copy()

    class1_num = len(dict_['class1'].keys())
    class2_num = len(dict_['class2'].keys())
    class3_num = len(dict_['class3'].keys())
    class4_num = len(dict_['class4'].keys())
    class5_num = len(dict_['class5'].keys())
    class6_num = len(dict_['class6'].keys())

    class1_random_idx = np.arange(1, class1_num + 1)
    np.random.shuffle(class1_random_idx)
    class1_random_idx = class1_random_idx[:class1_copy_num]

    class2_random_idx = np.arange(1, class2_num + 1)
    np.random.shuffle(class2_random_idx)
    class2_random_idx = class2_random_idx[:class2_copy_num]

    class3_random_idx = np.arange(1, class3_num + 1)
    np.random.shuffle(class3_random_idx)
    class3_random_idx = class3_random_idx[:class3_copy_num]
    
    class4_random_idx = np.arange(1, class4_num + 1)
    np.random.shuffle(class4_random_idx)
    class4_random_idx = class1_random_idx[:class4_copy_num]

    class5_random_idx = np.arange(1, class5_num + 1)
    np.random.shuffle(class5_random_idx)
    class5_random_idx = class5_random_idx[:class5_copy_num]

    class6_random_idx = np.arange(1, class6_num + 1)
    np.random.shuffle(class6_random_idx)
    class6_random_idx = class6_random_idx[:class6_copy_num]


    for class_id, class_idx in enumerate([class1_random_idx, class2_random_idx, class3_random_idx, class4_random_idx, class5_random_idx, class6_random_idx]):


        for num in range(len(class_idx)):
            image_instance = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., :3]
            mask_instance = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., 3]
            mask_class = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., 4]


            
            
            if (image_instance.shape[0] < 2) | (image_instance.shape[1] < 2):
                continue
                
                
            
            ################ Resize ###############
            
            random_num = np.random.uniform(1, 1 + resize_rate, 1)[0]
            resize_ratio = round(random_num, 2)
            
#             random_num = np.random.uniform(0.2, 1.8, 1)[0]
#             resize_ratio_y = round(random_num, 2)            
            
            image_instance_resize = cv2.resize(image_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            mask_instance = cv2.resize(mask_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
            mask_class = cv2.resize(mask_class, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)    

            image_bool = mask_instance.astype('bool')[..., None].repeat(3, 2)
            image_instance = image_instance_resize * image_bool
            
            output = A.Compose([
                A.HorizontalFlip(p=0.5)],
                #A.VerticalFlip(p=0.5)],
                additional_targets={'mask1' : 'image',
                    'mask2' : 'image'})(image=image_instance, 
                                        mask1=mask_class,
                                        mask2=mask_instance)
            
            
            image_instance = output['image']
            #image_instance = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)(image=image_instance.astype('uint8'))['image']
            
            mask_class = output['mask1']
            mask_instance = output['mask2']         
            
            
            
            temp_bool = image_instance.astype('bool')
            
            
            height = image_instance.shape[0]
            width = image_instance.shape[1]

            
            
            if (target_image.shape[0] <= height) | (target_image.shape[1] <= width):
                continue
            
            
            x_random = np.random.randint(target_image.shape[1] - width)
            y_random = np.random.randint(target_image.shape[1] - height) 

            
            ################# where to copy? -> if thers's overlapping cell in original, then pass #################
            if nooverlap:
                if target_mask_instance[y_random : y_random + height, x_random : x_random + width].sum() > 0:
                    
                    continue            
     
            
            
            # image
            image_random = target_image[y_random : y_random + height, x_random : x_random + width]
            image_pasted = np.where(temp_bool == True, image_instance, image_random)
            target_image[y_random : y_random + height, x_random : x_random + width] = image_pasted

            # mask instance
            mask_instance_random = target_mask_instance[y_random : y_random + height, x_random : x_random + width]
            mask_instance_pasted = np.where(temp_bool[..., 0] == True, mask_instance, mask_instance_random)
            target_mask_instance[y_random : y_random + height, x_random : x_random + width] = mask_instance_pasted      


            # mask class
            mask_class_random = target_mask_class[y_random : y_random + height, x_random : x_random + width]
            mask_class_pasted = np.where(temp_bool[..., 0] == True, mask_class, mask_class_random)
            target_mask_class[y_random : y_random + height, x_random : x_random + width] = mask_class_pasted  
            
    return {"pasted_image" : target_image, 
             "pasted_mask_instance" : target_mask_instance, 
             "pasted_mask_class" : target_mask_class}









def copypaste_candidate_instance(image, mask_instance, mask_class, class1_rate=0.3, class2_rate=0.05, class3_rate=0.1, class4_rate=0.15, class5_rate=0.3, class6_rate=0.1, dict_=None, resize_rate=0, nooverlap=True):

    target_image = image.copy()
    target_mask_instance = mask_instance.copy()
    target_mask_class = mask_class.copy()
    
    
    ###################################################################################
    ###########  cal region candidate to which the instance will be copied  ########### 
    ###################################################################################
    stride = 15

    num_w = int(target_mask_class.shape[1] / stride)
    num_h = int(target_mask_class.shape[0] / stride)


    region_candidate = []
    for h in range(num_h):
        for w in range(num_w):

            x_min = w * stride
            x_max = (w + 1) * stride

            y_min = h * stride
            y_max = (h + 1) * stride        

            region_stride = target_mask_class[y_min : y_max, x_min : x_max]


            if region_stride.sum().item() == 0:
                region_candidate += [[x_min, x_max, y_min, y_max]]





    class1_num = len(dict_['class1'].keys())
    class2_num = len(dict_['class2'].keys())
    class3_num = len(dict_['class3'].keys())
    class4_num = len(dict_['class4'].keys())
    class5_num = len(dict_['class5'].keys())
    class6_num = len(dict_['class6'].keys())


    candidate_rate = np.random.uniform(0, 1)


    candidate_num = int(len(region_candidate) * candidate_rate)
    print("All Candidate num : {}".format(len(region_candidate)))
    print("Candidate_rate : {}".format(candidate_rate))
    print("Selected Candidate num : {}".format(candidate_num))
    random_idx = np.random.choice(len(region_candidate), candidate_num)




    #     all class
    #     class1_copy_num = round(len(random_idx) * class1_rate)
    #     class2_copy_num = round(len(random_idx) * class2_rate)
    #     class3_copy_num = round(len(random_idx) * class3_rate)
    #     class4_copy_num = round(len(random_idx) * class4_rate)
    #     class5_copy_num = round(len(random_idx) * class5_rate)
    #     class6_copy_num = len(random_idx) - class1_copy_num - class2_copy_num - class3_copy_num - class4_copy_num - class5_copy_num



    #   145 class   
    class1_copy_num = round(len(random_idx) * class1_rate)
    class2_copy_num = round(len(random_idx) * class2_rate)
    class3_copy_num = round(len(random_idx) * class3_rate)
    class4_copy_num = round(len(random_idx) * class4_rate)
    class5_copy_num = round(len(random_idx) * class5_rate)
    class6_copy_num = len(random_idx) - class1_copy_num - class2_copy_num - class3_copy_num - class4_copy_num - class5_copy_num
    
#     if class6_copy_num < 0:
#         class6_copy_num = 0




    all_copy_num = np.array([class1_copy_num, class2_copy_num, class3_copy_num, class4_copy_num, class5_copy_num, class6_copy_num]) 
    print("copy num : {} - total {}".format(all_copy_num.tolist(), all_copy_num.sum()))



    class1_idx = random_idx[:all_copy_num[:1].sum()]
    class2_idx = random_idx[all_copy_num[:1].sum() : all_copy_num[:2].sum()]
    class3_idx = random_idx[all_copy_num[:2].sum() : all_copy_num[:3].sum()]
    class4_idx = random_idx[all_copy_num[:3].sum() : all_copy_num[:4].sum()]
    class5_idx = random_idx[all_copy_num[:4].sum() : all_copy_num[:5].sum()]
    class6_idx = random_idx[all_copy_num[:5].sum() : all_copy_num[:6].sum()]




    class1_random_idx = np.arange(1, class1_num + 1)
    np.random.shuffle(class1_random_idx)
    class1_random_idx = class1_random_idx[:class1_copy_num]

    class2_random_idx = np.arange(1, class2_num + 1)
    np.random.shuffle(class2_random_idx)
    class2_random_idx = class2_random_idx[:class2_copy_num]

    class3_random_idx = np.arange(1, class3_num + 1)
    np.random.shuffle(class3_random_idx)
    class3_random_idx = class3_random_idx[:class3_copy_num]

    class4_random_idx = np.arange(1, class4_num + 1)
    np.random.shuffle(class4_random_idx)
    class4_random_idx = class1_random_idx[:class4_copy_num]

    class5_random_idx = np.arange(1, class5_num + 1)
    np.random.shuffle(class5_random_idx)
    class5_random_idx = class5_random_idx[:class5_copy_num]

    class6_random_idx = np.arange(1, class6_num + 1)
    np.random.shuffle(class6_random_idx)
    class6_random_idx = class6_random_idx[:class6_copy_num]    




    real_copy_num = [0, 0, 0, 0, 0, 0]



    for class_id, class_idx in enumerate([class1_random_idx, class5_random_idx, class3_random_idx, class4_random_idx, class2_random_idx, class6_random_idx]):


        if class_id == 0:
            class_id_ = 0
            class_candidate_coord = np.array(region_candidate)[class1_idx]

        if class_id == 1:
            class_id_ = 4
            class_candidate_coord = np.array(region_candidate)[class5_idx]   

        if class_id == 2:
            class_id_ = 2
            class_candidate_coord = np.array(region_candidate)[class3_idx] 

        if class_id == 3:
            class_id_ = 3
            class_candidate_coord = np.array(region_candidate)[class4_idx]  

        if class_id == 4:
            class_id_ = 1
            class_candidate_coord = np.array(region_candidate)[class2_idx]

        if class_id == 5:
            class_id_ = 5
            class_candidate_coord = np.array(region_candidate)[class6_idx] 





        for num in range(len(class_idx)):
            image_instance = dict_['class{}'.format(class_id_ + 1)]['{}'.format(class_idx[num])][..., :3]
            mask_instance = dict_['class{}'.format(class_id_ + 1)]['{}'.format(class_idx[num])][..., 3]
            mask_class = dict_['class{}'.format(class_id_ + 1)]['{}'.format(class_idx[num])][..., 4]


            if (image_instance.shape[0] < 2) | (image_instance.shape[1] < 2):
                continue



            ################ Resize ###############

            random_num = np.random.uniform(1, 1 + resize_rate, 1)[0]
            resize_ratio = round(random_num, 2)


            image_instance_resize = cv2.resize(image_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            mask_instance = cv2.resize(mask_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
            mask_class = cv2.resize(mask_class, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)    

            image_bool = mask_instance.astype('bool')[..., None].repeat(3, 2)
            image_instance = image_instance_resize * image_bool

            output = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)],
                additional_targets={'mask1' : 'image',
                    'mask2' : 'image'})(image=image_instance, 
                                        mask1=mask_class,
                                        mask2=mask_instance)


            image_instance = output['image']
            #image_instance = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)(image=image_instance.astype('uint8'))['image']

            mask_class = output['mask1']
            mask_instance = output['mask2']         



            temp_bool = image_instance.astype('bool')


            height = image_instance.shape[0]
            width = image_instance.shape[1]



            if (target_image.shape[0] <= height) | (target_image.shape[1] <= width):
                continue


            try:
                x_min = class_candidate_coord[num][0]
                y_min = class_candidate_coord[num][2]


                # if instance is out of target image, continue
                if ((y_min + height) > target_image.shape[0]) | ((x_min + width) > target_image.shape[0]):
                    continue


                ################# where to copy? -> if thers's overlapping cell in original, then pass #################
                if nooverlap:
                    if target_mask_instance[y_min : y_min + height, x_min : x_min + width].sum() > 0:
                        continue            



                # image
                image_random = target_image[y_min : y_min + height, x_min : x_min + width]
                image_pasted = np.where(temp_bool == True, image_instance, image_random)
                target_image[y_min : y_min + height, x_min : x_min + width] = image_pasted

                # mask instance
                mask_instance_random = target_mask_instance[y_min : y_min + height, x_min : x_min + width]
                mask_instance_pasted = np.where(temp_bool[..., 0] == True, mask_instance, mask_instance_random)
                target_mask_instance[y_min : y_min + height, x_min : x_min + width] = mask_instance_pasted      


                # mask class
                mask_class_random = target_mask_class[y_min : y_min + height, x_min : x_min + width]
                mask_class_pasted = np.where(temp_bool[..., 0] == True, mask_class, mask_class_random)
                target_mask_class[y_min : y_min + height, x_min : x_min + width] = mask_class_pasted
            except:
                pass

            real_copy_num[class_id_] += 1
    print("Real copy num : {} - total {}".format(real_copy_num, sum(real_copy_num)))                
            
    return {"pasted_image" : target_image, 
             "pasted_mask_instance" : target_mask_instance, 
             "pasted_mask_class" : target_mask_class}









def copypaste_candidate_instance_fixedboxes(image, mask_instance, mask_class, class1_rate=0.3, class2_rate=0.05, class3_rate=0.1, class4_rate=0.15, class5_rate=0.3, class6_rate=0.1, dict_=None, resize_rate=0, nooverlap=True):

    target_image = image.copy()
    target_mask_instance = mask_instance.copy()
    target_mask_class = mask_class.copy()
    
    
    ###################################################################################
    ###########  cal region candidate to which the instance will be copied  ########### 
    ###################################################################################
    stride = 15

    num_w = int(target_mask_class.shape[1] / stride)
    num_h = int(target_mask_class.shape[0] / stride)


    region_candidate = []
    for h in range(num_h):
        for w in range(num_w):

            x_min = w * stride
            x_max = (w + 1) * stride

            y_min = h * stride
            y_max = (h + 1) * stride        

            region_stride = target_mask_class[y_min : y_max, x_min : x_max]


            if region_stride.sum().item() == 0:
                region_candidate += [[x_min, x_max, y_min, y_max]]





    class1_num = len(dict_['class1'].keys())
    class2_num = len(dict_['class2'].keys())
    class3_num = len(dict_['class3'].keys())
    class4_num = len(dict_['class4'].keys())
    class5_num = len(dict_['class5'].keys())
    class6_num = len(dict_['class6'].keys())


    candidate_rate = np.random.uniform(0, 1)
    

    candidate_num = int(len(region_candidate) * candidate_rate)
    print("All Candidate num : {}".format(len(region_candidate)))
    print("Candidate_rate : {}".format(candidate_rate))
    print("Selected Candidate num : {}".format(candidate_num))
    random_idx = np.random.choice(len(region_candidate), candidate_num)




#     all class
#     class1_copy_num = round(len(random_idx) * class1_rate)
#     class2_copy_num = round(len(random_idx) * class2_rate)
#     class3_copy_num = round(len(random_idx) * class3_rate)
#     class4_copy_num = round(len(random_idx) * class4_rate)
#     class5_copy_num = round(len(random_idx) * class5_rate)
#     class6_copy_num = len(random_idx) - class1_copy_num - class2_copy_num - class3_copy_num - class4_copy_num - class5_copy_num
    
    
    
#   145 class   
    class1_copy_num = round(len(random_idx) * class1_rate)
    class2_copy_num = round(len(random_idx) * class2_rate)
    class3_copy_num = round(len(random_idx) * class3_rate)
    class4_copy_num = round(len(random_idx) * class4_rate)
    class6_copy_num = round(len(random_idx) * class6_rate)
    class5_copy_num = len(random_idx) - class1_copy_num - class2_copy_num - class3_copy_num - class4_copy_num - class6_copy_num    
    
    
    
    
    all_copy_num = np.array([class1_copy_num, class2_copy_num, class3_copy_num, class4_copy_num, class5_copy_num, class6_copy_num]) 
    print("copy num : {} - total {}".format(all_copy_num.tolist(), all_copy_num.sum()))



    class1_idx = random_idx[:all_copy_num[:1].sum()]
    class2_idx = random_idx[all_copy_num[:1].sum() : all_copy_num[:2].sum()]
    class3_idx = random_idx[all_copy_num[:2].sum() : all_copy_num[:3].sum()]
    class4_idx = random_idx[all_copy_num[:3].sum() : all_copy_num[:4].sum()]
    class5_idx = random_idx[all_copy_num[:4].sum() : all_copy_num[:5].sum()]
    class6_idx = random_idx[all_copy_num[:5].sum() : all_copy_num[:6].sum()]




    class1_random_idx = np.arange(1, class1_num + 1)
    np.random.shuffle(class1_random_idx)
    class1_random_idx = class1_random_idx[:class1_copy_num]

    class2_random_idx = np.arange(1, class2_num + 1)
    np.random.shuffle(class2_random_idx)
    class2_random_idx = class2_random_idx[:class2_copy_num]

    class3_random_idx = np.arange(1, class3_num + 1)
    np.random.shuffle(class3_random_idx)
    class3_random_idx = class3_random_idx[:class3_copy_num]

    class4_random_idx = np.arange(1, class4_num + 1)
    np.random.shuffle(class4_random_idx)
    class4_random_idx = class1_random_idx[:class4_copy_num]

    class5_random_idx = np.arange(1, class5_num + 1)
    np.random.shuffle(class5_random_idx)
    class5_random_idx = class5_random_idx[:class5_copy_num]

    class6_random_idx = np.arange(1, class6_num + 1)
    np.random.shuffle(class6_random_idx)
    class6_random_idx = class6_random_idx[:class6_copy_num]    

    real_copy_num = [0, 0, 0, 0, 0, 0]
    
    for class_id, class_idx in enumerate([class1_random_idx, class2_random_idx, class3_random_idx, class4_random_idx, class5_random_idx, class6_random_idx]):


        for num in range(len(class_idx)):
            image_instance = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., :3]
            mask_instance = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., 3]
            mask_class = dict_['class{}'.format(class_id + 1)]['{}'.format(class_idx[num])][..., 4]


            
            
            if (image_instance.shape[0] < 2) | (image_instance.shape[1] < 2):
                continue
                
                
            
            ################ Resize ###############
            
            random_num = np.random.uniform(1, 1 + resize_rate, 1)[0]
            resize_ratio = round(random_num, 2)
            
#             random_num = np.random.uniform(0.2, 1.8, 1)[0]
#             resize_ratio_y = round(random_num, 2)            
            
            image_instance_resize = cv2.resize(image_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
            mask_instance = cv2.resize(mask_instance, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
            mask_class = cv2.resize(mask_class, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)    

            image_bool = mask_instance.astype('bool')[..., None].repeat(3, 2)
            image_instance = image_instance_resize * image_bool
            
            output = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5)],
                additional_targets={'mask1' : 'image',
                    'mask2' : 'image'})(image=image_instance, 
                                        mask1=mask_class,
                                        mask2=mask_instance)
            
            
            image_instance = output['image']
            #image_instance = A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)(image=image_instance.astype('uint8'))['image']
            
            mask_class = output['mask1']
            mask_instance = output['mask2']         
            
            
            
            temp_bool = image_instance.astype('bool')
            
            
            height = image_instance.shape[0]
            width = image_instance.shape[1]

            
            
            if (target_image.shape[0] <= height) | (target_image.shape[1] <= width):
                continue
            
            
            x_random = np.random.randint(target_image.shape[1] - width)
            y_random = np.random.randint(target_image.shape[1] - height) 

            
            ################# where to copy? -> if thers's overlapping cell in original, then pass #################
            if nooverlap:
                if target_mask_instance[y_random : y_random + height, x_random : x_random + width].sum() > 0:
                    
                    continue            
     
            
            
            # image
            image_random = target_image[y_random : y_random + height, x_random : x_random + width]
            image_pasted = np.where(temp_bool == True, image_instance, image_random)
            target_image[y_random : y_random + height, x_random : x_random + width] = image_pasted

            # mask instance
            mask_instance_random = target_mask_instance[y_random : y_random + height, x_random : x_random + width]
            mask_instance_pasted = np.where(temp_bool[..., 0] == True, mask_instance, mask_instance_random)
            target_mask_instance[y_random : y_random + height, x_random : x_random + width] = mask_instance_pasted      


            # mask class
            mask_class_random = target_mask_class[y_random : y_random + height, x_random : x_random + width]
            mask_class_pasted = np.where(temp_bool[..., 0] == True, mask_class, mask_class_random)
            target_mask_class[y_random : y_random + height, x_random : x_random + width] = mask_class_pasted  
            
            real_copy_num[class_id] += 1
            
    return {"pasted_image" : target_image, 
             "pasted_mask_instance" : target_mask_instance, 
             "pasted_mask_class" : target_mask_class}














def mosaic_aug(selected_batch_image, selected_batch_mask_class, selected_batch_mask_instance, selected_batch_hv_map):

    output_size = (256, 256)  # Height, Width
    scale_range = (0.3, 0.7)

    output_img = np.zeros([output_size[0], output_size[1], 3], dtype=np.uint8)
    output_mask_class = np.zeros([output_size[0], output_size[1]], dtype=np.uint8)
    output_mask_instance = np.zeros([output_size[0], output_size[1]], dtype=np.uint8)
    output_hv_map = np.zeros([output_size[0], output_size[1], 2], dtype=np.float32)
    
    scale_x = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    scale_y = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])
    divid_point_x = int(scale_x * output_size[1])
    divid_point_y = int(scale_y * output_size[0])


    image1 = (selected_batch_image[0].permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    mask1_class = (selected_batch_mask_class[0][0].cpu().numpy()).astype('uint8')
    mask1_instance = (selected_batch_mask_instance[0][0].cpu().numpy()).astype('uint8')
    hv_map1 = (selected_batch_hv_map[0].permute(1, 2, 0).cpu().numpy()).astype('float32')
    
    image2 = (selected_batch_image[1].permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    mask2_class = (selected_batch_mask_class[1][0].cpu().numpy()).astype('uint8')
    mask2_instance = (selected_batch_mask_instance[1][0].cpu().numpy()).astype('uint8')
    hv_map2 = (selected_batch_hv_map[1].permute(1, 2, 0).cpu().numpy()).astype('float32')
    
    image3 = (selected_batch_image[2].permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    mask3_class = (selected_batch_mask_class[2][0].cpu().numpy()).astype('uint8')
    mask3_instance = (selected_batch_mask_instance[2][0].cpu().numpy()).astype('uint8')
    hv_map3 = (selected_batch_hv_map[2].permute(1, 2, 0).cpu().numpy()).astype('float32')
    
    image4 = (selected_batch_image[3].permute(1, 2, 0).cpu().numpy()*255).astype('uint8')
    mask4_class = (selected_batch_mask_class[3][0].cpu().numpy()).astype('uint8')
    mask4_instance = (selected_batch_mask_instance[3][0].cpu().numpy()).astype('uint8')
    hv_map4 = (selected_batch_hv_map[3].permute(1, 2, 0).cpu().numpy()).astype('float32')

    
    
    image1 = cv2.resize(image1, (divid_point_x, divid_point_y))
    mask1_class = cv2.resize(mask1_class, (divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    mask1_instance = cv2.resize(mask1_instance, (divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    hv_map1 = cv2.resize(hv_map1, (divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    
    output_img[:divid_point_y, :divid_point_x, :] = image1
    output_mask_class[:divid_point_y, :divid_point_x] = mask1_class
    output_mask_instance[:divid_point_y, :divid_point_x] = mask1_instance
    output_hv_map[:divid_point_y, :divid_point_x] = hv_map1

    
    
    image2 = cv2.resize(image2, (output_size[1] - divid_point_x, divid_point_y))
    mask2_class = cv2.resize(mask2_class, (output_size[1] - divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    mask2_instance = cv2.resize(mask2_instance, (output_size[1] - divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    hv_map2 = cv2.resize(hv_map2, (output_size[1] - divid_point_x, divid_point_y), interpolation=cv2.INTER_NEAREST)
    
    output_img[:divid_point_y, divid_point_x:output_size[1], :] = image2
    output_mask_class[:divid_point_y, divid_point_x:output_size[1]] = mask2_class
    output_mask_instance[:divid_point_y, divid_point_x:output_size[1]] = mask2_instance
    output_hv_map[:divid_point_y, divid_point_x:output_size[1]] = hv_map2
    
    

    image3 = cv2.resize(image3, (divid_point_x, output_size[0] - divid_point_y))
    mask3_class = cv2.resize(mask3_class, (divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    mask3_instance = cv2.resize(mask3_instance, (divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    hv_map3 = cv2.resize(hv_map3, (divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    
    
    output_img[divid_point_y:output_size[0], :divid_point_x, :] = image3
    output_mask_class[divid_point_y:output_size[0], :divid_point_x] = mask3_class
    output_mask_instance[divid_point_y:output_size[0], :divid_point_x] = mask3_instance
    output_hv_map[divid_point_y:output_size[0], :divid_point_x] = hv_map3
    

    image4 = cv2.resize(image4, (output_size[1] - divid_point_x, output_size[0] - divid_point_y))
    mask4_class = cv2.resize(mask4_class, (output_size[1] - divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    mask4_instance = cv2.resize(mask4_instance, (output_size[1] - divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    hv_map4 = cv2.resize(hv_map4, (output_size[1] - divid_point_x, output_size[0] - divid_point_y), interpolation=cv2.INTER_NEAREST)
    
    output_img[divid_point_y:output_size[0], divid_point_x:output_size[1], :] = image4
    output_mask_class[divid_point_y:output_size[0], divid_point_x:output_size[1]] = mask4_class
    output_mask_instance[divid_point_y:output_size[0], divid_point_x:output_size[1]] = mask4_instance
    output_hv_map[divid_point_y:output_size[0], divid_point_x:output_size[1]] = hv_map4
    
    
    output_img = transforms.ToTensor()(output_img).unsqueeze(0)
    output_mask_class = torch.from_numpy(output_mask_class).unsqueeze(0)
    output_mask_instance = torch.from_numpy(output_mask_instance).unsqueeze(0)    
    output_hv_map = torch.from_numpy(output_hv_map).permute(2, 0, 1).unsqueeze(0)

    
    return output_img, output_mask_class, output_mask_instance, output_hv_map















from torch.nn.modules.module import Module
from torch.nn.modules.loss import _Loss

class focalloss(_Loss):
    
    def __init__(self):
        super(focalloss, self).__init__()

        
    def forward(self, pred, mask, gamma):
        
        prob = torch.softmax(pred, 1)
        mask_onehot = F.one_hot(mask, num_classes=7)

        true_prob = prob[mask_onehot[:, 0, ...].permute(0, 3, 1, 2).type(torch.bool)]

        celoss = -torch.log(true_prob)

        weight = (1-true_prob) ** gamma
        
        return (weight * celoss).mean()
    
    

class dice_loss(_Loss):
    
    def __init__(self):
        super(dice_loss, self).__init__()

        
    def forward(self, pred, target):
        
        dice = dice_coef(pred, target).mean()    
        dice_loss = 1.0 - dice
        
        return dice_loss
    
    
    
def get_sobel_kernel(size):
    """Get sobel kernel with a given size."""
    assert size % 2 == 1, "Must be odd, get size=%d" % size

    h_range = torch.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
    )
    v_range = torch.arange(
        -size // 2 + 1,
        size // 2 + 1,
        dtype=torch.float32,
        device="cpu",
        requires_grad=False,
    )
    h, v = torch.meshgrid(h_range, v_range)
    kernel_h = h / (h * h + v * v + 1.0e-15)
    kernel_v = v / (h * h + v * v + 1.0e-15)
    return kernel_h, kernel_v        



####
def get_gradient_hv(hv):
    """For calculating gradient."""
    kernel_h, kernel_v = get_sobel_kernel(5)
    kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
    kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

    h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
    v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

    # can only apply in NCHW mode
    h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
    v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
    dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
    dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
    return dhv       
    
    
    
    
class msge_loss(_Loss):
    
    def __init__(self):
        super(msge_loss, self).__init__()
   
    def forward(self, true, pred, focus):
        
        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1)
        true_grad = get_gradient_hv(true)
        pred_grad = get_gradient_hv(pred)
        loss = pred_grad - true_grad

        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        
        return loss, true_grad, pred_grad


