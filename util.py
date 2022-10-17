import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import os
import time
import glob

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.nn.functional as F
from itertools import permutations, combinations
from sklearn.metrics import mean_squared_error

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
from scipy import linalg
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

import albumentations as A
import math
from datetime import datetime


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



def rmse_score(true, pred):
    score = math.sqrt(np.mean((true - pred) ** 2))
    return score


def psnr_score(true, pred, pixel_max):
    score = 20 * np.log10(pixel_max / rmse_score(true, pred))
    return score

def get_today():
    year = datetime.today().year
    month = datetime.today().month
    day = datetime.today().day  
    today = str(year) + str(month) + str(day)
    
    return today