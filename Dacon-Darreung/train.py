#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
import os
import time
import glob


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from xgboost import XGBRegressor, plot_tree, plot_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier, Pool

import torch.nn.functional as F
from itertools import permutations, combinations
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier, Pool, CatBoostRegressor
import argparse

from util.util import *
warnings.filterwarnings(action = 'ignore')



parser = argparse.ArgumentParser()
parser.add_argument('--model', default='nn', choices=['nn', 'lgbm', 'catboost', 'xgboost'] , type=str)
args = parser.parse_args()


def main(model, csv):
    
    if model == 'nn':
        train_nn(csv)
        
    elif model == 'lgbm':
        train_lgbm(csv)
        
    elif model == 'catboost':
        train_catboost(csv)
        
    elif model == 'xgboost':
        train_xgboost(csv)
        
        
if __name__ == "__main__":
    csv = pd.read_csv('data/train.csv')
    main(args.model, csv)
        

        
        