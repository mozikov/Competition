#!/usr/bin/env python
# coding: utf-8


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

from util.util import *
warnings.filterwarnings(action = 'ignore')





#######################################################################################################################
###############################                                                         ###############################
###############################                      Neural Networks                    ###############################
###############################                                                         ###############################
#######################################################################################################################
trainset = pd.read_csv('data/train.csv')
testset = pd.read_csv('data/test.csv')


trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())
trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())

testset['precipitation'] = testset['precipitation'].fillna(testset['precipitation'].median())
testset['sunshine_sum'] = testset['sunshine_sum'].fillna(testset['sunshine_sum'].median())




trainset['date'] = pd.to_datetime(trainset['date'])
trainset['Year'] = trainset['date'].dt.year
trainset['Month'] = trainset['date'].dt.month
trainset['day'] = trainset['date'].dt.day


testset['date'] = pd.to_datetime(testset['date'])
testset["Year"] = testset['date'].dt.year
testset['Month'] = testset['date'].dt.month
testset["day"] = testset['date'].dt.day


trainset.drop(['date'] , axis = 1 , inplace = True)
testset.drop(['date'] , axis = 1 , inplace = True)


target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month']
target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month']

target_columns3 = ['PM10', 'PM2.5', 'Year', 'Month']


combis1 = list(combinations(target_columns1, 2))
combis2 = list(combinations(target_columns2, 2))
combis3 = list(combinations(target_columns3, 2))

for com in combis1:
    trainset['{}*{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] * trainset['{}'.format(com[1])]
    testset['{}*{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] * testset['{}'.format(com[1])]

for com in combis2:
    trainset['{}+{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] + trainset['{}'.format(com[1])]
    testset['{}+{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] + testset['{}'.format(com[1])]
    
for com in combis3:
    trainset['{}/{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] / trainset['{}'.format(com[1])]
    testset['{}/{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] / testset['{}'.format(com[1])]
    

    
trainset.drop(['wind_mean', 'wind_max', 'day'] , axis = 1 , inplace = True)
testset.drop(['wind_mean', 'wind_max', 'day'] , axis = 1 , inplace = True)

Y = trainset['rental'].values
trainset.drop(['rental'] , axis = 1 , inplace = True )




    
scaler = StandardScaler()
trainset_nn = scaler.fit_transform(trainset)
testset_nn = scaler.transform(testset)



device = 'cuda'
model = NN(1024).to(device)
    
test_dataset = CustomDataset(testset_nn, np.zeros(shape=(testset_nn.shape[0])))

batch_size = 8
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




path = glob.glob('data/weight/*.pth')
nn_all = []
for p in path:

    model = NN(1024).to(device)
    model.load_state_dict((torch.load(p)))

    model.eval()
    output_all = []
    for i, batch in enumerate(test_dataloader):
        x = batch[0].to(device)
        y = batch[1].to(device)

        output = model(x)
        output_all += output.cpu().detach()[:, 0].tolist()

    pred = np.array(output_all) * 1000
    nn_all += [pred]

pred = np.array(nn_all).sum(0) / len(path)





#######################################################################################################################
###############################                                                         ###############################
###############################                          LGBM                           ###############################
###############################                                                         ###############################
#######################################################################################################################


trainset = pd.read_csv('data/train.csv')
testset = pd.read_csv('data/test.csv')

trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())
trainset['precipitation'] = trainset["precipitation"].replace(np.nan, 0)

testset['precipitation'] = testset['precipitation'].fillna(testset['precipitation'].median())
testset['sunshine_sum'] = testset['sunshine_sum'].fillna(testset['sunshine_sum'].median())




trainset['date'] = pd.to_datetime(trainset['date'])
trainset['Year'] = trainset['date'].dt.year
trainset['Month'] = trainset['date'].dt.month
trainset['day'] = trainset['date'].dt.day


testset['date'] = pd.to_datetime(testset['date'])
testset["Year"] = testset['date'].dt.year
testset['Month'] = testset['date'].dt.month
testset["day"] = testset['date'].dt.day

# 날짜 삭제 
trainset.drop(['date'] , axis = 1 , inplace = True )
testset.drop(['date'] , axis = 1 , inplace = True)



################################################################################
######################        Feature engineering         ######################
################################################################################
# target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# 'precipitation', 'temp_mean', 'temp_highest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year'  # Combination when multiplying21.22
target_columns1 = ['temp_mean', 'sunshine_rate', 'PM2.5'] # Best Combination when multiplying #0.1793
target_columns2 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5'] # Best Combination when Adding (For LGBM !!!!) # 20.06

combis1 = list(combinations(target_columns1, 2))
combis2 = list(combinations(target_columns2, 2))

# for com in combis1:
#     trainset['{}*{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] * trainset['{}'.format(com[1])]
#     testset['{}*{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] * testset['{}'.format(com[1])]

for com in combis2:
    trainset['{}+{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] + trainset['{}'.format(com[1])]
    testset['{}+{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] + testset['{}'.format(com[1])]


# t1 = 'temp_mean'
# t2 = 'Month'
# trainset['{}+{}'.format(t1, t2)] = trainset['{}'.format(t1)] + trainset['{}'.format(t2)]
# testset['{}+{}'.format(t1, t2)] = testset['{}'.format(t1)] + testset['{}'.format(t2)]

trainset.drop(['humidity' ,  'day' ,'wind_max', 'wind_mean'] , axis = 1 , inplace = True )
testset.drop(['humidity' ,  'day', 'wind_max', 'wind_mean'] , axis = 1 , inplace = True )





Y = trainset['rental'].values
trainset.drop(['rental'] , axis = 1 , inplace = True )


param = {'boosting_type': 'dart', 'learning_rate': 0.25774367660645275, 'n_estimators': 80, 'reg_alpha': 0.43074436854191267, 'reg_lambda': 0.058373746154573275, 'max_depth': 4, 'num_leaves': 89, 'colsample_bytree': 0.4790437715474308, 'subsample': 0.41078707765019573, 'subsample_freq': 8, 'min_child_samples': 7, 'max_bin': 982, 'random_state' : 357}

lgbm = LGBMRegressor(**param)
lgbm.fit(trainset, Y)

test_pred_lgbm = lgbm.predict(testset)
test_pred_lgbm *= 1.5









#######################################################################################################################
###############################                                                         ###############################
###############################                          Catboost                        ##############################
###############################                                                         ###############################
#######################################################################################################################
trainset = pd.read_csv('data/train.csv')
testset = pd.read_csv('data/test.csv')

trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())
trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())

testset['precipitation'] = testset['precipitation'].fillna(testset['precipitation'].median())
testset['sunshine_sum'] = testset['sunshine_sum'].fillna(testset['sunshine_sum'].median())



trainset['date'] = pd.to_datetime(trainset['date'])
trainset['Year'] = trainset['date'].dt.year
trainset['Month'] = trainset['date'].dt.month
trainset['day'] = trainset['date'].dt.day


testset['date'] = pd.to_datetime(testset['date'])
testset["Year"] = testset['date'].dt.year
testset['Month'] = testset['date'].dt.month
testset["day"] = testset['date'].dt.day

# 날짜 삭제 
trainset.drop(['date'] , axis = 1 , inplace = True )
testset.drop(['date'] , axis = 1 , inplace = True)



################################################################################
######################        Feature engineering         ######################
################################################################################
# target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# 'precipitation', 'temp_mean', 'temp_highest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year'  # Combination when multiplying21.22
target_columns1 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5'] # Best Combination when multiplying #0.1793
target_columns2 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5'] # Best Combination when Adding (For LGBM !!!!) # 20.06
# target_columns2 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5', 'Year', 'Month'] # Best Combination when Adding (For LGBM !!!!) # 20.06


combis1 = list(combinations(target_columns1, 2))
combis2 = list(combinations(target_columns2, 2))

for com in combis1:
    trainset['{}*{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] * trainset['{}'.format(com[1])]
    testset['{}*{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] * testset['{}'.format(com[1])]

# for com in combis2:
#     trainset['{}+{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] + trainset['{}'.format(com[1])]
#     testset['{}+{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] + testset['{}'.format(com[1])]


# t1 = 'temp_mean'
# t2 = 'Month'
# trainset['{}+{}'.format(t1, t2)] = trainset['{}'.format(t1)] + trainset['{}'.format(t2)]
# testset['{}+{}'.format(t1, t2)] = testset['{}'.format(t1)] + testset['{}'.format(t2)]

trainset.drop(['humidity' ,  'day' ,'wind_max', 'wind_mean'] , axis = 1 , inplace = True )
testset.drop(['humidity' ,  'day', 'wind_max', 'wind_mean'] , axis = 1 , inplace = True )

# trainset.drop(['day' ,'wind_max'] , axis = 1 , inplace = True )
# testset.drop(['day', 'wind_max'] , axis = 1 , inplace = True )

Y = trainset['rental'].values
trainset.drop(['rental'] , axis = 1 , inplace = True)

                                

param = {"loss_function" : "MAE", 'iterations': 459, 'learning_rate': 0.15934764403884843, 'l2_leaf_reg': 35, 'subsample': 0.4953048791712704, 'max_depth': 5, 'random_strength': 64, 'colsample_bylevel': 0.987480087783371, 'min_child_samples': 98, 'max_bin': 977, 'random_state' : 357}

cat = CatBoostRegressor(**param)
cat.fit(trainset, Y, verbose=False)


test_pred_cat = cat.predict(testset)
test_pred_cat *= 1.3







#######################################################################################################################
###############################                                                         ###############################
###############################                          XGBoost                        ###############################
###############################                                                         ###############################
#######################################################################################################################
trainset = pd.read_csv('data/train.csv')
testset = pd.read_csv('data/test.csv')

trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())
trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())

testset['precipitation'] = testset['precipitation'].fillna(testset['precipitation'].median())
testset['sunshine_sum'] = testset['sunshine_sum'].fillna(testset['sunshine_sum'].median())


trainset['date'] = pd.to_datetime(trainset['date'])
trainset['Year'] = trainset['date'].dt.year
trainset['Month'] = trainset['date'].dt.month
trainset['day'] = trainset['date'].dt.day


testset['date'] = pd.to_datetime(testset['date'])
testset["Year"] = testset['date'].dt.year
testset['Month'] = testset['date'].dt.month
testset["day"] = testset['date'].dt.day

# 날짜 삭제 
trainset.drop(['date'] , axis = 1 , inplace = True )
testset.drop(['date'] , axis = 1 , inplace = True)



################################################################################
######################        Feature engineering         ######################
################################################################################
# target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month', 'day']
# 'precipitation', 'temp_mean', 'temp_highest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year'  # Combination when multiplying21.22
target_columns1 = ['temp_mean', 'temp_lowest', 'sunshine_sum'] # Best Combination when multiplying #0.1793
target_columns2 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5'] # Best Combination when Adding (For LGBM !!!!) # 20.06
# target_columns2 = ['precipitation', 'temp_mean', 'temp_lowest', 'sunshine_sum', 'PM10', 'PM2.5', 'Year', 'Month'] # Best Combination when Adding (For LGBM !!!!) # 20.06


combis1 = list(combinations(target_columns1, 2))
combis2 = list(combinations(target_columns2, 2))

# for com in combis1:
#     trainset['{}*{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] * trainset['{}'.format(com[1])]
#     testset['{}*{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] * testset['{}'.format(com[1])]

for com in combis2:
    trainset['{}+{}'.format(com[0], com[1])] = trainset['{}'.format(com[0])] + trainset['{}'.format(com[1])]
    testset['{}+{}'.format(com[0], com[1])] = testset['{}'.format(com[0])] + testset['{}'.format(com[1])]


# t1 = 'temp_mean'
# t2 = 'Month'
# trainset['{}+{}'.format(t1, t2)] = trainset['{}'.format(t1)] + trainset['{}'.format(t2)]
# testset['{}+{}'.format(t1, t2)] = testset['{}'.format(t1)] + testset['{}'.format(t2)]

trainset.drop(['humidity' ,  'day' ,'wind_max', 'wind_mean'] , axis = 1 , inplace = True )
testset.drop(['humidity' ,  'day', 'wind_max', 'wind_mean'] , axis = 1 , inplace = True )

# trainset.drop(['day' ,'wind_max'] , axis = 1 , inplace = True )
# testset.drop(['day', 'wind_max'] , axis = 1 , inplace = True )

Y = trainset['rental'].values
trainset.drop(['rental'] , axis = 1 , inplace = True)







param = {"loss_function" : "MAE", 'booster': 'dart', 'max_depth': 3, 'learning_rate': 0.2201893125908342, 'n_estimators': 220, 'colsample_bytree': 0.5317917587768499, 'colsample_bylevel': 0.958765023737799, 'colsample_bynode': 0.7166770062152078, 'reg_lambda': 8.451435159016341e-05, 'reg_alpha': 1.0073430552399502e-05, 'subsample': 0.6599999999999999, 'min_child_weight': 19, 'gamma': 0.1779767013221274, 'random_state' : 357}

xgb = XGBRegressor(**param)
xgb.fit(trainset, Y)


test_pred_xgb = xgb.predict(testset)
test_pred_xgb *= 1.3









lgbm_w, xgb_w, cat_w, nn_w = 0.25, 0.05, 0.1, 0.62
final = (test_pred_lgbm * lgbm_w +
         test_pred_xgb * xgb_w +
         test_pred_cat * cat_w +
         pred * nn_w)

filename = '3중대3소대장_submission.csv'

sample_submission = pd.read_csv('data/sample_submission.csv')
sample_submission["rental"] = final-300
sample_submission.to_csv(filename, index=False)

print("Inference finished")
print("{} file made".format(filename))