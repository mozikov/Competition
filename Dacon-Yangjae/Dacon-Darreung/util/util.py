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

import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from xgboost import XGBRegressor, plot_tree, plot_importance
from lightgbm import LGBMRegressor
from catboost import CatBoostClassifier, Pool, CatBoostRegressor

import torch.nn.functional as F
from itertools import permutations, combinations


warnings.filterwarnings(action = 'ignore')




def NMAE(true, pred):
    score = np.mean(np.abs(true-pred) / true)
    return score


def get_nmae(pred, y):
    nmae = np.mean(abs(pred-y)/y)
    return nmae


def nmae_loss(pred, true):
    nmae = ((pred - true).abs() / true).mean()
    return nmae



class CustomDataset(Dataset):
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

  # 총 데이터의 개수를 리턴
    def __len__(self):
        
        return len(self.x)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        
        x = torch.from_numpy(self.x[idx]).type(torch.FloatTensor)
        y = torch.FloatTensor([self.y[idx] / 1000])

        return x, y


    
    
class NN(nn.Module):
    def __init__(self, dim):
        super(NN,self).__init__()

        self.dim = dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.layer1 = nn.Linear(127, self.dim, bias=True)
        self.layer2 = nn.Linear(self.dim, int(self.dim / 2), bias=True)
        self.layer3 = nn.Linear(int(self.dim / 2), int(self.dim / 4), bias=True)
        self.layer4 = nn.Linear(int(self.dim / 4), int(self.dim / 8), bias=True)
        self.layer5 = nn.Linear(int(self.dim / 8), 1, bias=True)
        
    def forward(self,x):

        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer4(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer5(x)
        x = self.relu(x)

        return x

    
    
    

def train_nn(csv):
    
    trainset_original = csv
    trainset = trainset_original.iloc[:1050, :]
    testset = trainset_original.iloc[1050:, :]


    trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())
    trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
    trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
    trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())

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

    # target_columns1 = ['temp_mean', 'sunshine_rate', 'PM2.5']
    target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month']
    target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10', 'PM2.5', 'humidity', 'Year', 'Month']
    # target_columns3 = ['PM10', 'PM2.5', 'Year', 'Month']
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
    trainset.drop(['rental'] , axis = 1 , inplace = True)

    test_y_real = testset['rental'].values
    testset.drop(['rental'] , axis = 1 , inplace = True)
    
    scaler = StandardScaler()
    trainset_nn = scaler.fit_transform(trainset)
    testset_nn = scaler.transform(testset) 
    
    
    
    
    import torch.cuda.amp as amp
    scaler = amp.GradScaler()

    nn_seed = np.arange(10) + 1
    seed_best_nmae = []
    best = []
    for s in nn_seed:

        print("Seed : {}".format(s))
        print('\n')

        now = time.localtime()
        save = '{}_{}_{}_{}_{}_{}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
        os.makedirs('checkpoint/{}'.format(save), exist_ok=True)

        device = 'cuda'
        model = NN(1024).to(device)

        train_dataset = CustomDataset(trainset_nn, Y)
        test_dataset = CustomDataset(testset_nn, test_y_real)

        batch_size = 8
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        num_epochs = 100
        # criterion = nn.MSELoss()
        criterion = nn.L1Loss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #     optimizer = torch.optim.RAdam(model.parameters(), lr=0.001)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        total_steps = int(len(train_dataset)*num_epochs/batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps, eta_min=1e-7)



        # train model

        nmae_all = []
        pred_all = []

        best_nmae = 1
        for epoch in range(0, num_epochs):

            #print("Current LR : {:.6f}".format(scheduler.get_lr()[0]))

            model.train()
            for i, batch in enumerate(train_dataloader):


                scheduler.step()

                x = batch[0].to(device)
                y = batch[1].to(device)

                with amp.autocast():
                    output = model(x)

                loss = nmae_loss(output, y)

                optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()

                scaler.scale(loss).backward()
                scaler.step(optimizer)    
                scaler.update() 

            model.eval()
            output_all = []
            for i, batch in enumerate(test_dataloader):
                x = batch[0].to(device)
                y = batch[1].to(device)

                output = model(x)
                output_all += output.cpu().detach()[:, 0].tolist()

            pred = np.array(output_all) * 1000

            pred_all += [pred]
            nmae_all += [get_nmae(pred, test_y_real)]


            nmae = get_nmae(pred, test_y_real)
            if nmae < best_nmae:
                best_nmae = nmae
                torch.save(model.state_dict(), 'checkpoint/{}/epoch_{}_mae_{:.6f}.pth'.format(save, epoch + 1, get_nmae(pred, test_y_real)))
    #             print("epoch : {}, test nmae : {:.6f}".format(epoch + 1, get_nmae(pred, test_y_real)))
    #             print('\n')

        min_idx = np.argmin(np.array(nmae_all))
        print("Best epoch : {}, Best nmae : {:.6f}".format(min_idx + 1, np.array(nmae_all)[min_idx]))
        print('\n')

        seed_best_nmae += [np.array(nmae_all)[min_idx]]

        best += [np.array(nmae_all)[min_idx]]

        
        
        
        
def train_lgbm(csv):
    
    trainset_original = csv
    trainset = trainset_original.iloc[:1050, :]
    testset = trainset_original.iloc[1050:, :]


    trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())
    trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
    trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
    trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())

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

        

    trainset.drop(['day' ,'wind_max', 'wind_mean'] , axis = 1 , inplace = True )
    testset.drop(['day', 'wind_max', 'wind_mean'] , axis = 1 , inplace = True )

    # trainset.drop(['day' ,'wind_max'] , axis = 1 , inplace = True )
    # testset.drop(['day', 'wind_max'] , axis = 1 , inplace = True )

    Y = trainset['rental'].values
    trainset.drop(['rental'] , axis = 1 , inplace = True)
    
    test_y_real = testset['rental'].values
    testset.drop(['rental'] , axis = 1 , inplace = True)
    
    def objective(trial):
        param = {
            "boosting_type": trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            "random_state": 357,
            "verbosity": -1,
            "learning_rate": trial.suggest_uniform('learning_rate', 0.1, 0.5),
            "n_estimators": trial.suggest_int("n_estimators", 10, 300, step=10),
            "objective": "mae",
            "metric": "nmae",
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 1),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "num_leaves": trial.suggest_int("num_leaves", 50, 150),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "max_bin": trial.suggest_int("max_bin", 800, 1000),
        }     


        lgbm = LGBMRegressor(**param)
        lgbm.fit(trainset, Y, verbose=False)

        ex = ((np.arange(10) + 1) / 10) + 1
        result_all = []
        for e in ex:
            test_pred = lgbm.predict(testset)
            test_pred *= e

            test_nmae = get_nmae(test_pred, test_y_real)
            result_all += [test_nmae]

        min_idx_lgbm = np.argmin(np.array(result_all))
        best_e = ex[min_idx_lgbm]

        best_score = np.array(result_all)[min_idx_lgbm]

        return best_score


    sampler = TPESampler(seed=357)
    study = optuna.create_study(
        study_name = 'lgbm',
        direction = 'minimize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=30000)
    print("Best Score:", study.best_value)
    print("Best trial", study.best_trial.params)

    
    
    
def train_xgboost(csv):
    
    trainset_original = csv
    trainset = trainset_original.iloc[:1050, :]
    testset = trainset_original.iloc[1050:, :]


    trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())
    trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
    trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
    trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())

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

    test_y_real = testset['rental'].values
    testset.drop(['rental'] , axis = 1 , inplace = True)
    

    
    def objective(trial):


        param = {
            "eval_metric":'mae',
            "booster":  trial.suggest_categorical('booster',['dart']),
            "tree_method": 'exact', 'gpu_id': -1,
            "verbosity": 0,
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_uniform('learning_rate', 0.1, 0.4),
            'n_estimators': trial.suggest_int("n_estimators", 10, 300, step=10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0., 1.0),
            "colsample_bynode": trial.suggest_float("colsample_bynode", 0.5, 1.0),
            "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-5, 1),
            "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-5, 1),
            'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.01),     
            'min_child_weight': trial.suggest_int('min_child_weight', 10, 30),
            "gamma": trial.suggest_float("gamma", 0.1, 1.0, log=True),
            "loss_function" : "MAE",
            'random_state' : 357}


        xgb = XGBRegressor(**param)
        xgb.fit(trainset, Y, verbose=False)

        ex = ((np.arange(10) + 1) / 10) + 1
        result_all = []
        for e in ex:
            test_pred = xgb.predict(testset)
            test_pred *= e

            test_nmae = get_nmae(test_pred, test_y_real)
            result_all += [test_nmae]

        min_idx_lgbm = np.argmin(np.array(result_all))
        best_e = ex[min_idx_lgbm]

        best_score = np.array(result_all)[min_idx_lgbm]

        return best_score


    sampler = TPESampler(seed=357)
    study = optuna.create_study(
        study_name = 'xgb',
        direction = 'minimize',
    )
    study.optimize(objective, n_trials=30000)
    print("Best Score:", study.best_value)
    print("Best trial", study.best_trial.params)


    
    
def train_catboost(csv):
    
    trainset_original = csv
    trainset = trainset_original.iloc[:1050, :]
    testset = trainset_original.iloc[1050:, :]


    trainset['precipitation'] = trainset['precipitation'].fillna(trainset['precipitation'].median())
    trainset["PM10"] = trainset["PM10"].fillna(trainset["PM10"].median())
    trainset["PM2.5"] = trainset["PM2.5"].fillna(trainset["PM2.5"].median())
    trainset["sunshine_sum"] = trainset["sunshine_sum"].fillna(trainset["sunshine_sum"].median())

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
    
    


    ################################################################################
    ######################        Feature engineering         ######################
    ################################################################################
    target_columns1 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10',
                       'PM2.5', 'humidity', 'Year', 'Month']
    target_columns2 = ['precipitation', 'temp_mean', 'temp_highest', 'temp_lowest', 'sunshine_sum', 'sunshine_rate', 'PM10',
                       'PM2.5', 'humidity', 'Year', 'Month']
    # target_columns3 = ['PM10', 'PM2.5', 'Year', 'Month']
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
    trainset.drop(['humidity' ,  'day' ,'wind_max', 'wind_mean'] , axis = 1 , inplace = True )
    testset.drop(['humidity' ,  'day', 'wind_max', 'wind_mean'] , axis = 1 , inplace = True )

    Y = trainset['rental'].values
    trainset.drop(['rental'] , axis = 1 , inplace = True)
    

    test_y_real = testset['rental'].values
    testset.drop(['rental'] , axis = 1 , inplace = True)
    

    
    def objective(trial):
        # param = {
        #     "random_state": 357,
        #     'iterations' : trial.suggest_int("iterations", 100, 500),
        #     'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.9),
        #     "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 10, 100),
        #     "subsample" : trial.suggest_float("subsample", 0.1, 0.9),
        #     'bagging_temperature': trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
        #     "max_depth": trial.suggest_int("max_depth", 4, 16),
        #     'random_strength': trial.suggest_int('random_strength', 0, 100),
        #     "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.4, 1.0),
        #     "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        # }

        param = {
            "random_state": 357,
            'iterations' : trial.suggest_int("iterations", 400, 600),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.1, 0.2),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 30, 50),
            "subsample" : trial.suggest_float("subsample", 0.3, 0.5),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            'random_strength': trial.suggest_int('random_strength', 60, 80),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.8, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "loss_function" : "MAE",
            "max_bin": trial.suggest_int("max_bin", 400, 1000),
        }

        # cat_features = [0, 1, 2, 5, 6, 7, 8, 15, 18]
        cat = CatBoostRegressor(**param)
        cat.fit(trainset, Y, verbose=False)

        ex = ((np.arange(10) + 1) / 10) + 1
        result_all = []
        for e in ex:
            test_pred = cat.predict(testset)
            test_pred *= e

            test_nmae = get_nmae(test_pred, test_y_real)
            result_all += [test_nmae]

        min_idx_lgbm = np.argmin(np.array(result_all))
        best_e = ex[min_idx_lgbm]

        best_score = np.array(result_all)[min_idx_lgbm]

        return best_score


    sampler = TPESampler(seed=357)
    study = optuna.create_study(
        study_name = 'catboost',
        direction = 'minimize',
        sampler = sampler,
    )
    study.optimize(objective, n_trials=30000)
    print("Best Score:",study.best_value)
    print("Best trial",study.best_trial.params)

