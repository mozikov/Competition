import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os
import json 
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

import albumentations as A

from torchvision import transforms

import logging

import timm

from sklearn.metrics import f1_score

import random

import torch.nn.functional as F
import random



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





sample = glob('data/train/*')[42]

sample_csv = pd.read_csv(glob(sample+'/*.csv')[0])
sample_image = cv2.imread(glob(sample+'/*.jpg')[0])
sample_json = json.load(open(glob(sample+'/*.json')[0], 'r'))






csv_features = ['내부 온도 1 평균', '내부 온도 1 최고', '내부 온도 1 최저', '내부 습도 1 평균', '내부 습도 1 최고', 
                '내부 습도 1 최저', '내부 이슬점 평균', '내부 이슬점 최고', '내부 이슬점 최저']

csv_files = sorted(glob('data/train/*/*.csv'))

temp_csv = pd.read_csv(csv_files[0])[csv_features]
max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()






csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3],
 '내부 온도 1 최고': [3.4, 47.6],
 '내부 온도 1 최저': [3.3, 47.0],
 '내부 습도 1 평균': [23.7, 100.0],
 '내부 습도 1 최고': [25.9, 100.0],
 '내부 습도 1 최저': [0.0, 100.0],
 '내부 이슬점 평균': [0.1, 34.5],
 '내부 이슬점 최고': [0.2, 34.7],
 '내부 이슬점 최저': [0.0, 34.4]}






label_description = {
"1_00_0" : "딸기", 
"2_00_0" : "토마토",
"2_a5_2" : "토마토_흰가루병_중기",
"3_00_0" : "파프리카",
"3_a9_1" : "파프리카_흰가루병_초기",
"3_a9_2" : "파프리카_흰가루병_중기",
"3_a9_3" : "파프리카_흰가루병_말기",
"3_b3_1" : "파프리카_칼슘결핍_초기",
"3_b6_1" : "파프리카_다량원소결필(N)_초기",
"3_b7_1" : "파프리카_다량원소결필(P)_초기",
"3_b8_1" : "파프리카_다량원소결필(K)_초기",
"4_00_0" : "오이",
"5_00_0" : "고추",
 "5_a7_2" : "고추_탄저병_중기",
 "5_b6_1" : "고추_다량원소결필(N)_초기",
"5_b7_1" : "고추_다량원소결필(P)_초기",
 "5_b8_1" : "고추_다량원소결필(K)_초기",
"6_00_0" : "시설포도",
"6_a11_1" : "시설포도_탄저병_초기",
 "6_a11_2" : "시설포도_탄저병_중기",
 "6_a12_1" : "시설포도_노균병_초기",
"6_a12_2" : "시설포도_노균병_중기",
 "6_b4_1" : "시설포도_일소피해_초기",
 "6_b4_3" : "시설포도_일소피해_말기",
"6_b5_1" : "시설포도_축과병_초기"   }




label_encoder = {key:idx for idx, key in enumerate(label_description)}
label_decoder = {val:key for key, val in label_encoder.items()}



class CustomDataset(Dataset):
    def __init__(self, files, labels=None, mode='train'):
        self.mode = mode
        self.files = files
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.max_len = 144
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('/')[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            
            
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
                
            # zero padding
            pad = np.zeros((self.max_len, 9))
            length = min(self.max_len, len(df))
            pad[-length:] = df[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        
        # image
        image_path = f'{file}/{file_name}.jpg'
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        if self.mode == 'train':
        
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            points = json_file['annotations']['bbox'][0]
            x1, y1, x2, y2 = int(points['x']), int(points['y']), int((points['x']+points['w'])), int((points['y']+points['h']))
            
            
            
            a = x1
            b = image.shape[1] - x2
            c = y1
            d = image.shape[0] - y2
            
            width_offset = np.random.choice(min(a + 1, b + 1, c + 1, d + 1), 1)[0]
            height_offset = np.random.choice(min(a + 1, b + 1, c + 1, d + 1), 1)[0]
            
            # Box Crop
            if random.random() > 1:
                image = image[y1 - height_offset : y2 + height_offset, x1 - width_offset : x2 + width_offset, :]
                image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            else:        
                image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)

        
        else:
            image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        
        
        if self.mode == 'train':
        
            image = A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5)(image=image)['image']
            image = A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, interpolation=0, border_mode=4, p=0.5)(image=image)['image']
#             image = A.OneOf([
#                               A.GaussianBlur(blur_limit=(1, 3), p=1),
#                               A.MedianBlur(blur_limit=3, p=1),
#                               A.GaussNoise (var_limit=(10.0, 30.0), p=1)
#                               ], p=0.5)(image=image)['image']       

#             image = A.GridDistortion(distort_limit=(-0.3, 0.3), border_mode=cv2.BORDER_CONSTANT, p=1)(image=image)['image']
            image = A.VerticalFlip(p=0.5)(image=image)['image']
            image = A.HorizontalFlip(p=0.5)(image=image)['image']
            #image = A.RandomRotate90(90, p=0.5)(image=image)['image']
            image = A.CoarseDropout(max_holes=4, max_height=40, max_width=40, 
                                    min_holes=2, min_height=20, min_width=20, p=0.5)(image=image)['image']



            image = transforms.ToTensor()(image)
        else:
            image = transforms.ToTensor()(image)

        
        
        
        
        
        json_path = f'{file}/{file_name}.json'
        with open(json_path, 'r') as f:
            json_file = json.load(f)

        crop = json_file['annotations']['crop']
        disease = json_file['annotations']['disease']
        risk = json_file['annotations']['risk']
        label = f'{crop}_{disease}_{risk}'

        return {
            'img' : image,
            'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
            'label' : torch.tensor(self.label_encoder[label], dtype=torch.long)
        }



device = torch.device("cuda")
batch_size = 16
class_n = len(label_encoder)
embedding_dim = 512
num_features = len(csv_feature_dict)
max_len = 144
dropout_rate = 0




train = np.array(sorted(glob('data/train/*')))
test = np.array(sorted(glob('data/test/*')))

labelsss = pd.read_csv('data/train.csv')['label']
train_labels = np.array([label_encoder[i] for i in labelsss.values])







from datetime import datetime

year = datetime.today().year
month = datetime.today().month
day = datetime.today().day  

today = str(year) + str(month) + str(day)  






from sklearn.model_selection import StratifiedKFold
n_splits = 100
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state = 357) 


for fold, (train_index, test_index) in enumerate(kf.split(train, train_labels)):
    
    X_train, X_test = train[train_index], train[test_index] 
    y_train, y_test = train_labels[train_index], train_labels[test_index]
    
    if fold == 0:
        break




run_info = '{}_resnet18_fold{}_{}'.format(today, n_splits, fold+1)


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



        
        
        
        
train_dataset = CustomDataset(X_train, mode = 'train')
test_dataset = CustomDataset(X_test, mode = 'test')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=16, shuffle=False)




# # 모델
# 
# ## 이미지 분류 모델 : Resnet50

# In[192]:


class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('densenet201', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('resnet18', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('res2net50_26w_4s', pretrained=True, num_classes=1000)
#         self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=1000)

    def forward(self, inputs):
        output = self.model(inputs)
        return output


    

# In[193]:


class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(RNN_Decoder, self).__init__()
        self.lstm = nn.LSTM(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n) # resnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output




class CNN2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_features, class_n, rate)

    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)

        return output




model = CNN2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features, class_n=class_n, rate=dropout_rate)
model = model.to(device)



optimizer = torch.optim.RAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)



def focalloss(pred, mask, gamma):

    prob = torch.softmax(pred, 1)
    mask_onehot = F.one_hot(mask, num_classes=25)

    true_prob = prob[mask_onehot.type(torch.bool)]

    celoss = -torch.log(true_prob)

    weight = ((1-true_prob) ** gamma)
    return (weight * celoss).mean()



def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score


log.info("Training Set : {}".format(len(train_dataset)))
log.info("Test Set : {}".format(len(test_dataset)))
log.info('\n')


######################################################### TRAIN #########################################################


num_epochs = 300
cutmix = True
beta = 1

for epoch in range(num_epochs):

    log.info("Epoch : {}".format(epoch+1))

    train_loss_sum = []
    train_pred_label = []
    train_true_label = []
    F1_score = []

    model.train()
    for idx, batch in enumerate(train_dataloader):

        image = batch['img'].to(device)
        csv_feature = batch['csv_feature'].to(device)
        label = batch['label'].to(device)
        
        
        
        if cutmix and random.random() > 0.5:
            
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(image.size()[0]).cuda()
            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            
            target_a = label
            target_b = label[rand_index]
            
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            output = model(image, csv_feature)     
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        
        
        else:
            output = model(image, csv_feature)

            loss = criterion(output, label)
    #         loss = focalloss(output, label, 2)


        f1 = accuracy_function(label, output)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

#         train_loss_sum += [loss.detach().cpu().tolist()]
        train_true_label += label.cpu().numpy().tolist()
        train_pred_label += torch.argmax(torch.softmax(output, 1), 1).tolist()
        F1_score += [f1]

        true_number = np.array(train_true_label) == np.array(train_pred_label)
#         train_loss = sum(train_loss_sum) / len(train_loss_sum)
        Accuracy = sum(true_number) / len(true_number)
        F1_Score = sum(F1_score) / len(F1_score)




    log.info("Train Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, F1 : {:.4f}".format(epoch+1, num_epochs, idx+1, len(train_dataloader), Accuracy, F1_Score))
    log.info("\n")

        



    test_loss_sum = []
    test_pred_label = []
    test_true_label = []
    F1_score = []

    model.eval()    
    for i, batch in enumerate(test_dataloader):

        image = batch['img'].to(device)
        csv_feature = batch['csv_feature'].to(device)
        label = batch['label'].to(device)
        output = model(image, csv_feature)

        loss = criterion(output, label)
        f1 = accuracy_function(label, output)


        test_loss_sum += [loss.detach().cpu().tolist()]
        test_true_label += label.cpu().numpy().tolist()
        test_pred_label += torch.argmax(torch.softmax(output, 1), 1).tolist()

        true_number = np.array(test_true_label) == np.array(test_pred_label)
        test_loss = sum(test_loss_sum) / len(test_loss_sum)
        Accuracy = sum(true_number) / len(true_number)

        F1_score += [f1]
        F1_Score = sum(F1_score) / len(F1_score)




    log.info("TEST Epoch : {}/{}, Batch : {}/{}, ACC : {:.3f}, F1 : {:.4f}, Loss : {:.3f}".format(epoch+1, num_epochs, i+1, len(test_dataloader), Accuracy, F1_Score, test_loss))
    log.info("\n")

    if F1_Score > 0.8:
        torch.save(model.state_dict(), 'checkpoint/{}/Epoch {} ACC {:.3f} F1 : {:.4f} TEST Loss{:.3f}.pth'.format(run_info, epoch+1, Accuracy, F1_Score, test_loss))



# In[ ]:





# In[ ]:




