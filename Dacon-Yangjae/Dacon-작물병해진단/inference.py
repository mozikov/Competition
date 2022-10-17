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





#변수 설명 csv 파일 참조
crop = {'1':'딸기','2':'토마토','3':'파프리카','4':'오이','5':'고추','6':'시설포도'}
disease = {'1':{'a1':'딸기잿빛곰팡이병','a2':'딸기흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '2':{'a5':'토마토흰가루병','a6':'토마토잿빛곰팡이병','b2':'열과','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '3':{'a9':'파프리카흰가루병','a10':'파프리카잘록병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '4':{'a3':'오이노균병','a4':'오이흰가루병','b1':'냉해피해','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '5':{'a7':'고추탄저병','a8':'고추흰가루병','b3':'칼슘결핍','b6':'다량원소결핍 (N)','b7':'다량원소결핍 (P)','b8':'다량원소결핍 (K)'},
           '6':{'a11':'시설포도탄저병','a12':'시설포도노균병','b4':'일소피해','b5':'축과병'}}
risk = {'1':'초기','2':'중기','3':'말기'}






label_description = {}
for key, value in disease.items():
    label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    for disease_code in value:
        for risk_code in risk:
            label = f'{key}_{disease_code}_{risk_code}'
            label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'
list(label_description.items())[:10]









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









csv_feature_dict = {'내부 온도 1 평균': [3.4, 47.3],
 '내부 온도 1 최고': [3.4, 47.6],
 '내부 온도 1 최저': [3.3, 47.0],
 '내부 습도 1 평균': [23.7, 100.0],
 '내부 습도 1 최고': [25.9, 100.0],
 '내부 습도 1 최저': [0.0, 100.0],
 '내부 이슬점 평균': [0.1, 34.5],
 '내부 이슬점 최고': [0.2, 34.7],
 '내부 이슬점 최저': [0.0, 34.4]}








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
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
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
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_AREA)
        
        
        if self.mode == 'train':
        
            image = A.ColorJitter(brightness=0.3, contrast=0.6, saturation=0.2, hue=0.2, p=0.5)(image=image)['image']

            image = A.OneOf([
                             A.GaussianBlur(blur_limit=(1, 3), p=1),
                             A.MedianBlur(blur_limit=3, p=1),
                             A.GaussNoise (var_limit=(10.0, 50.0), p=1)
                             ], p=0.5)(image=image)['image']        


            image = A.VerticalFlip(p=0.5)(image=image)['image']
            image = A.HorizontalFlip(p=0.5)(image=image)['image']
            image = A.RandomRotate90(90, p=0.5)(image=image)['image']
            image = A.Cutout(num_holes=4, max_h_size=40, max_w_size=40, p=0.5)(image=image)['image']



            image = transforms.ToTensor()(image)
        else:
            image_vflip = A.VerticalFlip(p=1)(image=image)['image']
            image_vflip = transforms.ToTensor()(image_vflip)
            
            image_hflip = A.HorizontalFlip(p=1)(image=image)['image']  
            image_hflip = transforms.ToTensor()(image_hflip)
            
            image90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            image90 = transforms.ToTensor()(image90)

            image180 = cv2.rotate(image, cv2.ROTATE_180)
            image180 = transforms.ToTensor()(image180)

            image270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            image270 = transforms.ToTensor()(image270)
            
            image = transforms.ToTensor()(image)

        
        
        
        
        
        if self.mode == 'train':
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
        else:
            return {
                'img' : image,
                'img_hflip' : image_hflip,
                'img_vflip' : image_vflip,
                'img_90' : image90,
                'img_180' : image180,
                'img_270' : image270,
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32)
            }






device = torch.device("cuda")
batch_size = 256
class_n = 25
embedding_dim = 512
num_features = 9
max_len = 144
dropout_rate = 0




train = np.array(sorted(glob('../작물/data/train/*')))
test = np.array(sorted(glob('../작물/data/test/*')))

labelsss = pd.read_csv('../작물/data/train.csv')['label']




test_dataset = CustomDataset(test, mode = 'test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=16, shuffle=False)







log = logging.getLogger('staining_log')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

log.addHandler(streamHandler)    










total_model_name = ['resnet50', 'efficientnet_b3', 'densenet201', 'eca_vovnet39b', 'resnet18', 'res2net50_26w_4s', 'resnext50_32x4d']


for name in total_model_name:

    class CNN_Encoder(nn.Module):
        def __init__(self, class_n, rate=0.1):
            super(CNN_Encoder, self).__init__()
            self.model = timm.create_model('{}'.format(name), pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('resnet50', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('densenet201', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('resnet18', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('res2net50_26w_4s', pretrained=True, num_classes=1000)
    #         self.model = timm.create_model('resnext50_32x4d', pretrained=True, num_classes=1000)

        def forward(self, inputs):
            output = self.model(inputs)
            return output






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

    weight = glob('checkpoint/{}/*.pth'.format(name))[0]
    
    model.load_state_dict(torch.load(weight))



    temp_label = np.array([  0,  19,  21,  41,  42,  43,  44,  48,  51,  54,  57,  60,  79,
                            81,  89,  92,  95,  98,  99, 100, 102, 103, 105, 107, 108])





    model.eval()
    tqdm_dataset = tqdm(enumerate(test_dataloader))
    results = []

    TTA = True

    log.info("Model {}  Inference Start".format(name))
    log.info('\n')
    
    log.info("Total batch : {}".format(len(test_dataloader)))

    ensemble_numpy = np.zeros(shape=(len(test_dataset), 25))
    for batch, batch_item in tqdm_dataset:

        
        img = batch_item['img'].to(device)
        seq = batch_item['csv_feature'].to(device)
        img_hflip = batch_item['img_hflip'].to(device)
        img_vflip = batch_item['img_vflip'].to(device)
        img_90 = batch_item['img_90'].to(device)
        img_180 = batch_item['img_180'].to(device)
        img_270 = batch_item['img_270'].to(device)



        with torch.no_grad():

            if TTA:
                
                if (name == 'resnet50'):
                    output_original = model(img, seq)
                    output_hflip = model(img_hflip, seq)
                    output_vflip = model(img_vflip, seq)
                    output_90 = model(img_90, seq)
                    output_180 = model(img_180, seq)
                    output_270 = model(img_270, seq)


                    output = (output_original + output_hflip + output_vflip + output_90 + output_180 + output_270) / 6
                else:
                    output_original = model(img, seq)
                    output_hflip = model(img_hflip, seq)
                    output_vflip = model(img_vflip, seq)

                    output = (output_original + output_hflip + output_vflip) / 3
                
                
            else:
                output = model(img, seq)


        ensemble_numpy[batch * batch_size : (batch+1) * batch_size] = output.cpu().detach().numpy()
        output = torch.tensor(torch.argmax(torch.softmax(output, 1), dim=1), dtype=torch.int32).cpu().numpy()
        output = [temp_label[i] for i in output]
        results.extend(output)


    np.save('{}.npy'.format(name), ensemble_numpy)
    log.info("Model {}  Inference End".format(name))
    log.info("Model {}  Inference Result Saved".format(name))
    log.info('\n')




# best

result1 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[0])))
result2 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[1])))
result3 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[2])))
result4 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[3])))
result5 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[4])))
result6 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[5])))
result7 = torch.from_numpy(np.load('{}.npy'.format(total_model_name[6])))

ensemble = (result1 + result2 + result3 + result4 + result5 + result6 + result7) / 7

ensemble_ = torch.argmax(torch.softmax(ensemble, 1), 1).tolist()
preds = np.array([label_decoder[int(val)] for val in ensemble_])



submission = pd.read_csv('../작물/data/sample_submission.csv')
submission['label'] = preds
submission


submission.to_csv('submission.csv', index=False)