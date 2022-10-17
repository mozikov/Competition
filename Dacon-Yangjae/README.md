# AI 양재 허브 인공지능 오픈소스 경진대회 - Private 6th , 3중대 3소대장


## 0. Requirements
- torch == 1.12.1+cu113
- torchvision == 0.13.1+cu113
- segmentation-models-pytorch == 0.2.1
- albumentations == 1.1.0
- opencv-python == 4.5.2.54
- sklearn


<br />
<br />

## 1. Train
### 1.1 Dataset Preparation

```bash
├── datasets
│   ├── train
│   │   ├── lr
│   │   ├── hr
│   ├── test
│   │   ├── lr
``` 

#### 1.1.1 대회서 주어진 Dataset으로 위 폴더 구조 형성
#### 1.1.2  ```python makepatch.py ``` 실행 -> 실행 후 'datasets/patchdata_768/Foldx/lr' 폴더에 5개 Fold에 대한 Training Patch Data가 생성됨

<br />

### 1.2 Training
#### 1.2.1 ```python train.py --fold x ``` -> --fold arguments에 훈련을 원하는 fold 숫자(x) 입력 후 훈련 실시


<br />
<br />


## 2. Test
#### 2.1. 'checkpoint' 폴더 생성 후 5개 가중치를 다운로드 받아 넣어줌 (Download)[https://drive.google.com/drive/folders/1oD8iJ5gg5nNqsGjC5vhY_icJhpyDKRbt]
#### 2.2 ```python test.py``` 실행
#### 2.3 실행 후 Private 점수(24.26192) 재현이 가능한 `submission.zip` 파일이 생성됨
