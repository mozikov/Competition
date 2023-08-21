# Kaggle Ink Detection 4th solution by POSCO DX - Heeyoung Ahn


Hello!

Below you can find a outline of how to reproduce my solution for the <Vesuvius Challenge - Ink Detection> competition.   
If you run into any trouble with the setup/code or have any questions please contact me at hnefa335@gmail.com   
The detalis of my solution is in my solution post written in Kaggle Discusstion Tab.(https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779)   


# Environment
- Docker 23.0.2
- Ubuntu 20.04
- gdown
- Kaggle API


# Setup

### Dataset preparation
By running code below, you can prepare competition datasets.
```
source prepare_datasets.sh
```

After running code, the structure of folder should be like:

```
$ tree ./vesuvius-challenge-ink-detection -L 1

./
├── vesuvius-challenge-ink-detection.zip
├── sample_submission.csv
├── test
└── train
```

### Pretrained weights preparation (for training)
By running code below, you can prepare 3 pretrained weights for training.   
```
source prepare_pretrained_weights.sh
```
All pretrained weights is MIT LICENSE, which is not against commercial use.   
- r3d152_KM_200ep.pth, r3d200_KM_200ep.pth : from https://github.com/kenshohara/3D-ResNets-PyTorch
- kinetics_resnext_101_RGB_16_best.pth : from https://github.com/okankop/Efficient-3DCNNs

### Submitted weights preparation (for inference)
By running code below, you can prepare 12 submitted weights for training.
```
source prepare_submitted_weights.sh
```

### Docker setup
``` 
docker build -t kaggle -f Dockerfile . 
docker run --gpus all --rm -it -v ${PWD}:/home/dev/kaggle --name kaggle kaggle:latest
```

※If you meet gpg error while running ```apt-get update``` in Dockerfile, try:
```
docker image prune
docker container prune
```


# Train
You can train 3 each model(leading to ensemble) using code below.
```
python train.py --model resnet152 --pretrained_weights pretrained_weights/r3d152_KM_200ep.pth
python train.py --model resnet200 --pretrained_weights pretrained_weights/r3d200_KM_200ep.pth
python train.py --model resnext101 --pretrained_weights pretrained_weights/kinetics_resnext_101_RGB_16_best.pth
```

If you want to train specific fold, you can add the argument **--valid_id**(range of 1 ~ 4) like:
```
python train.py --model resnet152 --pretrained_weights pretrained_weights/r3d152_KM_200ep.pth --valid_id 1
```

During training, the results(logs, weights, etc) will be saved in ```checkpoints``` folder.


# Inference
```
python inference.py
```

This inference code is almost same(with a little modification) as [my final submission code](https://www.kaggle.com/code/ahnheeyoung1/ink-detection-inference) which record the best private score(private leaderboard 4th place)   
In other words, this code will predict the test data in dataset folder(**./vesuvius-challenge-ink-detection**), which is the same as kaggle submission.   
