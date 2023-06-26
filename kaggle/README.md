# Kaggle Ink Detection 4th solution by POSCO DX - Heeyoung Ahn


Hello!

Below you can find a outline of how to reproduce my solution for the <Vesuvius Challenge - Ink Detection> competition.
If you run into any trouble with the setup/code or have any questions please contact me at hnefa335@gmail.com
The detalis of my solution is in my solution post written in Kaggle Discusstion Tab.(https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/417779)


# Environment
- Docker 23.0.2
- Ubuntu 20.04


# Setup

### Dataset Preparation
Download competition datasets from ([kaggle link])(https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/data)
and put the zip file in ```vesuvius-challenge-ink-detection``` folder and unzip.

$ tree ./vesuvius-challenge-ink-detection -L 1
./
├── vesuvius-challenge-ink-detection.zip
├── sample_submission.csv
├── test
└── train

### Docker setup
``` 
docker build -t kaggle -f Dockerfile . 
docker run --gpus all --rm -it -v ${PWD}:/home/dev/kaggle --name kaggle kaggle:latest
```

# Train
```
python train.py --model resnet152 --pretrained_weights pretrained_weights/r3d152_KM_200ep.pth
python train.py --model resnet200 --pretrained_weights pretrained_weights/r3d200_KM_200ep.pth
python train.py --model resnext101 --pretrained_weights pretrained_weights/kinetics_resnext_101_RGB_16_best.pth 
```

During training, the results(logs, weights, etc) will be saved in ```checkpoints``` folder.


# Inference
```
python inference.py
```

This inference.py code is same as my best private score submission code (https://www.kaggle.com/code/ahnheeyoung1/ink-detection-inference)



