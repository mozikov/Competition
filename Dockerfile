FROM nvidia/cuda:11.6.0-cudnn8-runtime-ubuntu20.04

LABEL title="ink"
LABEL maintainer="hnefa335@gmail.com"

########## Reset apt list
RUN rm /etc/apt/sources.list.d/cuda.list
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /home/dev/kaggle


### Install dependencies
RUN apt-get update
RUN apt-get install vim python3-dev python3-pip python3-setuptools python3-wheel git libopencv-dev -y

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

########## Install bashrc
RUN echo "alias python=python3" >> /root/.bashrc

