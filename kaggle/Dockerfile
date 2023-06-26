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

# # opencv
# RUN git clone https://github.com/opencv/opencv.git & git clone https://github.com/opencv/opencv_contrib.git
# RUN cd opencv && git remote update && git fetch &&  git checkout tags/4.6.0
# RUN cd opencv_contrib && git remote update && git fetch && git checkout tags/4.6.0
#
# # OpenCV build
# RUN mkdir -p /opt/opencv

# RUN cd opencv && mkdir build && cd build && cmake \
# -D CMAKE_BUILD_TYPE=RELEASE \
# -D CMAKE_INSTALL_PREFIX=/opt/opencv \
# -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules/ \
# -D WITH_GSTREAMER=ON \
# -D WITH_FFMPEG=ON \
# -D WITH_V4L=ON  \
# -D WITH_XINE=ON \
# -D BUILD_TESTS=OFF \
# -D BUILD_DOCS=OFF \
# -D PYTHON3_EXECUTABLE=/usr/bin/python3 \
# -D PYTHON3_INCLUDE_DIR=/usr/include/python3.8 \
# -D PYTHON3_INCLUDE_PATH=/usr/include/python3.8 \
# -D PYTHON3_LIBRARIES=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.so \
# -D PYTHON3_LIBRARY=/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.a \
# -D PYTHON3_NUMPY_INCLUDE_DIRS=/usr/lib/python3/dist-packages/numpy/core/include \
# -D OPNECV_PYTHON_INSTALL_PATH_SETUPVARS=/usr/lib/python3/dist-packages \
# -D BUILD_EXAMPLES=OFF
#
# RUN cd opencv/build && make -j`nproc`
# RUN cp opencv/build/lib/python3/cv2.cpython-38-x86_64-linux-gnu.so /usr/local/lib/python3.8/


COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

########## Install bashrc
RUN echo "alias python=python3" >> /root/.bashrc

