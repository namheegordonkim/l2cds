FROM nvidia/cuda:11.4.1-runtime-ubuntu18.04

# Install Python
ENV LANG C.UTF-8
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

# to prevent timezone setup from asking questions
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get install -y \
python3.7-dev \
wget

# DART dependencies
RUN apt-get install -y \
build-essential \
cmake \
coinor-libipopt-dev \
freeglut3-dev \
git \
libassimp-dev \
libboost-regex-dev \
libboost-system-dev \
libbullet-dev \
libccd-dev \
libeigen3-dev \
libfcl-dev \
libflann-dev \
libnlopt-dev \
libode-dev \
libopenscenegraph-dev \
libtinyxml2-dev \
liburdfdom-dev \
libxi-dev \
libxmu-dev \
mesa-utils \
libgl1-mesa-glx \
pkg-config

# PyDART dependencies
RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.7 get-pip.py
RUN ln -s /usr/bin/python3.7 /usr/local/bin/python3

RUN apt-get install -y \
python3-pyqt4 \
python3-pyqt4.qtopengl \
python3-tk \
swig

# could add torch==1.9.1+cu111 for GPU support
RUN pip3 install \
torch==1.9.1+cu111 \
gym \
joblib \
matplotlib \
numpy \
psutil \
pygame \
pyyaml \
scipy \
scikit-learn==0.22.0 \
setuptools \
tqdm \
stable-baselines3 \
-f https://download.pytorch.org/whl/torch_stable.html

### Install DART
WORKDIR /app
RUN git clone git://github.com/dartsim/dart.git
WORKDIR /app/dart
RUN git checkout tags/v6.6.2
RUN mkdir build
WORKDIR /app/dart/build
RUN cmake ..
RUN make -j4
RUN make install

# PyDart 2
WORKDIR /app
RUN git clone https://github.com/sehoonha/pydart2.git
WORKDIR /app/pydart2
RUN python3 setup.py build build_ext
RUN python3 setup.py develop

WORKDIR /app
RUN git clone https://github.com/DartEnv/dart-env.git
COPY ./custom_dart_assets /app/dart-env/gym/envs/dart/assets
WORKDIR /app/dart-env
RUN pip3 install -e '.[dart]'

RUN apt-get install libgl1-mesa-dev
RUN apt-get install libpcre16-3

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# make sure to bind mount scripts here
RUN mkdir /l2cds
WORKDIR /l2cds
