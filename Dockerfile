FROM ubuntu:16.04

# Install Python
ENV LANG C.UTF-8
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update

RUN apt-get install -y \
python3.6-dev \
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
RUN python3.6 get-pip.py
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3

RUN apt-get install -y \
python3-pyqt4 \
python3-pyqt4.qtopengl \
python3-tk \
swig

RUN pip3 install \
gym \
joblib \
matplotlib \
numpy \
psutil \
pygame \
pyyaml \
scikit-learn \
setuptools \
torch \
tqdm

## Install DART
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

# OpenAI Baselines
WORKDIR /app
RUN git clone https://github.com/openai/baselines.git
WORKDIR /app/baselines
RUN pip3 install tensorflow
RUN pip3 install -e .

WORKDIR /app
RUN git clone https://github.com/DartEnv/dart-env.git
COPY ./custom_dart_assets /app/dart-env/gym/envs/dart/assets
WORKDIR /app/dart-env
RUN pip3 install -e '.[dart]'

RUN pip3 install scikit-image

RUN pip3 install git+https://github.com/IssamLaradji/sls.git

RUN apt-get install libgl1-mesa-dev
RUN apt-get install libpcre16-3
RUN pip3 install roboschool

RUN export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH