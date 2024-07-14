FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
RUN apt-get update \
    && pip install gpustat \
    && apt-get -y install ffmpeg \
    && apt-get -y install tmux \
    && apt-get -y install wget \
    && apt-get -y install git \
    && apt-get -y install vim \
    && apt-get -y install gcc \
    && apt-get -y install libsm6 \
    && apt-get -y install libxext-dev \
    && apt-get -y install libxrender1 \
    && apt-get -y install libglib2.0-dev \
    && apt-get -y install default-jre \
    && apt-get -y install freeglut3-dev \
    && pip install scipy \
    && pip install pandas \
    && pip install ipdb \
    && pip install numpy \
    && pip install tqdm \
    && pip install spacy \
    && pip install blobfile \
    && pip install trimesh \
    && pip install pyrender \
    && pip install imageio-ffmpeg \
    && pip install setuptools==49.3.0 \
    && pip install timm==0.4.12 \
    && pip install transformers==4.15.0 \
    && pip install pycocoevalcap \
    && pip install fairscale==0.4.4 \
    && pip install opencv-python==4.1.2.30 \
    && pip install mmcv \
    && pip install tensorboard \
    && pip install matplotlib \
    && pip install sklearn \
    && pip install imageio \
    && pip install pyyaml \
    && pip install smplx \
    && pip install chumpy \
    && pip install h5py \
    && pip install scikit-learn \
    && pip install einops \
    && pip install torchpack==0.2.2 \
    && pip install python-igraph==0.9.1 \
    && pip install git+https://github.com/openai/CLIP.git \
    && pip install markupsafe==2.0.1
RUN wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
RUN dpkg -i ./mesa_18.3.3-0.deb || true
RUN apt install -f -y
RUN pip install pyglet==1.5.27 \
    && pip install pyopengl==3.1.6
RUN apt-get -y install libopenmpi-dev
RUN pip install mpi4py
RUN pip install aitviewer
