FROM nvidia/pytorch:21.05-py3

USER root
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

RUN chmod -R 777 /root
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda --version
RUN conda create -y --name cvt python=3.8
RUN conda activate cvt
RUN conda install -y pytorch torchvision cudatoolkit=11.3 -c pytorch
COPY . /home/cross-view-transformers
WORKDIR /home/cross-view-transformers
RUN pip install -r requirements.txt
RUN pip install -e .

#RUN chmod -R 777 .
# CMD /home/carla/CarlaUE4.sh -benchmark -fps=10 -windowed 2>&1 > /dev/null & &&\
#     python3 main.py rnd_drive --timeout=15.0

