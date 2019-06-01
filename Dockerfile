# needed only for numpy/scipy/matplotlib/etc
FROM tensorflow/tensorflow:1.10.1-gpu-py3

RUN useradd -u 1053 maksym
RUN alias lst="ls -lrth"; alias nv="watch -n 1 nvidia-smi"; alias python="python3"
RUN apt-get update -y
RUN apt-get install -y htop curl vim python3-tk git

RUN pip install --upgrade pip
RUN pip install numba==0.43.1 numexpr seaborn
RUN pip install ipdb

RUN cd /
ENTRYPOINT bash
