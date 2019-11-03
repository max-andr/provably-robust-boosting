# needed only for numpy/scipy/matplotlib/etc
FROM tensorflow/tensorflow:1.10.1-py3

RUN alias lst="ls -lrth"; alias nv="watch -n 1 nvidia-smi"; alias python="python3"
RUN apt-get update -y
RUN apt-get install -y htop wget curl vim python3-tk git memory_profiler

RUN pip install --upgrade pip
RUN pip install numba==0.43.1 numexpr seaborn billiard robustml
RUN pip install ipdb

RUN cd /
ENTRYPOINT bash
