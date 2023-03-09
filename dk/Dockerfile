# 130 GPU
# ---------- face detection recognition ----------

FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

RUN echo "" > /etc/apt/sources.list.d/cuda.list
RUN sed -i "s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN sed -i "s@/security.ubuntu.com/@/mirrors.aliyun.com/@g" /etc/apt/sources.list
RUN apt-get update --fix-missing && apt-get install -y fontconfig --fix-missing
#RUN apt-get install -y python3.7 python3-pip
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && echo "Asia/Shanghai" > /etc/timezone

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torchvision==0.8.0 torchaudio==0.7.0
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple certifi==2021.5.30 chardet==3.0.4 click==8.0.3 colorama==0.4.4 cycler==0.10.0 dataclasses
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple easydict==1.9 Flask==2.0.2 future==0.18.2 graphviz==0.8.4 idna==2.6 importlib-metadata==4.8.1
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple itsdangerous==2.0.1 Jinja2==3.0.2 joblib==1.0.1 kiwisolver==1.3.1 MarkupSafe==2.0.1 matplotlib==3.3.4
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mxnet==1.7.0.post2 numpy==1.16.6 olefile==0.46 opencv-python==4.5.2.54 Pillow==8.4.0 protobuf==3.17.2
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pyparsing==2.4.7 python-dateutil==2.8.1 requests==2.18.4 scikit-learn==0.24.2 scipy==1.5.4 setuptools==58.0.4
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple six==1.16.0 tensorboardX==2.2 threadpoolctl==2.1.0 tqdm==4.61.0 typing-extensions==3.10.0.2 urllib3==1.22
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Werkzeug==2.0.2 wheel==0.37.0 wincertstore==0.2 zipp==3.6.0 easydict
RUN apt-get install libgl1-mesa-glx -y
RUN apt-get install libglib2.0-dev -y
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bcolz-zipline

WORKDIR /home

# cd /data/wgs/face_recognition
# docker build -t wgs-torch/face_recognition:1.0 -f ./dk/Dockerfile .
