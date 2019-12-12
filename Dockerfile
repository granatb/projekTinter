FROM ubuntu:latest
MAINTAINER fnndsc "dev@babymri.org"

RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && apt-get install -y libsm6 libxext6 libxrender-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip
COPY requirements.txt .
COPY classify-dirt.py .
COPY train-dirt.py .
RUN pip3 install -r requirements.txt

ENTRYPOINT ["python3"]
