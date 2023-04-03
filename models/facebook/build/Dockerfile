FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl && \
    apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip \
        nvidia-container-toolkit && \
    apt-get -y clean

COPY . /src/

RUN pip install -r /src/requirements.txt
RUN pip install -e /src/.

WORKDIR src

EXPOSE 5285/tcp

ENTRYPOINT ["python3"]
CMD ["/src/main.py"]