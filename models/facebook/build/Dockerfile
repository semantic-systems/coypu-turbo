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
        python3-pip && \
    apt-get -y clean

COPY . /src/

RUN /usr/local/bin/python3.9 -m pip install -r /src/requirements.txt
RUN /usr/local/bin/python3.9 -m pip install -e /src/.

WORKDIR src

EXPOSE 5285/tcp

ENTRYPOINT ["python"]
CMD ["/src/main.py"]