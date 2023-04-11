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

RUN pip3 install -r /src/requirements.txt

RUN pip3 install fschat

RUN pip3 install git+https://github.com/huggingface/transformers

RUN pip3 freeze

WORKDIR src

EXPOSE 5288/tcp

ENTRYPOINT ["python3"]
CMD ["-m", "fastchat.serve.model_worker", "--model-path /data/models/vicuna/7B", "--port 5000"]