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

WORKDIR src

EXPOSE 5288/tcp

ENTRYPOINT ["python3"]
CMD ["-m", "fastchat.serve.model_worker", "--model-path", "/data/models/vicuna/7B", "--host", "127.0.0.1", "--port", "5000", "--controller-address", "http://vicuna_controller:5000", "--worker-address", "http://localhost:5000"]