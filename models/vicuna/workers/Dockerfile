FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl && \
    apt-get -y clean

COPY . /src/

RUN pip3 install -r /src/requirements.txt

WORKDIR src

EXPOSE 5288/tcp

ENTRYPOINT ["python3"]
CMD ["-m", "fastchat.serve.model_worker", "--model-path", "/data/models/vicuna/13B", "--load-8bit", "--host", "http://vicuna_worker", "--port", "5000", "--controller-address", "http://vicuna_controller:5000", "--worker-address", "http://vicuna_worker:5000"]
