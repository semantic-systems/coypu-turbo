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
RUN pip3 install -e ".[model_worker,webui]"

WORKDIR src

EXPOSE 5288/tcp

ENTRYPOINT ["python3"]

CMD ["-m", "fastchat.serve.model_worker", "--model-path", "lmsys/vicuna-13b-v1.5","--load-8bit", "--model-name", "vicuna-13b-v1.5", "--host", "0.0.0.0", "--port", "5288", "--controller-address", "http://vicuna_controller:5287", "--worker-address", "http://vicuna_worker:5288"]
