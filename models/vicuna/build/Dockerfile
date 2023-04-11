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

RUN pip install -r /src/requirements.txt

WORKDIR src

EXPOSE 5287/tcp

RUN nohup python3 -m fastchat.serve.controller --port 5288

RUN nohup python3 -m fastchat.serve.model_worker --model-path /data/models/vicuna/7B --port 5289

ENTRYPOINT ["python3"]
CMD ["-m fastchat.serve.gradio_web_server --port 5000"]