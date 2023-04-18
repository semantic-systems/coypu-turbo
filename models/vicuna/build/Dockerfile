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

WORKDIR src

EXPOSE 5289/tcp

ENTRYPOINT ["python3"]
CMD ["-m", "fastchat.serve.gradio_web_server", "--host", "127.0.0.1", "--port", "5002", "--controller-url", "http://127.0.0.1:5000"]