FROM python:3.9-slim

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

RUN pip3 install fschat

RUN pip3 install git+https://github.com/huggingface/transformers

WORKDIR src

EXPOSE 5287/tcp

ENTRYPOINT ["python3"]
CMD ["-m", "fastchat.serve.controller", "--host", "127.0.0.1", "--port", "5000"]