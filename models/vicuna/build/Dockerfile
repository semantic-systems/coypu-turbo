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

WORKDIR src

EXPOSE 5000/tcp

ENTRYPOINT ["python3"]
#CMD ["-u", "/src/main.py"]
CMD ["-m", "fastchat.serve.gradio_web_server", "--port", "5000", "--controller-url", "http://vicuna_controller:5001", "--worker-address", "http://vicuna_worker:5001"]