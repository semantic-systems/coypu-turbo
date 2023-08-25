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
RUN pip3 install -e ".[model_worker,webui]"

WORKDIR src

EXPOSE 5285/tcp

ENTRYPOINT ["python3"]
#CMD ["-m", "fastchat.serve.gradio_web_server", "--port", "5289", "--controller-url", "http://vicuna_controller:5287"]
CMD ["api.py"]