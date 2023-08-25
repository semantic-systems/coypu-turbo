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

EXPOSE 5285/tcp

ENTRYPOINT ["python3"]
#CMD ["-m", "fastchat.serve.gradio_web_server", "--port", "5289", "--controller-url", "http://vicuna_controller:5287"]
CMD ["api.py"]
#CMD ["-m", "fastchat.serve.openai_api_server", "--host", "0.0.0.0",  "--port", "5285", "--controller-address", "http://vicuna_controller:5287"]
