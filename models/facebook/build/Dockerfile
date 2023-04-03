FROM python:3.9-slim

RUN \
    apt-get update && \
    apt-get -y upgrade && \
    apt-get -y dist-upgrade && \
    apt-get -y autoremove && \
    apt-get -y install \
        git \
        curl \
    && \
    apt-get -y clean

COPY . /src/

RUN pip install -r /src/requirements.txt
RUN pip install -e /src/.

WORKDIR src

EXPOSE 5285/tcp

ENTRYPOINT ["python"]
CMD ["/src/main.py"]
