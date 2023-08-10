# hash:sha256:d7e264e2f132c11afbcbaa3d329ce0b4f65310c01bdace9da72de311b6aac790
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        docker.io=20.10.21-0ubuntu1~20.04.2 \
        gcc-9=9.4.0-1ubuntu1~20.04.1 \
        libdeal.ii-dev \
    && rm -rf /var/lib/apt/lists/*

COPY postInstall /
RUN /postInstall
