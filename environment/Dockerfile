# hash:sha256:d7e264e2f132c11afbcbaa3d329ce0b4f65310c01bdace9da72de311b6aac790
FROM registry.codeocean.com/codeocean/ubuntu:20.04-cuda11.7.0-cudnn8

ARG DEBIAN_FRONTEND=noninteractive
ARG VERSION=9.4.0-1ubuntu2~bpo20.04.1~ppa1
ARG REPO=ppa:ginggs/deal.ii-9.4.0-backports
ARG CLANG_VERSION=11
ARG CLANG_REPO=https://github.com/dealii/dealii/releases/download/v9.3.0/
ARG VER=v9.4.2
ARG PROCS=2

RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository $REPO \
    && apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git \
    libboost-dev \
    libboost-python-dev \
    libcgal-dev \
    libdeal.ii-dev=$VERSION \
    libflint-dev \
    libgmp-dev \
    libgtest-dev \
    libmpc-dev \
    locales \
    ninja-build \
    numdiff \
    ssh \
    sudo \
    wget \
    && apt-get remove -y libdeal.ii-dev=$VERSION libcgal-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

# CGAL
RUN cd /usr/src && \
    wget https://github.com/CGAL/cgal/releases/download/v5.5.2/CGAL-5.5.2.tar.xz && \
    tar xvf CGAL-5.5.2.tar.xz && \
    rm -rf CGAL-5.5.2.tar.xz  && \
    cd CGAL-5.5.2/  && \ 
    mkdir build && cd build && \
    cmake -GNinja .. -DCMAKE_INSTALL_PREFIX=/usr/ && \
    ninja install && cd ../ && rm -rf build 

RUN cd /usr/src \
    && git clone https://github.com/dealii/dealii dealii-$VER \
    && cd dealii-$VER && git checkout $VER && \
    mkdir build && cd build \
    && cmake -GNinja \
    -DCMAKE_PREFIX_PATH="/usr/lib/x86_64-linux-gnu/hdf5/openmpi;/usr/include/hdf5/openmpi" \
    -DDEAL_II_COMPILE_EXAMPLES=OFF \
    -DCMAKE_CXX_FLAGS="-std=c++17" \
    -DDEAL_II_WITH_MPI=ON \
    .. \
    && ninja -j $PROCS install \
    && cd ../ && rm -rf .git build

COPY postInstall /
RUN /postInstall