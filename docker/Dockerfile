FROM dealii/dealii:v9.4.2-jammy

USER root 

COPY . /usr/src/non_matching_test_suite

RUN \
    cd /usr/src/non_matching_test_suite && mkdir build && cd build && \
    cmake -GNinja -DCMAKE_BUILD_TYP=DebugRelease .. && \
    ninja -j1 && \
    (ninja install || echo "Nothing to install") && \
    cd ../ && rm -rf build .git

RUN mv /usr/local/bin/clang-format /usr/bin/clang-format

USER $USER
