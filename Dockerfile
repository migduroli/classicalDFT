FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq -y update && apt-get -qq -y install --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    libgsl-dev \
    libboost-serialization-dev \
    libboost-dev \
    libarmadillo-dev \
    libfftw3-dev \
    grace \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt ./
COPY cmake/ cmake/
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY examples/ examples/

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DDFT_BUILD_EXAMPLES=OFF \
    && cmake --build build --parallel "$(nproc)"

ENTRYPOINT ["ctest", "--test-dir", "build", "-V"]