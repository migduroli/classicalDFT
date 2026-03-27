FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq -y update && apt-get -qq -y install --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    ca-certificates \
    libgsl-dev \
    libarmadillo-dev \
    libfftw3-dev \
    pkg-config \
    nlohmann-json3-dev \
    grace \
    xvfb \
    xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
COPY examples/ examples/

ARG EXAMPLE=console

RUN cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DDFT_BUILD_TESTS=OFF \
      -DDFT_BUILD_EXAMPLES=ON \
      -DDFT_USE_GRACE=ON \
    && cmake --build build --parallel "$(nproc)"

ENTRYPOINT ["/bin/sh", "-c"]
CMD ["echo 'Usage: docker run <image> ./build/examples/<path>/example_<name>'"]
