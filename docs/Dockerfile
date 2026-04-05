FROM debian:trixie-slim AS grace-donor
RUN apt-get -qq -y update && apt-get -qq -y install --no-install-recommends grace \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /grace-out/lib \
    && cp /usr/include/grace_np.h /grace-out/ \
    && cp "$(find /usr/lib -name libgrace_np.a -print -quit)" /grace-out/lib/

FROM debian:trixie-slim

ENV DEBIAN_FRONTEND=noninteractive

COPY --from=grace-donor /grace-out/grace_np.h /usr/include/grace_np.h
COPY --from=grace-donor /grace-out/lib/libgrace_np.a /usr/lib/libgrace_np.a

RUN apt-get -qq -y update && apt-get -qq -y install --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    git \
    ca-certificates \
    libgsl-dev \
    libarmadillo-dev \
    libeigen3-dev \
    libfftw3-dev \
    pkg-config \
    nlohmann-json3-dev \
    python3-dev \
    python3-matplotlib \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DDFT_BUILD_TESTS=OFF \
    -DDFT_BUILD_DOCS=ON \
    -DDFT_USE_MATPLOTLIB=ON \
    && cmake --build build --parallel

WORKDIR /app/build/docs
