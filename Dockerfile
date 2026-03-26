FROM debian:bookworm-slim AS grace-donor
RUN apt-get -qq -y update && apt-get -qq -y install --no-install-recommends grace \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /grace-out/lib \
    && cp /usr/include/grace_np.h /grace-out/ \
    && cp "$(find /usr/lib -name libgrace_np.a -print -quit)" /grace-out/lib/

FROM debian:bookworm-slim

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
    nlohmann-json3-dev \
    lcov \
    clang-format \
    clang-tidy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app