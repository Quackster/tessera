# syntax=docker/dockerfile:1
#
# Tessera container image. One binary uses one compute backend, chosen at
# build time:
#
#   docker build -t tessera:rocm .                                  # AMD ROCm
#   docker build -t tessera:vulkan --build-arg TESSERA_BACKEND=vulkan .
#
# The entrypoint runs `serve` and reads every flag from TESSERA_* variables
# (see docker/entrypoint.sh). Model weights are mounted at run time, never
# baked into the image.

ARG TESSERA_BACKEND=rocm
ARG UBUNTU_VERSION=24.04
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.2
ARG GPU_ARCH=gfx1201
ARG CMAKE_VERSION=3.31.6

# ---------------------------------------------------------------------------
# CMake toolchain. Ubuntu 24.04 ships cmake 3.28 and the ROCm image pins its
# apt sources, so apt cannot upgrade it. The official static tarball depends
# on no repository and is copied into both builders.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS cmake-toolchain
ARG CMAKE_VERSION
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget \
    && rm -rf /var/lib/apt/lists/* \
    && wget -qO /tmp/cmake.tar.gz \
        "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz" \
    && mkdir /opt/cmake \
    && tar -xzf /tmp/cmake.tar.gz -C /opt/cmake --strip-components=1 \
    && rm /tmp/cmake.tar.gz

# ---------------------------------------------------------------------------
# Vulkan builder. The Vulkan loader, headers and glslangValidator compile
# the built-in SPIR-V kernels during configure.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS build-vulkan
ARG DEBIAN_FRONTEND=noninteractive
ARG BUILD_JOBS=2
ARG GLSLANG_VERSION=16.6.0
# gcc-13 and gcc-14 internal-compiler-error on this code; gcc-15, the
# toolchain the project builds with, does not. It comes from the Ubuntu
# toolchain test PPA. Build with few jobs: the project documents compiler
# crashes under heavy parallel loads.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common gpg ca-certificates wget \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential g++-15 ninja-build git \
        libvulkan-dev libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
# noble's glslang-tools (15.1.0) segfaults on cast_f32_f16.comp and
# conv1d.comp. Use the Khronos release the project is validated against.
RUN wget -qO /tmp/glslang.tar.gz \
        "https://github.com/KhronosGroup/glslang/releases/download/${GLSLANG_VERSION}/glslang-${GLSLANG_VERSION}-linux-x86_64-release.tar.gz" \
    && mkdir /opt/glslang \
    && tar -xzf /tmp/glslang.tar.gz -C /opt/glslang \
    && rm /tmp/glslang.tar.gz
COPY --from=cmake-toolchain /opt/cmake /opt/cmake
ENV PATH=/opt/glslang/bin:/opt/cmake/bin:$PATH
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja -DTESSERA_BACKEND=vulkan \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-15 \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
    && cmake --build /build -j"${BUILD_JOBS}"

# ---------------------------------------------------------------------------
# ROCm builder. The dev image carries hipcc, the HIP headers, rocm-cmake and
# the ROCm runtime .so set. No GPU is visible at build time, so the target
# ISA must be passed explicitly.
# ---------------------------------------------------------------------------
FROM ${ROCM_IMAGE} AS build-rocm
ARG DEBIAN_FRONTEND=noninteractive
ARG GPU_ARCH
ARG BUILD_JOBS=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common gpg ca-certificates \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        g++-15 ninja-build libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
COPY --from=cmake-toolchain /opt/cmake /opt/cmake
ENV PATH=/opt/cmake/bin:$PATH
WORKDIR /src
COPY . .
RUN cmake -S . -B /build -G Ninja -DTESSERA_BACKEND=rocm \
        -DHIP_ARCHITECTURES="${GPU_ARCH}" -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_COMPILER=g++-15 -DCMAKE_PREFIX_PATH=/opt/rocm \
        -DCMAKE_CXX_SCAN_FOR_MODULES=OFF \
    && cmake --build /build -j"${BUILD_JOBS}"

# ---------------------------------------------------------------------------
# Vulkan runtime. The loader and the RADV driver ship as Ubuntu packages.
# ---------------------------------------------------------------------------
FROM ubuntu:${UBUNTU_VERSION} AS runtime-vulkan
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libvulkan1 mesa-vulkan-drivers libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-vulkan /build/tessera-cli /usr/local/bin/tessera-cli

# ---------------------------------------------------------------------------
# ROCm runtime. The dev image already holds the ROCm runtime libraries, so
# the built binary runs against the same ROCm version it was built with.
# ---------------------------------------------------------------------------
FROM ${ROCM_IMAGE} AS runtime-rocm
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcurl4 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build-rocm /build/tessera-cli /usr/local/bin/tessera-cli

# ---------------------------------------------------------------------------
# Final image: the runtime for the selected backend plus the entrypoint.
# ---------------------------------------------------------------------------
FROM runtime-${TESSERA_BACKEND} AS runtime
COPY docker/entrypoint.sh /usr/local/bin/tessera-entrypoint
RUN chmod +x /usr/local/bin/tessera-entrypoint
ENV TESSERA_HOST=0.0.0.0 \
    TESSERA_PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/tessera-entrypoint"]
