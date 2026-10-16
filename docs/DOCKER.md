# Docker

This file is the administrator guide for the tessera container image. It
explains the image layout, the build arguments, the publish steps, and the
known build constraints. Use it to build, tag, and publish the image.

## Image layout

One image uses one compute backend. The backend is set at build time with
the `TESSERA_BACKEND` build argument. The Dockerfile has these stages:

| Stage | Base | Purpose |
| --- | --- | --- |
| `cmake-toolchain` | `ubuntu:24.04` | Fetches CMake 3.31.6 from the official tarball. |
| `build-vulkan` | `ubuntu:24.04` | Builds `tessera-cli` for Vulkan. |
| `build-rocm` | `rocm/dev-ubuntu-24.04:7.2` | Builds `tessera-cli` for ROCm. |
| `runtime-vulkan` | `ubuntu:24.04` | Vulkan loader, RADV driver, and the binary. |
| `runtime-rocm` | `rocm/dev-ubuntu-24.04:7.2` | ROCm runtime libraries and the binary. |
| `runtime` | `runtime-<backend>` | Adds the entrypoint and the defaults. |

The `runtime` stage selects its base from `TESSERA_BACKEND`. Only the stages
for the selected backend build under BuildKit.

The image holds the binary and the entrypoint only. Model weights stay
outside the image. Mount them at run time.

## Build arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `TESSERA_BACKEND` | `rocm` | Compute backend: `rocm` or `vulkan`. |
| `UBUNTU_VERSION` | `24.04` | Ubuntu base for the CMake and Vulkan stages. |
| `ROCM_IMAGE` | `rocm/dev-ubuntu-24.04:7.2` | ROCm base image. Match the host driver. |
| `GPU_ARCH` | `gfx1201` | HIP target ISA. No GPU is visible at build time. |
| `CMAKE_VERSION` | `3.31.6` | CMake release to fetch. The project needs 3.30 or later. |
| `GLSLANG_VERSION` | `16.6.0` | Khronos glslang release for the Vulkan builder. |
| `BUILD_JOBS` | `2` | Compile jobs. Keep this low. See the constraints below. |

Set the GPU ISA to the target card. The R9700 is `gfx1201` and the 7900 XTX
is `gfx1100`.

## Build

The classic Docker builder builds every stage in order. It builds the
unused backend stage too, and that stage can fail. Use `docker buildx`, or
set `DOCKER_BUILDKIT=1`, so only the selected backend builds.

Build the ROCm image:

```sh
docker buildx build -t tessera:rocm --load .
```

Build the Vulkan image:

```sh
docker buildx build --build-arg TESSERA_BACKEND=vulkan -t tessera:vulkan --load .
```

Build one image at a time. Each build is a full compile and holds host
memory for its whole length.

On a large machine, raise the jobs:

```sh
docker buildx build --build-arg BUILD_JOBS=8 -t tessera:rocm --load .
```

## Publish to GHCR

The images live in the GitHub Container Registry under
`ghcr.io/quackster/tessera`. The path must use lower case.

Tags:

| Tag | Image |
| --- | --- |
| `rocm` | AMD ROCm backend. |
| `vulkan` | Vulkan backend. |
| `latest` | The ROCm image. The default backend. |

Log in. The token needs the `write:packages` scope.

```sh
echo "$CR_PAT" | docker login ghcr.io -u Quackster --password-stdin
```

Build and tag the ROCm image, then the Vulkan image:

```sh
docker buildx build \
  -t ghcr.io/quackster/tessera:rocm \
  -t ghcr.io/quackster/tessera:latest \
  --load .

docker buildx build \
  --build-arg TESSERA_BACKEND=vulkan \
  -t ghcr.io/quackster/tessera:vulkan \
  --load .
```

Push every tag:

```sh
docker push ghcr.io/quackster/tessera:rocm
docker push ghcr.io/quackster/tessera:latest
docker push ghcr.io/quackster/tessera:vulkan
```

Replace `--load .` with `--push .` to build and push in one command.

The first push creates a private package. Set it to public in the package
settings at `github.com/users/Quackster/packages/container/tessera/settings`.

If `latest` must mean Vulkan instead, move the `-t ...:latest` line to the
Vulkan build.

## Run

The image runs `serve`. Pass the GPUs through and mount the models
directory read only.

```sh
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --ipc=host --security-opt seccomp=unconfined \
  -v ~/models:/models:ro -p 8080:8080 \
  -e TESSERA_MODEL=/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  -e TESSERA_GPUS=0 \
  -e TESSERA_SPECULATE=1 -e TESSERA_DRAFT_BLOCK=2 \
  -e TESSERA_CONTEXT=262144 -e TESSERA_KV=q8 \
  tessera:rocm
```

A Vulkan container needs only `--device=/dev/dri`. It does not need
`/dev/kfd`.

## Environment variables

The entrypoint maps each variable to one `serve` flag. An unset variable
keeps the CLI default. A boolean variable is true for `1`, `true`, `yes`,
or `on`.

| Variable | Flag | Default |
| --- | --- | --- |
| `TESSERA_MODEL` | `--model` | required for serve |
| `TESSERA_DRAFT` | `--draft` | none |
| `TESSERA_MTP` | `--mtp` | none |
| `TESSERA_CALIBRATION` | `--calibration` | none |
| `TESSERA_GPUS` | `--gpus` | 0 |
| `TESSERA_SPECULATE` | `--speculate` | off |
| `TESSERA_PIPELINE` | `--pipeline` | off |
| `TESSERA_PIPELINE_SPLIT` | `--pipeline-split` | none |
| `TESSERA_TP` | `--tp` | off |
| `TESSERA_CP` | `--cp` | off |
| `TESSERA_SPEC_SPLIT` | `--spec-split` | off |
| `TESSERA_HYBRID` | `--hybrid` | off |
| `TESSERA_DP` | `--dp` | 1 |
| `TESSERA_CONTEXT` | `--context` | 32768 |
| `TESSERA_GPU_LAYERS` | `--gpu-layers` | 0 |
| `TESSERA_DRAFT_BLOCK` | `--draft-block` | 0 |
| `TESSERA_DRAFT_CONTEXT` | `--draft-context` | 0 |
| `TESSERA_PREFILL_CHUNK` | `--prefill-chunk` | 0 |
| `TESSERA_ATTENTION_SPLIT` | `--attention-split` | 0 |
| `TESSERA_SPLIT_TARGET` | `--split-target` | 0 |
| `TESSERA_SPLIT_CAP` | `--split-cap` | 0 |
| `TESSERA_KV` | `--kv-f16`, `--kv-q8`, `--kv-q4`, `--kv-fp8` | fp32 |
| `TESSERA_KV_OFFLOAD` | `--kv-offload` | off |
| `TESSERA_HOST` | `--host` | 0.0.0.0 |
| `TESSERA_PORT` | `--port` | 8080 |
| `TESSERA_STATS_INTERVAL` | `--stats-interval` | 5 |
| `TESSERA_API_KEY` | `--api-key` | none |
| `TESSERA_ALLOW_ORIGIN` | `--allow-origin` | none |
| `TESSERA_NO_AUTO_TITLE` | `--no-auto-title` | off |
| `TESSERA_THINKING` | `--enable-thinking` or `--no-thinking` | on |
| `TESSERA_REASONING_EFFORT` | `--reasoning-effort` | unset |
| `TESSERA_MAX_COMPLETION_TOKENS` | `--max-completion-tokens` | 0 |
| `TESSERA_MAX_THINKING_TOKENS` | `--max-thinking-tokens` | 0 |
| `TESSERA_QUIET` | `--quiet` | off |
| `TESSERA_VERBOSE` | `--verbose` | off |
| `TESSERA_ARGS` | raw extra flags | none |

`TESSERA_KV` accepts only `f16`, `q8`, `q4`, or `fp8`. Any other value stops
the container with exit code 2. `TESSERA_ALLOW_ORIGIN` accepts a comma or
space separated list. `TESSERA_ARGS` splits on whitespace and appends the
flags verbatim. `TESSERA_COMMAND` selects the subcommand and defaults to
`serve`. `TESSERA_CLI` overrides the binary path.

The entrypoint never prints the argument list, because it can hold
`TESSERA_API_KEY`.

## Build constraints

These constraints come from the base images and the compiler. Do not remove
the workarounds without a test.

- **CMake version.** Ubuntu 24.04 ships CMake 3.28. The project needs 3.30
  or later. The ROCm image pins its apt sources, so apt cannot upgrade
  CMake. The `cmake-toolchain` stage fetches the official static tarball.
- **Compiler.** gcc-13 and gcc-14 internal-compiler-error on this code, in
  `std::expected` and `std::move_only_function`. gcc-15 works. It comes from
  the `ubuntu-toolchain-r/test` PPA.
- **Module scanning.** CMake enables C++ module scanning for gcc-14 and
  later. That adds `-fmodules-ts`, which crashes the compiler. The code uses
  no modules, so the build sets `CMAKE_CXX_SCAN_FOR_MODULES=OFF`.
- **glslang.** The noble `glslang-tools` package is 15.1.0. It segfaults on
  `cast_f32_f16.comp` and `conv1d.comp`. The Vulkan builder installs the
  Khronos glslang 16.6.0 release.
- **ROCm CMake package.** The ROCm package is at
  `/opt/rocm/share/rocmcmakebuildtools/cmake`. CMake does not search
  `/opt/rocm` by default. The ROCm build sets
  `-DCMAKE_PREFIX_PATH=/opt/rocm`.
- **Parallel builds.** The compiler crashes on this code base under heavy
  parallel load. `BUILD_JOBS` defaults to 2. Raise it only with care.
- **GPU ISA.** No GPU is visible at build time. Pass `GPU_ARCH`.
- **Platform.** ROCm has no arm64 build. The image is amd64 only.

## Maintenance

To update a version, change the build argument default and rebuild both
backends:

1. `ROCM_IMAGE` when the host driver or ROCm changes.
2. `CMAKE_VERSION` when a newer CMake is needed.
3. `GLSLANG_VERSION` when a newer glslang is needed.
4. `GPU_ARCH` when the target card changes.

After a change, build the ROCm image and the Vulkan image, run a serve on
each backend, and push new tags. State the backend used in the report.

Keep the README Docker section and this file in step. The README holds the
user-facing commands. This file holds the operator details.
