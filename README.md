<div align="center">
<img width="620" height="256" alt="tessera_logo" src="https://github.com/user-attachments/assets/acb38f3d-9014-4566-a4bd-0bf4444b4aa4" />
</div>

</div>

<div align="center">

[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&color=ff69b4)](https://github.com/sponsors/Quackster) | [Website](https://oldskooler.org)

</div>

> Status: Heavy development. The API changes without notice. There is no stable text generation yet. The code is usable for development and for backend tests. It is not ready for production use.

Tessera is a C++20 LLM inference engine. It runs open weight models on GPU. It has two compute backends. Vulkan supports cross vendor GPUs. ROCm supports AMD GPUs. One binary uses one backend. The backend is set at configure time.

First class targets are Qwen 3.8 27B in GGUF and Qwen 3.8 in MXFP4, with and without DFlash2 speculative decoding.

## Features

| Area | State | Detail |
| --- | --- | --- |
| Public API | Done | `Engine` owns backend and models. `Backend` covers buffers and launch. `SpeculativeStrategy` covers draft methods. |
| GGUF loader | Done | Parses v2 and v3 headers. Checks bounds. Rejects bad input. |
| MXFP4 loader | Done | Parses tensor map. Uploads weights. FP8 and MXFP4 GEMM verified. |
| Backends | Done | Init and buffer alloc on Vulkan and ROCm. Copy and sync on both. |
| Kernel launch | Done | Binds buffers and 64 bit scalars. `fill` and `gemm_q4k` kernels verified by read back on both backends. |
| DFlash2 | Skeleton | Validates checkpoint layout. Draft logic is in work. |
| CLI | Partial | Loads model, prints summary, generates tokens (`--tokens`), picks the GPU (`--gpu`) and lists them (`--list-gpus`). Text prompts need a tokenizer. |
| Generation | Done | `Engine::Generate` greedy decode on the non-speculative path. Runtime options: `--context`, `--draft-block`. |
| GEMM | Done | Generic GEMM with Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ, FP8/MXFP4 dequant, plain fp32 and bf16. Host reference check. Per backend tolerance. |
| Attention and RoPE | Done | GQA path driven by model data. RoPE kernel verified by read back on both backends. |
| Weight upload | Done | Manifest to device buffers. `Model::Weights` holds them. |
| Decode loop | Done | Single token loop on vanilla and hybrid (gated attention + gated-delta linear) GGUF. 27B generates. |
| Hybrid SSM | Partial | Definition, load, kernels and both decode paths done. MTP head done. |
| Speculative decode | Partial | MTP draft and greedy verifier run end to end (`--speculate`); output equals greedy. Batched scoring and the DFlash2 draft are in work. |
| MXFP4 path | Done | Tensor map parsing. FP8 and MXFP4 kernels. MTP draft weights mapped. |
| DFlash2 decode | Todo | Grouped causal convolutions. Low rank selector. Verify loop. |
| Baseline pinning | Done | Fixed-seed hybrid fixtures pin exact greedy sequences; identical on Vulkan and ROCm. |
| MoE, MLP, norms | Partial | RMSNorm and sigmoid-gate kernels done. MLP and embedding kernels in work. MoE is planned for Ornith-1.5-35B-A3B. No per model branches. |

See <a href="https://github.com/Quackster/tessera/blob/main/docs/PROGRESS.md">PROGRESS.md</a> for full status.

## Architecture

Core code is backend agnostic. All device work goes through the `Backend` interface. Only `src/backends/vulkan/` includes Vulkan headers. Only `src/backends/rocm/` includes HIP and ROCm headers. No vendor type crosses the backend boundary.

Models are data. A new architecture arrives as config plus weights. It drives generic kernels for GEMM, attention, MLP, MoE, norms, RoPE, activations, and embeddings. Generic code has no per model branches.

Speculative decoding plugs in behind the strategy interface. It is off by default. DFlash2 is the current strategy. The non speculative path is the reference baseline. It stays stable within per backend tolerance.

Layout:

* `include/tessera/` holds the public API.
* `src/core/` holds the engine, the model, and the weight loaders.
* `src/backends/vulkan/` holds the Vulkan backend.
* `src/backends/rocm/` holds the ROCm backend.
* `src/spec/` holds speculative decoding strategies.
* `tools/cli/` holds the thin CLI.
* `tests/` holds the single GoogleTest binary.
* `docs/PROGRESS.md` tracks progress.

See `AGENTS.md` for architecture rules and for hard rules.

## Model support

| Format | State | Detail |
| --- | --- | --- |
| GGUF | Done | Parses v2 and v3. Supports Q4_K, Q3_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS and IQ3_S. Checks bounds. |
| MXFP4 safetensors | Partial | Parses map and uploads weights. GEMM verified. No decode config yet. |
| NVFP4 MoE | Todo | Planned: MoE model support (Ornith-1.5-35B-A3B). Needs MoE routing and NVFP4 kernels. |
| DFlash2 FP8 draft | Skeleton | Validates layout only. Decode logic is in work. |

Model weights live outside the repo. Each variant uses one flat directory. Model paths are runtime configuration. Code and tests never hard code model paths.

| Path | Contents | Use | Hugging Face |
| --- | --- | --- | --- |
| `~/models/Qwen3.8-27B-GGUF/` | `Qwen3.8-27B-UD-Q4_K_M.gguf`, `mmproj-BF16.gguf`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, `config.json` | Main GGUF target. | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| `~/models/Qwen3.8-27B-MXFP4-MTPFP8/` | MXFP4 weights with FP8 MTP | Main MXFP4 target. | [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) |
| `~/models/Qwen3.8-27B-DFlash2-FP8/` | DFlash2 draft checkpoint | Draft layout tests. | [tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) |
| `~/models/Ornith-1.5-35B-A3B-GGUF/` | MoE GGUF weights | To do: MoE support. | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| `~/models/Ornith-1.5-35B-A3B-NVFP4/` | NVFP4 MoE weights | To do: MoE support. | [Ornith-ai/Ornith-1.5-35B-A3B-NVFP4](https://huggingface.co/Ornith-ai/Ornith-1.5-35B-A3B-NVFP4) |
| `/home/alex/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` | `IQ3_S` split, `IQ3_XXS` split, `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, RCO allocation files | (Planned, not working) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

CLI example:

```sh
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf --draft ~/models/Qwen3.8-27B-DFlash2-FP8
```

The CLI loads the model and uploads weights. It attaches the draft strategy when `--draft` is set. It prints tensor count and total elements. With `--tokens N` it runs N greedy decode steps from `--prompt` and prints the token ids.

## Build and test

Requirements:

* CMake 3.30 or later.
* A C++ compiler with C++23 support. The code stays C++20 style.
* For Vulkan: Vulkan SDK and `glslangValidator`.
* For ROCm: ROCm toolchain, `rocminfo`, and a supported AMD GPU.

Configure and build:

```sh
cmake -B cmake-build-vulkan -DTESSERA_BACKEND=vulkan
cmake --build cmake-build-vulkan -j
```

```sh
cmake -B cmake-build-rocm -DTESSERA_BACKEND=rocm
cmake --build cmake-build-rocm -j
```

Run tests:

```sh
ctest --test-dir cmake-build-vulkan --output-on-failure
ctest --test-dir cmake-build-rocm --output-on-failure
```

All runtime coverage lives in one test binary. Add test functions to `tests/*.cpp`. Do not add new CMake test executables. Device tests skip cleanly when no device is present. Numerical checks use per backend tolerance.

Verified devices:

| Device | Architecture | Memory | Status |
| --- | --- | --- | --- |
| AMD Radeon RX 7900 XTX | RDNA 3, Navi 31, gfx1100, 96 compute units | 24 GB GDDR6 | Verified on Vulkan (RADV NAVI31) and ROCm (gfx1100). |
| AMD Radeon AI PRO R9700 | RDNA 4, gfx1201, 64 compute units | 32 GB GDDR6 | Verified on Vulkan (RADV) and ROCm (gfx1201). |

State the backend used when reporting results.

## Current progress

See <a href="https://github.com/Quackster/tessera/blob/main/docs/PROGRESS.md">PROGRESS.md</a>.

## License

Tessera uses the Apache License 2.0. See `LICENSE`.
