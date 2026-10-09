<div align="center">
<img width="620" height="256" alt="tessera_logo" src="https://github.com/user-attachments/assets/acb38f3d-9014-4566-a4bd-0bf4444b4aa4" />
</div>

</div>

<div align="center">

[![Sponsor](https://img.shields.io/static/v1?label=Sponsor&message=%E2%9D%A4&color=ff69b4)](https://github.com/sponsors/Quackster) | [Website](https://oldskooler.org) | [Twitter](https://twitter.com/monitormatti)

</div>

> Status: Heavy development. The API changes without notice. Greedy text generation is stable on the 27B targets. DFlash2 and MTP speculation run end to end. The code is usable for development and for backend tests. It is not ready for production use.

Tessera is a C++20 LLM inference engine. It runs open weight models on GPU. It has two compute backends. Vulkan supports cross vendor GPUs. ROCm supports AMD GPUs. One binary uses one backend. The backend is set at configure time.

First class targets are Qwen 3.8 27B in GGUF and Qwen 3.8 in MXFP4, with and without DFlash2 speculative decoding.

**Why AMD (ROCm) and Vulkan only?**
The project author tests with a 7900 XTX and two R9700 cards. There is no recent NVIDIA GPU in the lab. The last NVIDIA GPU used here was a 1080 Ti. Backend support follows hardware available for testing.

## Features

| Area | State | Detail |
| --- | --- | --- |
| Public API | Done | `Engine` owns backend and models. `Backend` covers buffers and launch. `SpeculativeStrategy` covers draft methods. |
| GGUF loader | Done | Parses v2 and v3 headers. Checks bounds. Rejects bad input. |
| MXFP4 loader | Done | Parses tensor map. Uploads weights. FP8 and MXFP4 GEMM verified. |
| Backends | Done | Init and buffer alloc on Vulkan and ROCm. Copy and sync on both. |
| Kernel launch | Done | Binds buffers and 64 bit scalars. `fill` and `gemm_q4k` kernels verified by read back on both backends. ROCm launches are asynchronous; Vulkan pipelines launches through a four-slot command-buffer and fence ring, so the host does not wait on a fence after every kernel. |
| DFlash2 | Partial | Draft forward built: grouped dynamic convolution, sliding attention, candidate selector and target-hidden fusion. The real draft loads and runs the block. The 27B MXFP4 target accepts 4.4 tokens per step. Output equals greedy. The draft step costs more than greedy, so speed work remains. |
| CLI | Partial | Loads a model, prints a tensor summary, streams generated text to stdout (`--tokens`), picks the GPU (`--gpu`) and lists them (`--list-gpus`). It serves HTTP (`serve`), samples (`--sample` and parameter flags), selects the KV cache (`--kv-f16`, `--kv-q8`, `--kv-q4`, `--kv-fp8`), speculates (`--speculate`, `--draft`) and encodes images (`--mmproj`, `--image`). Text prompts work for GGUF and MXFP4. |
| Generation | Done | `Engine::Generate` greedy decode on the non-speculative path; the prompt prefills in chunk-sized batched forwards (512 by default, `--prefill-chunk`), so a 220k-token prompt on the 8-bit KV cache (`--context 220000 --kv-q8`) prefills in bounded memory. Each forward is also clamped to the attention work budget, so a long prompt cannot wedge the device with one oversized launch. A prompt past the context is rejected before any device work. A zero `max_tokens` fills the remaining context. Generation stops at the model's declared stop tokens (GGUF `tokenizer.ggml.eos_token_id`, HuggingFace `eos_token_id`) and does not emit them; callers add stops through `GenerateOptions::stop_tokens`. Runtime options: `--context`, `--tokens`, `--prefill-chunk`, `--draft-block`, fp16 (`--kv-f16`), int8 (`--kv-q8`), 4-bit (`--kv-q4`) or FP8 (`--kv-fp8`) KV cache. |
| KV cache | Done | fp32 (default), fp16 (`--kv-f16`), int8 (`--kv-q8`), 4-bit (`--kv-q4`) and FP8 E4M3 (`--kv-fp8`, the served target's cache quant) full-attention storage on both backends. |
| Sampling | Done | Optional seeded sampling with the Qwen 3.8 27B defaults (temperature, top_p, top_k, min_p, presence/repetition penalties); `--sample` and parameter flags. |
| GEMM | Done | Generic GEMM with Q4_K, Q5_K, Q6_K, Q3_K, Q8_0, IQ, FP8/MXFP4 dequant, block-scaled FP8, plain fp32 and bf16. Tiled batched kernels for the hot formats (four partial sums per thread) and warp-per-output decode GEMVs. Small verify batches use a warp-per-column multi-row MXFP4 kernel. The vocab-width head reads its weights once with the tiled kernel. Host reference check. Per backend tolerance. |
| Activation quant | Done | Per-token FP8 E4M3 quantize-dequantize (`quantize_fp8`) on Vulkan and ROCm, the W4A8 activation contract. The DFlash2 drafter input uses it. The served target's W4A8 linear activation is opt-in (`TESSERA_MXFP4_W4A8=1`). |
| Attention and RoPE | Done | GQA path driven by model data; tiled O(n*d) attention (one workgroup per query/head, online softmax), including the int8, 4-bit and FP8 KV variants. RoPE kernel verified by read back on both backends. |
| Weight upload | Done | Manifest to device buffers. `Model::Weights` holds them. |
| Decode loop | Done | Single token loop on vanilla and hybrid (gated attention + gated-delta linear) GGUF. The 27B hybrid path generates coherent text. |
| Hybrid SSM | Partial | Definition, load, kernels and both decode paths done. MTP head done. The causal conv1d runs on the device (`conv1d_state`). Embedding gather runs on the device. Some linear-path glue still runs on the host. |
| Speculative decode | Partial | MTP (`--speculate`) and DFlash2 (`--draft`) run end to end; output equals greedy. Multi-token drafts score in one batched target forward. MTP accepts about half its drafts with a default chain of one. DFlash2 accepts 4.4 tokens per step but still decodes slower than greedy. |
| MXFP4 path | Done | Tensor map parsing. FP8 and MXFP4 kernels. HuggingFace config, weight name map, value-head reorder and tokenizer. The Qwen 3.8 27B MXFP4 target loads, tokenizes and decodes end to end. Greedy output matches the GGUF reference. The FP8 MTP head loads. W4A8 activation quant is opt-in. |
| DFlash2 decode | Partial | DFlash2 speculation runs end to end with the real draft (target hidden capture, mask block, selector, accept/reject). The draft conditions on a device-resident per-layer context K/V cache built from committed target positions. The concatenated target hidden is quantized per token to FP8 E4M3 before `fc`. Output equals greedy. On the 27B MXFP4 target the draft accepts 4.4 tokens per step, above the served reference (2.7 to 2.85), but decode is slower than greedy. Two defects were fixed (the candidate extraction ranked only the first 16 token ids on rows after the first, and the selector added the predecessor unary logit instead of the successor). |
| Baseline pinning | Done | Fixed-seed hybrid fixtures pin exact greedy sequences; identical on Vulkan and ROCm. |
| Tokenizer and chat | Done | Byte-level BPE tokenizer from GGUF metadata or HuggingFace `tokenizer.json`. Jinja2-subset chat template renderer. Generation stops at the declared stop tokens and never emits them. |
| Serving | Partial | Blocking HTTP/1.1 server with buffered and chunked (SSE) responses, API-key auth and CORS. First endpoint set done. `/v1/chat/completions` supports OpenAI-form tool calls (template-rendered tools, `tool_calls` answers, `tool_choice`). Thread pool, queueing and more surfaces deferred. |
| Vision | Partial | CLIP encoder plus merger, PPM load and resize, image-embedding injection, CLI wiring with auto `<|image_pad|>` detection. Image prefill speed work remains. |
| Architecture modules | Done | One `Architecture` module per model family (`src/models/qwen3_5/`). The hybrid trunk and state live behind it. Core names no model. |
| MoE, MLP, norms | Todo | RMSNorm, sigmoid-gate, add, silu_mul and the f32/bf16/Q4_K embedding gather kernels are done. The gated MLP runs as gemm plus silu_mul plus gemm on the device. No MoE kernels exist yet. MoE is planned for Ornith-1.5-35B-A3B. No per model branches. |

See <a href="https://github.com/Quackster/tessera/blob/main/docs/PROGRESS.md">PROGRESS.md</a> for full status.

## Architecture

Core code is backend agnostic. All device work goes through the `Backend` interface. Only `src/backends/vulkan/` includes Vulkan headers. Only `src/backends/rocm/` includes HIP and ROCm headers. No vendor type crosses the backend boundary.

Models are data plus one module per architecture. Each architecture owns its config parse, weight map and layer assembly behind the `Architecture` interface (today `src/models/qwen3_5/`). It drives generic kernels for GEMM, attention, MLP, MoE, norms, RoPE, activations, and embeddings. Generic code has no per model branches.

Speculative decoding plugs in behind the strategy interface. It is off by default. DFlash2 is the current strategy. The non speculative path is the reference baseline. It stays stable within per backend tolerance.

Layout:

* `include/tessera/` holds the public API.
* `src/core/` holds the engine, the model, and the weight loaders.
* `src/models/` holds one architecture module per model family.
* `src/backends/vulkan/` holds the Vulkan backend.
* `src/backends/rocm/` holds the ROCm backend.
* `src/spec/` holds speculative decoding strategies.
* `src/serve/` holds the HTTP layer.
* `tools/cli/` holds the thin CLI.
* `tests/` holds the single GoogleTest binary.
* `docs/PROGRESS.md` tracks progress.

See `AGENTS.md` for architecture rules and for hard rules.

## Model support

| Format | State | Detail |
| --- | --- | --- |
| GGUF | Done | Parses v2 and v3. Supports Q4_K, Q3_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS and IQ3_S. Checks bounds. |
| MXFP4 safetensors | Done | Parses the map, maps names, converts the value layout and packs the MXFP4 weights; config.json drives the decode config. |
| NVFP4 MoE | Todo | Planned: MoE model support (Ornith-1.5-35B-A3B). Needs MoE routing and NVFP4 kernels. |
| DFlash2 FP8 draft | Partial | The draft loads and runs the block. Speculation accepts 4.4 tokens per step on the 27B MXFP4 target. Step cost work remains. |

Model weights live outside the repo. Each variant uses one flat directory. Model paths are runtime configuration. Code and tests never hard code model paths.

| Path | Contents | Use | Hugging Face |
| --- | --- | --- | --- |
| `~/models/Qwen3.8-27B-GGUF/` | `Qwen3.8-27B-UD-Q4_K_M.gguf`, `mmproj-BF16.gguf`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, `config.json` | Main GGUF target. | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| `~/models/Qwen3.8-27B-MXFP4-MTPFP8/` | MXFP4 weights with FP8 MTP | Main MXFP4 target. | [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) |
| `~/models/Qwen3.8-27B-DFlash2-FP8/` | DFlash2 draft checkpoint | Draft layout tests. | [tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) |
| `~/models/Ornith-1.5-35B-A3B-GGUF/` | MoE GGUF weights | To do: MoE support. | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| `~/models/Ornith-1.5-35B-A3B-MXFP4-GGUF/` | MoE MXFP4 GGUF weights | To do: MoE support. | [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) |
| `~/models/Ornith-1.5-35B-A3B/` | MoE weights | To do: MoE support. | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| `/home/alex/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` | `IQ3_S` split, `IQ3_XXS` split, `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, RCO allocation files | (Planned, not working) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

CLI example:

```sh
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf --draft ~/models/Qwen3.8-27B-DFlash2-FP8
```

The CLI loads the model and uploads weights. It attaches the draft strategy when `--draft` is set. It prints tensor count, total elements and device bytes. With `--tokens N` it runs N greedy decode steps from `--prompt-text` and streams the decoded text to stdout as each token is generated. `--speculate` drafts with the MTP head. `--draft` drafts with the DFlash2 checkpoint. `--mmproj` plus `--image` prepend image tokens. The image token id is detected from `<|image_pad|>`; no flag sets it.

## Usage

Simple text prompt with the CLI:

```sh
./cmake-build-vulkan/tessera-cli run \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --prompt-text "Explain gravity in one sentence." \
  --tokens 64
```

`--tokens` is optional. Without it the run fills the remaining context. Output tokens stream to stdout during the run, so the first token shows at once instead of after the full run.

The CLI applies the model chat template by default. It uses the raw text with `--no-chat`. String prompts need a model with a tokenizer. Both formats provide one: GGUF carries it in metadata, and the MXFP4 loader parses the HuggingFace `tokenizer.json` (and the chat template from `tokenizer_config.json` or `chat_template.jinja`). The CLI and the HTTP server accept string prompts for either format. String prompts still fail on a model without a tokenizer, such as a bare DFlash2 draft directory.

C++ API with a text prompt:

```cpp
#include "tessera/engine.hpp"

auto engine = tessera::Engine::Create({});
if (!engine) return 1;
auto model = (*engine)->LoadModel(
    tessera::ModelOptions{"~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"});
if (!model) return 1;

const tessera::Tokenizer* tokenizer = (*model)->GetTokenizer();
std::string text = "Explain gravity in one sentence.";
if (auto rendered = (*model)->ChatPrompt(text)) text = *rendered;
auto ids = tokenizer->Encode(text);

tessera::GenerateOptions options;
options.max_tokens = 64;  // Optional. Zero fills the remaining context.
options.prompt_tokens = *ids;
auto generated = (*engine)->Generate(**model, options);
if (!generated) return 1;
auto text = tokenizer->Decode(*generated);
```

HTTP API with the server:

```sh
./cmake-build-vulkan/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --port 8080
curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gravity in one sentence.", "max_tokens": 64}'
curl http://127.0.0.1:8080/health
```

`max_tokens` is optional. Without it the server fills the remaining context.

With `--api-key` set, requests need `Authorization: Bearer <key>`. `GET /health` and `GET /metrics` stay public.

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

CMake defaults to a Release build. Measure speed only on Release. Build serially. The gcc-15 toolchain crashes under parallel builds.

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
