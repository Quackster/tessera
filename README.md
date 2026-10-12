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
| Public API | Done | `Engine`, `Backend`, `SpeculativeStrategy`. |
| GGUF loader | Done | v2/v3 headers, bounds-checked, memory-mapped (not copied). |
| MXFP4 loader | Done | Tensor map, weight upload. FP8 and MXFP4 GEMM verified. |
| Backends | Done | Vulkan and ROCm init, buffers, copy, sync. Batched host upload. |
| Parallelism | Done | One engine and model per GPU behind a pool with least loaded routing. Tensor, pipeline, context, expert, speculative split and hybrid modes match single device output. |
| Kernel launch | Done | Vulkan records a whole decode step into one command buffer. ROCm runs one blocking stream with pinned staging. |
| DFlash2 | Done | Draft forward built and running. 4.4 tokens/step on 27B MXFP4. |
| CLI | Done | Load, summary, streaming, sampling, serve, KV-cache flags, speculate, images, thinking flags, calibrate with an optional point limit. |
| Generation | Done | Greedy decode, chunked prefill, thinking budget, stop tokens, configurable KV cache. |
| KV cache | Done | fp32/fp16/int8/4-bit/FP8 E4M3 on both backends. Row-parallel quant kernel. |
| Sampling | Done | Seeded sampling with Qwen 27B defaults. |
| GEMM | Done | Q4_K to IQ, FP8/MXFP4, bf16, fp32. Tensor core paths for decode and verify. |
| Activation quant | Done | Per-token FP8 E4M3 quant-dequant. W4A8 opt-in. |
| Attention and RoPE | Done | GQA, tiled attention, all KV variants. Split key range for single token. |
| Weight upload | Done | Manifest to device buffers. One batched upload per load. |
| Decode loop | Done | Single-token vanilla and hybrid GGUF loops. |
| Hybrid SSM | Done | Load, kernels, both decode paths, MTP head done. |
| Speculative decode | Done | MTP and DFlash2 end to end. Output equals greedy. GGUF MTP reaches 31-37 tok/s on the 27B target (ROCm). DFlash2 about 60 vs 23 tok/s. |
| MXFP4 path | Done | Parse, kernels, HF config, tokenizer. Matches GGUF greedy output. |
| DFlash2 decode | Partial | Real draft end to end. 3.5 to 4.4 tokens/step. Two defects fixed. |
| Baseline pinning | Done | Fixed-seed fixtures identical on Vulkan and ROCm. |
| Tokenizer and chat | Done | Byte-level BPE, Jinja2-subset templates, stop tokens. |
| Serving | Partial | HTTP/1.1 server, sessions, web UI, streamed reasoning, tool calls. Capability harness (`tools/harness/probe_harness.py`) drives task steering, long-horizon context and session memory over the API on both backends. |
| Vision | Partial | CLIP encoder, merger, image injection. Prefill speed work remains. |
| Architecture modules | Done | One module per model family. |
| MoE, MLP, norms | Done | Norms and MLP kernels done. Ornith-1.5 MoE decodes on both backends (84 tok/s ROCm, 88 Vulkan). |
| Calibration | Done | Measures speed settings per machine and keeps the fastest. Covers split target and cap (ROCm), prefill chunk, attention split, and draft settings with a strategy attached. Reports and stores the memory-bound max context, and the example commands run at that context. `--calibrate-limit N` caps a run at N measurements. Writes JSON keyed by backend, device, model, context, KV type, strategy, and layout. |

See <a href="https://github.com/Quackster/tessera/blob/main/docs/PROGRESS.md">PROGRESS.md</a> for full status.

## Chat UI

Lightweight, dark-mode AI chat UI with a simple sidebar for API keys and conversation management. It supports streamed model responses, collapsible thinking output, retry controls, token limits, and rendering rich content like inline SVGs directly in the chat. Accessible via the IP:PORT serve command.

Here's an example of an SVG generation:

<img width="1796" height="1115" alt="image" src="https://github.com/user-attachments/assets/1b60b164-f462-467f-b696-812b27020de2" />

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
| MXFP4 safetensors | Done | Parses the map and packs the weights. config.json drives the decode config. |
| NVFP4 MoE | Done | GGUF mixture-of-experts (Ornith-1.5) loads and decodes on Vulkan and ROCm. |
| DFlash2 FP8 draft | Done | The draft loads and runs the block. Speculation accepts 4.4 tokens per step on the 27B MXFP4 target. |

Model weights live outside the repo. Each variant uses one flat directory. Model paths are runtime configuration. Code and tests never hard code model paths.

| Path | Contents | Use | Hugging Face |
| --- | --- | --- | --- |
| `~/models/Qwen3.8-27B-GGUF/` | `Qwen3.8-27B-UD-Q4_K_M.gguf`, `mmproj-BF16.gguf`, `MTP/mtp-Qwen3.8-27B-Q4_0.gguf`, `config.json` | Main GGUF target. | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| `~/models/Qwen3.8-27B-MXFP4-MTPFP8/` | MXFP4 weights with FP8 MTP | Main MXFP4 target. | [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) |
| `~/models/Qwen3.8-27B-DFlash2-FP8/` | DFlash2 draft checkpoint | Draft layout tests. | [tcclaviger/Qwen3.8-27B-DFlash2-FP8](https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8) |
| `~/models/Ornith-1.5-35B-A3B-GGUF/` | `Ornith-1.5-35B-Q4_K_M.gguf`, `mmproj-Ornith-1.5-35B-BF16.gguf` | MoE GGUF target. | [ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) |
| `~/models/Ornith-1.5-35B-A3B-MXFP4-GGUF/` | MoE MXFP4 GGUF weights | Not yet supported. | [tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF](https://huggingface.co/tsaipifong/Ornith-1.5-35B-A3B-MXFP4-GGUF) |
| `~/models/Ornith-1.5-35B-A3B-BF16/` | bf16 safetensors | Not yet supported. | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| `/home/alex/models/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/` | `IQ3_S` split, `IQ3_XXS` split, `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, RCO allocation files | (Planned, not working) | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) |

CLI examples:

```sh
# GGUF target, greedy decode.
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf

# GGUF target with MTP speculation. The MTP head drafts from the target, so it needs no draft path.
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf --speculate

# MXFP4 target with the DFlash2 draft checkpoint.
./cmake-build-vulkan/tessera-cli run --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8

# MXFP4 target and DFlash2 draft over the HTTP server.
./cmake-build-vulkan/tessera-cli serve --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 --port 8080
```

The CLI loads the model and uploads weights. It attaches the draft strategy when `--draft` or `--speculate` is set. With `--max-completion-tokens N` it runs N greedy decode steps and streams the text to stdout. `--speculate` drafts with the MTP head. `--draft` drafts with the DFlash2 checkpoint. The DFlash2 draft is tested against the MXFP4 target. `--mmproj` plus `--image` prepend image tokens.

## Calibration

Calibration measures the engine settings that depend on the machine. It keeps
the fastest value. It changes only speed. It never changes the produced
tokens. Run it once on each machine and model, before the first `run` or
`serve`.

The `calibrate` subcommand loads the model one time and sweeps the
candidate values for each setting. Without a limit it runs to
completion. It writes a JSON file and prints a report with the sweep
points, the chosen settings, and an example `run` command. The example
commands run at the memory-bound max context.

```sh
# Sweep on ROCm with the DFlash2 draft and write the result.
./cmake-build-rocm/tessera-cli calibrate \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --calibration ~/.config/tessera/calibration.json
```

The `calibrate` flags:

* `--model <path>` model to measure. Required.
* `--draft <dir>` DFlash2 draft checkpoint. It enables the draft block and
  draft context sweeps.
* `--speculate` MTP drafting from the target's own nextn head. It enables
  the draft block sweep.
* `--calibration <file>` file to read and write. The environment variable
  `TESSERA_CALIBRATION` supplies it when the flag is absent. One of the two is
  required.
* `--context <n>` context length in the hardware key. Default 4096.
* `--gpu <n>` device index. Default 0, the first device.
* `--kv-f16`, `--kv-q8`, `--kv-q4`, `--kv-fp8` KV cache type. Default int8
  (kv8).
* `--prompts <file>` fixed prompts, one per non-empty line. Default three
  fixed prompts.
* `--calibrate-limit <n>` measure at most n sweep points in total, in
  sweep order. Each point is one engine run. A cut setting keeps its
  fastest measured value. A skipped setting keeps its default.

The settings it sweeps are the same settings in the tables below:

* `mxfp4_split_target` and `mxfp4_split_cap` on ROCm only. On Vulkan the fp8
  kernel does not exist, so the command reports both as not applicable.
* `prefill_chunk_tokens` on both backends.
* `attention_split` on both backends, when the prompts reach the split
  threshold (1024 keys).
* `draft_tokens` and `draft_context` only when a draft is attached.

Apply a saved entry to `run` or `serve` with `--calibration <file>`, or set
`TESSERA_CALIBRATION`. The engine uses the saved value for the current machine
and model unless an explicit flag overrides it. An explicit flag always wins.
Only `calibrate` writes the file. `run` and `serve` only read it. A missing
file on `run` or `serve` is not an error. Create the parent directory before
you run `calibrate`.

```sh
# Serve with the calibrated values for this machine and model.
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --calibration ~/.config/tessera/calibration.json \
  --port 8080
```

The file is JSON. It holds one entry per hardware key. The key joins the
backend, the device name, the model name, the context, the KV type, and the
strategy. Each entry stores the chosen settings, the measured decode and
prefill rates, the memory-bound context for the KV type, and the date. A
machine, model, context, KV type, or strategy that differs from the key does
not match the entry. The path is runtime configuration and is never
hard-coded.

## Usage

Simple text prompt with the CLI:

```sh
./cmake-build-vulkan/tessera-cli run \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --prompt-text "Explain gravity in one sentence." \
  --max-completion-tokens 64
```

`--max-completion-tokens` is optional. Without it the run fills the remaining context. Output tokens stream to stdout.

The CLI applies the model chat template by default. It uses the raw text with `--no-chat`. String prompts need a model with a tokenizer. GGUF carries it in metadata. The MXFP4 loader parses `tokenizer.json`. String prompts fail on a model without a tokenizer, such as a bare DFlash2 draft directory.

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
options.max_completion_tokens = 64;  // Optional. Zero fills the remaining context.
options.prompt_tokens = *ids;
auto generated = (*engine)->Generate(**model, options);
if (!generated) return 1;
auto text = tokenizer->Decode(*generated);
```

HTTP API with the server:

```sh
# GGUF target.
./cmake-build-vulkan/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --port 8080

# MXFP4 target with the DFlash2 draft. MTP speculation uses --speculate instead.
./cmake-build-vulkan/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --port 8080

curl -X POST http://127.0.0.1:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain gravity in one sentence.", "max_completion_tokens": 64}'
curl http://127.0.0.1:8080/health
```

`max_completion_tokens` is optional. Without it the server fills the remaining context. Speculation is set when the server starts and applies to every served turn.

With `--api-key` set, requests need `Authorization: Bearer <key>`. `GET /health` and `GET /metrics` stay public.

## Command-line arguments

The first argument selects the mode: `run` decodes a prompt, `serve` starts the HTTP server, and `--list-gpus` lists the devices and exits. The table gives every flag, its scope, and its default.

| Argument | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `--model <path>` | both | required | A `.gguf` file or an MXFP4 model directory. |
| `--draft <dir>` | both | none | DFlash2 draft checkpoint directory. |
| `--context <n>` | both | 4096 | Maximum context length. |
| `--gpu <n>` | both | 0 | GPU index to use (0 is the first). |
| `--gpu-layers <n\|all>` | both | 0 | Layers kept in VRAM. The rest page from system RAM. 0 uploads everything. `help` prints counts without loading. |
| `--draft-block <n>` | both | 0 | Draft block tokens. 0 uses the checkpoint block. |
| `--prefill-chunk <n>` | both | 0 | Prefill tokens per forward. 0 selects 512. |
| `--split-target <n>` | both | 0 | fp8 MXFP4 split-K target (ROCm only). 0 uses 320. |
| `--split-cap <n>` | both | 0 | fp8 MXFP4 split-K cap (ROCm only). 0 uses 4. |
| `--attention-split <n>` | both | 0 | Flash-decoding chunks at long context. 0 uses 16. |
| `--draft-context <n>` | both | 0 | DFlash2 draft context rows. 0 keeps the checkpoint window. |
| `--quiet` | both | off | Suppress progress and info logs. |
| `--prompt-text <str>` | run | none | Text prompt. It is tokenized, so the model needs a tokenizer. |
| `--mmproj <path>` | run | none | Vision projector (mmproj) GGUF. |
| `--image <path>` | run | none | Binary PPM image to prepend as tokens. |
| `--no-chat` | run | off | Do not apply the chat template. |
| `--max-completion-tokens <n>` | run | 0 | Completion tokens. 0 fills the remaining context. |
| `--max-thinking-tokens <n>` | run | 0 | Think-block budget per turn. 0 leaves thinking unlimited. |
| `--enable-thinking` | run | off | Render the chat template with thinking enabled. |
| `--reasoning-effort <level>` | run | unset | Thinking difficulty for the chat template (for example `xhigh`). Ignored with `--no-chat`. |
| `--speculate` | both | off | MTP speculative decode from the target head. Needs no checkpoint. |
| `--sample` | run | off | Sample instead of greedy decode. |
| `--temperature <f>` | run | 0.6 | Sampling temperature. |
| `--top-p <f>` | run | 0.95 | Nucleus probability. |
| `--top-k <n>` | run | 20 | Keep the top n tokens. |
| `--min-p <f>` | run | 0.0 | Minimum probability. |
| `--presence-penalty <f>` | run | 0.0 | Presence penalty. |
| `--repetition-penalty <f>` | run | 1.0 | Repetition penalty. |
| `--seed <n>` | run | 0 | Sampling RNG seed. |
| `--kv-f16` | run | off | Store the KV cache in fp16. |
| `--kv-q8` | run | off | Store the KV cache in int8. |
| `--kv-q4` | run | off | Store the KV cache in 4-bit. |
| `--kv-fp8` | run | off | Store the KV cache in FP8 E4M3. |
| `--host <ip>` | serve | 127.0.0.1 | Bind address. |
| `--port <n>` | serve | 8080 | Bind port. |
| `--stats-interval <s>` | serve | 5 | Live tok/s cadence on streams and console. 0 disables. |
| `--no-auto-title` | serve | off | Keep the first user line as the chat title instead of asking the model. |
| `--api-key <k>` | serve | none | Accepted API key. Repeatable. |
| `--allow-origin <o>` | serve | none | CORS origin. Repeatable. `*` allows all origins. |
| `--calibration <file>` | run, serve, calibrate | env `TESSERA_CALIBRATION` | Calibration file. `run` and `serve` apply a saved entry for the current machine and model unless an explicit flag overrides it. `calibrate` reads and writes it. |
| `--prompts <file>` | calibrate | three fixed prompts | Fixed prompts, one per non-empty line, for the calibration sweep. |
| `--calibrate-limit <n>` | calibrate | none | Measure at most n sweep points in total. |
| **Parallelism** | | | The flags below spread work across the `--gpus` devices. |
| `--gpus <a,b>` | both | none | One model replica per listed GPU (for example `--gpus 0,1`). Overrides `--gpu`. |
| `--pipeline` | run | off | Split layers across the `--gpus` devices. |
| `--tp` | run | off | Split attention, FFN and MoE experts across the `--gpus` devices. |
| `--cp` | run | off | Split the key range across the `--gpus` devices. |
| `--spec-split` | run | off | Run the MTP draft head on the second `--gpus` device. Needs `--speculate`. |
| `--hybrid` | run | off | Data parallel groups of model parallel teams. |
| `--dp <n>` | run | 1 | Group count for `--hybrid`. |

The KV cache is fp32 when no `--kv-*` flag is set. The sampling defaults match the Qwen 3.8 27B defaults.

Environment variables:

* `TESSERA_API_KEY` supplies `--api-key` when the flag is absent.
* `TESSERA_MXFP4_W4A8=1` enables the served MXFP4 target's W4A8 linear activation.
* `TESSERA_CALIBRATION` supplies `--calibration` when the flag is absent.
* `TESSERA_MXFP4_SPLIT` and `TESSERA_MXFP4_SPLITCAP` override the split-K target and cap as a diagnostic (they win over the request value).
* `TESSERA_TILED_MIN_ROWS`, `TESSERA_TILED_MIN_COLS` and `TESSERA_PREFILL_ATTN_PAIRS` override the tiled-GEMM dispatch thresholds and the prefill attention work budget for measurement.

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
