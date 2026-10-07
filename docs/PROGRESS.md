# Progress

This file tracks tessera development. Update it as work progresses.
After each milestone, update "Done" and refresh "Next" (see AGENTS.md,
Working Principles).

## Current status

The boilerplate is complete and passes on both backends.
`ctest` passes 129/129 on both builds.
Both builds were verified on AMD Radeon AI PRO R9700 (vulkan
through RADV GFX1201, rocm through the system ROCm).

- Project layout and CMake. The backend is selected at configure time
  (`vulkan` or `rocm`).
- Public API: `Engine`, `Model`, `Backend`/`Buffer`,
  `SpeculativeStrategy`, `Diagnostics`.
- GGUF v2/v3 parser. It parses the header and the tensor manifest.
  It checks all bounds. Small metadata arrays stay retained.
  Bulk arrays stay dropped.
- Safetensors layout checks for MXFP4 model directories. A bounded
  schema-strict JSON reader parses the tensor map with no third party
  dependency. MXFP4 blobs pair with their E8M0 scales by name. MTP FP8
  weights map natively.
- Vulkan and ROCm backends: device init, buffer alloc/free, H2D/D2H
  copy, synchronize. `CopyD2HAt` reads byte slices for embedding rows.
- Kernel launch plumbing on both backends. `LaunchKernel` binds buffers
  and 64-bit scalars in parameter order and waits on a fence (vulkan) or
  the hip runtime (rocm). The built-in "fill" kernel runs end to end
  and is verified by a read-back test on both devices.
- Generic GEMM kernels with dequantization and fp32 accumulation.
  Q4_K, Q3_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS and IQ3_S are verified on both
  devices against host references with per backend tolerance. Exact
  codec unit tests cover the dequant math. FP8 (per-row scaled E4M3)
  and MXFP4 (32-element E8M0 blocks) kernels are verified the same way.
  The OCP E4M3 top bin needs care (only mantissa-all-ones is NaN,
  max 448).
- Attention (GQA) and RoPE kernels driven by the model definition.
  `Model::Attention` reads heads, kv groups, head dim and the RoPE
  range and base from GGUF metadata. The generic "rope" and
  "attention" built-ins take all dims as launch scalars and are
  verified against host references on both devices.
- Generic RMSNorm and sigmoid-gate kernels with fp32 sequential
  accumulation. Row-wise norms back QK-Norm; the gate scales SDPA
  outputs for gated attention. Both built-ins take all dims as launch
  scalars and are verified against host references on both devices.
- Generic causal depthwise conv1d with fp32 sequential accumulation.
  Linear-attention blocks run the qkv mix through this before the
  recurrent scan. Verified against the host reference on both devices.
- Generic gated delta-rule scan step with fp32 sequential
  accumulation. One step reads, edits, and writes the recurrent state
  in place and reports the query read-out. Verified over three chained
  steps (outputs and carried state) against the host reference on both
  devices.
- Generic multimodal RoPE with fp32 sequential accumulation. Pair
  sections take per-row temporal/height/width ids with a continuous
  global frequency schedule. Text-only rows reduce to plain RoPE
  (asserted on device). Verified against the host reference on both
  devices.
- Weight upload: `Model::Load` allocates one device buffer per
  manifest tensor and copies the file bytes through the backend.
  `Model::Weights` exposes them next to the manifest. Unsized
  layouts stay in the manifest but fail the load as unsupported.
- Single-token decode loop in the core (backend agnostic): embed,
  block forward (device GEMM/RoPE/attention, host norms), greedy
  sample. It runs on vanilla-layout GGUF models. Hybrid definitions
  load fully (the 27B target loads with 866 tensors on both backends)
  but `DecodeStep` rejects them as unsupported until the recurrent
  kernels land. Norms and SiLU run on the host until their device
  kernels land.
- DFlash2 strategy skeleton. It validates the draft checkpoint layout.
- CLI: `tessera-cli run --model <path> [--draft <dir>]`. It loads the
  model, uploads weights, and prints a tensor summary. It does not run
  decode steps yet.
- Single GoogleTest target. Device dependent tests skip cleanly when
  no device is present. Numerical checks use per backend tolerance.

## Done

- 2026-10-07: initial commit (AGENTS.md, LICENSE).
- 2026-10-07: boilerplate. Layout, public API, CMake, GGUF parser,
  safetensors checks, backend device plumbing, DFlash2 skeleton, CLI,
  tests.
- 2026-10-07: full test suite green on both builds (58/58 `ctest`
  each). A missing `vkBindBufferMemory` call made the vulkan driver
  crash at copy time. The fix binds every buffer to its memory at
  allocation. The multi weight file layout is now rejected as
  malformed.
- 2026-10-07: kernel launch plumbing (62/62 `ctest` each). Vulkan:
  SPIR-V pipelines on a shared descriptor layout, scalars as push
  constants, `fill.comp` compiled at configure time by
  glslangValidator. ROCm: built-in `__global__` kernels compiled with
  the ROCm toolchain (`enable_language(HIP)`), launched through
  `hipLaunchKernelExC`. The "fill" kernel is verified by read-back on
  both devices.
- 2026-10-07: generic GEMM kernel (76/76 `ctest` on both builds).
  Three launch binding bugs are fixed. Vulkan now exposes one
  descriptor binding per buffer slot (the shaders declare one block
  per binding). Both backends set argument pointers only after the
  target vector stops growing (the old code stored `&vec.back()`
  across a reallocation, so multi buffer launches read freed memory).
  The ROCm device helpers carry `__host__ __device__` (first ROCm
  build). The fix was verified on RADV and also reproduces on
  llvmpipe, so it is a backend bug, not a driver bug.
- 2026-10-07: attention (GQA) and RoPE kernels (84/84 `ctest` on
  both builds). `Model::Attention` parses the attention config from
  GGUF metadata (`<arch>.attention.head_count`, head_count_kv,
  embedding_length, `<arch>.rope.dimension_count`, rope.freq_base).
  New generic built-ins "rope" (NeoX pairing, in place) and
  "attention" (causal GQA, scale 1/sqrt(d)) on vulkan (GLSL) and
  rocm (HIP), checked against fp32 host references with per backend
  tolerance. Shared test helpers moved to `tests/test_helpers.hpp`;
  `tests/backend_test.cpp` exceeds 600 lines once (owner approved).
- 2026-10-07: weight upload (88/88 `ctest` on both builds). The new
  canonical `TensorBytes` sizer in `tessera/types.hpp` covers plain
  and block dtypes; the GGUF parser uses it and exposes the tensor
  data start. `Model::Load` reads the file once, uploads every
  tensor, and keeps the buffers in `Model::Weights` with a
  `FindWeight` lookup. The CLI prints the device byte count.
- 2026-10-07: decode loop (91/91 `ctest` on both builds).
  `TransformerConfig` comes from GGUF metadata and `DecodeStep`
  runs one greedy step (device projections/RoPE/attention, host
  norms/SiLU/argmax, host KV cache). A 1-layer vanilla fixture
  decodes deterministically on both backends. A fixed descriptor
  pool leak surfaced (the pool missed FREE_DESCRIPTOR_SET_BIT).
  `CopyD2HAt` reads embedding rows; the GGUF array cap is now 1M
  (248k-token vocabularies). `tests/backend_test.cpp` grows past
  the 600 cap again (one offset test; exemption stands).
- 2026-10-07: MXFP4 path (102/102 `ctest` on both builds). A bounded
  schema-strict JSON reader parses the safetensors map (no third
  party dependency, by decision). U8 blobs become F4E2M1 against
  their paired `.weight_scale`, U8 scales become F8E8M0, and the
  manifest carries element shapes. New generic "gemm_fp8" (per-row
  scaled E4M3) and "gemm_mxfp4" (32-element E8M0 blocks) built-ins
  on vulkan (GLSL) and rocm (HIP) with exact codec unit tests. The
  OCP E4M3 top bin needed care (only mantissa-all-ones is NaN,
  max 448). `tests/backend_test.cpp` grows further (exemption
  stands); `src/core/loaders/safetensors.cpp` nears the 600 cap.
- 2026-10-07: K-quant and IQ GEMM paths (111/111 `ctest` on both
  builds). Unknown ggml ids pinned from llama.cpp as IQ4_NL,
  IQ3_S, IQ4_XS (byte sizes match the file to the byte) and the
  matching dequant math ported with host references. Six generic
  kernels per backend with device-vs-reference tests. Fixed along
  the way: Q8_0 blocks are 34 bytes (the old 33 truncated uploads),
  a Q3_K scale-hoisting bug in GLSL, and unsigned wraparound in the
  ROCm Q3_K kernel. The rocm backend split into focused translation
  units (rule 4); shared device helpers are inline in the private
  kernel header. Borrowed math is credited in CREDITS.md.
- 2026-10-07: hybrid definition and load (115/115 `ctest` on both
  builds). `TransformerConfig` carries a hybrid flag,
  `SsmParams` and the full-attention interval with an
  `IsFullAttentionLayer` mapper. GGUF parsing reads explicit
  key/value lengths (no embedding divisibility needed), the ssm.*
  keys, and counts trunk blocks (block_count minus
  nextn_predict_layers). The 27B target loads on both backends (866
  tensors); `DecodeStep` rejects hybrid configs as
  UnsupportedFeature. Layer kinds follow the llama.cpp interval rule,
  credited in CREDITS.md.
- 2026-10-07: norm and gate kernels (119/119 `ctest` on both builds).
  Generic "rmsnorm" (one thread per row) and
  "sigmoid_gate" (one thread per element) built-ins on vulkan (GLSL)
  and rocm (HIP) with host references in `src/core/numerics/norm.*`.
  Contracts live in `include/tessera/backend.hpp` next to the other
  built-ins. Device-vs-reference tests use the per backend attention
  tolerance. `tests/backend_test.cpp` grows further (exemption
  stands). The math follows public papers (Gated DeltaNet, gated
  attention, Qwen3-Next blog), credited in CREDITS.md; no llama.cpp
  source was read, per the new AGENTS.md rule.
- 2026-10-07: conv1d kernel (121/121 `ctest` on both builds).
  Generic causal depthwise "conv1d" (one thread per
  output element) on vulkan (GLSL) and rocm (HIP) with a host
  reference in the new `src/core/numerics/conv.*` pair. Contract in
  `include/tessera/backend.hpp`; device-vs-reference and bad-arg
  tests use the per backend attention tolerance.
- 2026-10-07: delta scan step (123/123 `ctest` on both builds).
  Generic "delta_step" (one thread per state column,
  state updated in place) on vulkan (GLSL) and rocm (HIP) with a host
  reference next to the attention refs. Three chained steps prove the
  carried state matches on both devices. Contract in
  `include/tessera/backend.hpp`.
- 2026-10-07: mRoPE kernel (126/126 `ctest` on both builds).
  Generic "mrope" (one thread per rotated pair) on
  vulkan (GLSL) and rocm (HIP) with a host reference next to the rope
  ref. Sections count pairs with a global frequency index, so text
  rows equal plain RoPE (asserted on device). Fixed along the way: the
  shader read the wrong word of the u64 position triples, and CMake
  keeps stale SPIR-V until `cmake -B` re-runs (see notes).
- 2026-10-07: rope sections in config (129/129 `ctest` on both
  builds). Current head. The GGUF parser retains arrays up to 16
  elements in `small_arrays`; longer ones stay dropped. The model
  config carries the mRoPE section counts (3 entries, or 4 with a
  zero pad) and hybrid definitions require them. The 27B target
  reports [11, 11, 10] on both backends.

## Next (in order)

1. **Hybrid SSM decode** for the 27B target (arch `qwen35`):
   definition, load, sections, and norm/gate/conv/scan/mrope kernels
   are done (the file loads with 866 tensors on both backends).
   Still missing: fused Q-plus-gate splitting, wiring the scan into
   the decode loop, and MTP handling. `DecodeStep` rejects hybrid
   configs as unsupported today.
2. **DFlash2**: local dynamic convolution (grouped causal convolutions),
   candidate selector (low rank transition scores), verification loop.
   Requires the full verifier vocabulary.
3. **Baseline pinning**: run the non speculative path on both backends.
   Record per backend tolerance. Assert in tests (fixed seeds).
4. **MoE, MLP, RMSNorm and embedding kernels** as the Qwen 3.8
   definition needs them. The generic RMSNorm kernel is done. The
   decode loop runs projections, RoPE and attention on the device.
   Norms and SiLU still run on the host. Models are data. No per
   model branches.
5. **Serving API**: OpenAI-style `/v1/chat/completions` plus an
   Anthropic-style `/v1/messages` endpoint, served over HTTP from the
   engine. Streaming and non-streaming responses. The same limits
   apply to both shapes.
6. **Runtime options**: every serving and engine knob as a CLI flag
   and an engine option. Model path, draft path, mmproj path for
   vision input, KV cache quantization (q4, q8, fp16), maximum
   context size, batch caps. No hard-coded paths or sizes.
7. **Multimodal (mmproj)**: load the vision projector next to the
   model, encode images to embeddings, prepend them to the prompt
   sequence. Covers the mmproj file in the model directory.

## Notes and decisions

- `std::expected` is mandated by AGENTS.md. It is a C++23 feature of
  libstdc++. The project compiles with `-std=c++23`. The code stays
  C++20 style.
- One backend per binary, selected at configure time
  (`TESSERA_BACKEND`). Verify fixes on the backend being changed. State
  which backend was used.
- Tests: single test target (`tessera-tests`), no new top level test
  files. Device dependent tests skip cleanly when no device is present.
- No model names in tests. A qwen shaped test tests the generic path
  with qwen shaped parameters.
- Diagnostics: `tessera::log::Diagnostics` with a pluggable sink is the
  logging channel. Every line carries `prefix: message` context (for
  example `engine 1: ...`).
- Kernel contract: `grid_*` is the workgroup count on both backends.
  The workgroup size is fixed at compile time on vulkan (SPIR-V
  LocalSize), so `block_*` is honored only on rocm. Scalars bind as
  64-bit values. A 32 bit field `k` reads `v[2k]` in the push
  constant array.
- This radv build crashes compiling a non-main SPIR-V entry name, so
  the vulkan built-ins keep the standard entry. It also crashes
  `vkCmdCopyBuffer` for buffers that were never bound to memory.
- Vulkan shaders embed at configure time: editing a `.comp` file
  needs a fresh `cmake -B` run, a plain `cmake --build` keeps the
  stale SPIR-V.
