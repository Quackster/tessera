# Progress

This file tracks tessera development. Update it as work progresses.
After each milestone, update "Done" and refresh "Next" (see AGENTS.md,
Working Principles).

## Current status

The boilerplate is complete and passes on both backends:

- Project layout and CMake. The backend is selected at configure time
  (`vulkan` or `rocm`).
- Public API: `Engine`, `Model`, `Backend`/`Buffer`,
  `SpeculativeStrategy`, `Diagnostics`.
- GGUF v2/v3 parser. It parses the header and the tensor manifest.
  It checks all bounds.
- Safetensors layout checks for MXFP4 model directories.
- Vulkan and ROCm backends: device init, buffer alloc/free, H2D/D2H
  copy, synchronize.
- Kernel launch plumbing on both backends. `LaunchKernel` binds buffers
  and 64-bit scalars in parameter order and waits on a fence (vulkan) or
  the hip runtime (rocm). The built-in "fill" kernel runs end to end
  and is verified by a read-back test on both devices.
- DFlash2 strategy skeleton. It validates the draft checkpoint layout.
- CLI: `tessera-cli run --model <path> [--draft <dir>]`.
- Generic GEMM kernel with GGUF Q4_K dequantization (fp32
  accumulation). A host reference check verifies it on both devices
  with per backend tolerance.
- Attention (GQA) and RoPE kernels driven by the model definition.
  `Model::Attention` reads heads, kv groups, head dim and the RoPE
  range and base from GGUF metadata. The generic "rope" and
  "attention" built-ins take all dims as launch scalars and are
  verified against host references on both devices.
- Single GoogleTest target. `ctest` passes 102/102 on both builds.
  Both builds were verified on AMD Radeon AI PRO R9700 (vulkan
  through RADV GFX1201, rocm through the system ROCm).
- MXFP4 tensor map: the safetensors header parser reads the fixed
  schema with a bounded hand-rolled reader. MXFP4 blobs pair with
  their E8M0 scales by name; MTP FP8 weights map natively. Model
  load builds the manifest and uploads every tensor.
- FP8 and MXFP4 GEMM kernels on both backends, checked against fp32
  host references with per backend tolerance.
- Single-token decode loop in the core (backend agnostic): embed,
  block forward (device GEMM/RoPE/attention, host norms), greedy
  sample. Runs on vanilla-layout GGUF models. The 27B target is a
  hybrid SSM model and stays unsupported (see the hybrid note).
- Weight upload: `Model::Load` allocates one device buffer per
  manifest tensor and copies the file bytes through the backend.
  `Model::Weights` exposes them next to the manifest. Unsized
  layouts stay in the manifest but fail the load as unsupported.

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
- 2026-10-07: decode loop (91/91 `ctest` on both builds).
  `TransformerConfig` comes from GGUF metadata and `DecodeStep`
  runs one greedy step (device projections/RoPE/attention, host
  norms/SiLU/argmax, host KV cache). A 1-layer vanilla fixture
  decodes deterministically on both backends. A fixed descriptor
  pool leak surfaced (the pool missed FREE_DESCRIPTOR_SET_BIT).
  `CopyD2HAt` reads embedding rows; the GGUF array cap is now 1M
  (248k-token vocabularies). `tests/backend_test.cpp` grows past
  the 600 cap again (one offset test; exemption stands).

## Next (in order)

1. **Hybrid SSM decode** for the 27B target (arch `qwen35`): fused
   QKV splitting, selective-scan and conv1d kernels, Q3_K and
   2026-type layouts (ids 20, 21, 23), Q5_K/Q6_K GEMM paths. The
   file probes as 866 tensors (248320 vocab, 65 blocks, hidden
   5120); the loader reports the first unmapped layout today.
2. **DFlash2**: local dynamic convolution (grouped causal convolutions),
   candidate selector (low rank transition scores), verification loop.
   Requires the full verifier vocabulary.
3. **Baseline pinning**: run the non speculative path on both backends.
   Record per backend tolerance. Assert in tests (fixed seeds).
4. **MoE, MLP, RMSNorm and embedding kernels** as the Qwen 3.8
   definition needs them. Models are data. No per model branches.
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
