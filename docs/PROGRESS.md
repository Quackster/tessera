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
- Single GoogleTest target. `ctest` passes 76/76 on both builds.
  Both builds were verified on AMD Radeon AI PRO R9700 (vulkan
  through RADV GFX1201, rocm through the system ROCm).

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

## Next (in order)

1. **Attention (GQA) and RoPE kernels** driven by the model definition.
2. **Weight upload**: manifest to device buffers via the backend.
   Complete `Model::Load`. Today it only validates and parses the
   manifest.
3. **Decode loop**: single token generation in the core (backend
   agnostic). End to end smoke on `Qwen3.8-27B-UD-Q4_K_M`.
4. **MXFP4 path**: full safetensors tensor map parsing (JSON reader for
   the fixed schema, decision pending). fp8 and mxfp4 GEMM kernels.
   MTP-FP8 draft weights.
5. **DFlash2**: local dynamic convolution (grouped causal convolutions),
   candidate selector (low rank transition scores), verification loop.
   Requires the full verifier vocabulary.
6. **Baseline pinning**: run the non speculative path on both backends.
   Record per backend tolerance. Assert in tests (fixed seeds).
7. **MoE, MLP, RMSNorm and embedding kernels** as the Qwen 3.8
   definition needs them. Models are data. No per model branches.

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
