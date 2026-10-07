# Progress

This file tracks tessera development. Update it as work progresses.
After each milestone, update "Done" and refresh "Next" (see AGENTS.md,
Working Principles).

## Current status

The boilerplate is complete and passes on both backends.
`ctest` passes 212/212 on both builds.
Both builds were verified on AMD Radeon AI PRO R9700 (vulkan
through RADV GFX1201, rocm through the system ROCm).

- Project layout and CMake. The backend is selected at configure time
  (`vulkan` or `rocm`).
- Public API: `Engine`, `Model`, `Backend`/`Buffer`,
  `SpeculativeStrategy`, `Diagnostics`.
- GGUF v2/v3 parser. It parses the header and the tensor manifest.
  It checks all bounds. Small metadata arrays stay retained. Bulk
  arrays stay dropped, except the tokenizer definition arrays.
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
  Q4_K, Q3_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS and IQ3_S are verified
  on both devices against host references with per backend tolerance.
  Exact codec unit tests cover the dequant math. FP8 (per-row scaled
  E4M3) and MXFP4 (32-element E8M0 blocks) kernels are verified the
  same way. The OCP E4M3 top bin needs care (only mantissa-all-ones is
  NaN, max 448).
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
- Generic fused gated-attention split: one fused projection holds
  query then gate per head; the split yields the queries and the gates
  for the sigmoid output gate. Verified against the host reference on
  both devices.
- Device-to-device launch helpers (`ProjectDevice`, `RmsNormDevice`,
  `AddDevice`, `SiluMulDevice`): chain kernels without host copies.
- Generic elementwise `add` (residual) and `silu_mul` (gated MLP)
  kernels with host references; the first building blocks for a
  device-resident decode loop.
- Generic L2 normalization and gated RMS normalization for the linear
  attention path: L2 normalizes delta queries and keys before the
  scan; the gated form normalizes the scan output and scales it by a
  SiLU gate. Both verified against host references on both devices.
- Weight upload: `Model::Load` allocates one device buffer per
  manifest tensor and copies the file bytes through the backend.
  `Model::Weights` exposes them next to the manifest. Unsized
  layouts stay in the manifest but fail the load as unsupported.
- Single-token decode loop in the core (backend agnostic): embed,
  block forward (device GEMM/RoPE/attention/conv/scan, host norms),
  greedy sample. It runs on vanilla-layout GGUF models and on hybrid
  models: the gated full-attention path (fused Q-plus-gate split,
  QK-Norm, mRoPE, sigmoid output gate) and the recurrent linear-
  attention path (causal conv1d, gated-delta scan, L2/gated norms) are
  both wired. Embeddings gather through the quantized formats. The 27B
  target generates on both backends. Norms and elementwise ops run on
  the host until their device kernels land.
- DFlash2 strategy skeleton. It validates the draft checkpoint layout.
- Public generation API: `Engine::Generate(model, options)` runs greedy
  single-token decode on the non-speculative (reference) path and
  returns the produced token ids. The CLI drives it.
- HTTP serving: a blocking HTTP/1.1 server (`tessera::Serve`) with a
  buffered and a chunked (SSE) response path, API-key auth
  (Bearer/X-Api-Key/x-api-key, env `TESSERA_API_KEY`) and a CORS
  allowlist. Endpoints: `/health`, `/metrics`, `/v1/models`, `/props`,
  `/tokenize`, `/detokenize`, `/slots`, `/v1/completions` (streaming
  SSE), `/v1/chat/completions` (SSE), `/v1/messages` and
  `/v1/messages/count_tokens`; unimplemented surfaces return 501.
  `Engine::GenerateStreaming` emits tokens one at a time. Chat prefill
  is slow on the 27B (host-glue decode), a performance, not correctness,
  gap.
- Chat-template renderer: a compact Jinja2-subset engine
  (`tessera/serve/jinja`) that runs the model's GGUF chat template.
  `Model::ChatTemplate` exposes the template. It matches the reference
  HF Jinja2 output on the 27B template (system/user/assistant, thinking
  on and off). Tools, images and video branches parse but need those
  features.
- Byte-level BPE tokenizer (`tessera::Tokenizer`) built from the GGUF
  tokenizer definition (vocab, merges, types). It applies the Qwen
  pre-tokenization split, matches the reference tokenizer on ASCII,
  Latin and CJK text, and round-trips decode. `Model::GetTokenizer`
  exposes it. Not applied: NFC normalization, special-token matching,
  combining marks and emoji.
- CLI: `tessera-cli run --model <path> [--draft <dir>] [--context <n>]
  [--draft-block <n>] [--prompt <id>] [--prompt-text <str>]
  [--tokens <n>]`. It loads the model, uploads weights, prints a tensor
  summary, and with `--tokens` runs greedy decode through
  `Engine::Generate` and prints the ids. `--prompt-text` tokenizes text
  with the model's tokenizer.
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
  builds). Unknown ggml ids pinned from the published GGUF/ggml
  block layouts as IQ4_NL, IQ3_S and IQ4_XS (byte sizes match the file
  to the byte) and the matching dequant math written with host
  references. Six generic
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
  UnsupportedFeature. Layer kinds follow the hybrid interval rule
  (every Nth block is full attention).
- 2026-10-07: norm and gate kernels (119/119 `ctest` on both builds).
  Generic "rmsnorm" (one thread per row) and
  "sigmoid_gate" (one thread per element) built-ins on vulkan (GLSL)
  and rocm (HIP) with host references in `src/core/numerics/norm.*`.
  Contracts live in `include/tessera/backend.hpp` next to the other
  built-ins. Device-vs-reference tests use the per backend attention
  tolerance. `tests/backend_test.cpp` grows further (exemption
  stands). The math follows public papers (Gated DeltaNet, gated
  attention, Qwen3-Next blog), credited in CREDITS.md.
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
  builds). The GGUF parser retains arrays up to 16
  elements in `small_arrays`; longer ones stay dropped. The model
  config carries the mRoPE section counts (3 entries, or 4 with a
  zero pad) and hybrid definitions require them. The 27B target
  reports [11, 11, 10] on both backends.
- 2026-10-07: gated-attention split kernel (131/131 `ctest` on both
  builds). Generic "qgate_split" (one thread per head
  element) splits the fused Q-plus-gate projection into queries and
  gates on vulkan (GLSL) and rocm (HIP). Layout taken from the public
  Qwen3-Next reference: q_proj output views as [heads, 2*head_dim] and
  chunks per head into query then gate; credited in CREDITS.md. Host
  reference next to the attention refs; contract in
  `include/tessera/backend.hpp`.
- 2026-10-07: Q8_0 GEMM kernel (132/132 `ctest` on both builds).
  Generic "gemm_q80" (34-byte blocks: fp16 scale plus 32
  signed bytes) on vulkan (GLSL) and rocm (HIP), with a host dequant
  and GEMM reference in the quant/gemm pair. The hybrid linear layers
  use Q8_0 for their gate projections, so this closes the last missing
  GEMM format before decode wiring.
- 2026-10-07: linear-attention normalizations (135/135 `ctest` on both
  builds). Generic "l2norm" and "rmsnorm_gated" (one
  thread per row) on vulkan (GLSL) and rocm (HIP) with host references
  in `src/core/numerics/norm.*`. The gated delta rule normalizes q and
  k with L2 and gates the scan output through the gated RMS norm.
  Contracts in `include/tessera/backend.hpp`. All projection, norm,
  gate, conv, scan, mrope and split kernels the hybrid path needs are
  now in place.
- 2026-10-07: hybrid full-attention decode + CLI (136/136 `ctest` on
  both builds). Shared host ops moved to
  `src/core/decode_internal.hpp`; the new `src/core/decode_hybrid.cpp`
  runs the gated full-attention path (fused Q-plus-gate split, QK-Norm,
  mRoPE, attention, sigmoid output gate, multi-format GEMM selection)
  and `DecodeStep` dispatches to it on a hybrid config. A tiny gated
  hybrid fixture decodes deterministically on both backends. The CLI
  gains `--prompt` and `--tokens` and runs greedy decode.
- 2026-10-07: recurrent linear-attention decode (137/137 `ctest` on
  both builds). `src/core/decode_hybrid.cpp` now runs
  the gated-delta block: fused qkv/gate projections, causal conv1d
  over a per-layer conv history, L2-normalized q/k with head repeat,
  a per-value-head delta scan with a carried recurrent state, and the
  gated RMS norm. An all-linear fixture decodes deterministically on
  both backends. Quantized embeddings gather through a new canonical
  `DequantizeBlocks` helper. Fixed along the way: Q4_K used a
  `{k, n, m}` scalar order while every other GEMM used `{m, n, k}`;
  all quant GEMMs now share one order. The 27B target generates on
  both backends (CLI `--tokens`); only MTP remains.
- 2026-10-07: baseline pinning (139/139 `ctest` on both builds).
  Current head. `tests/decode_test.cpp` pins the fixed-seed hybrid
  fixtures to exact greedy token sequences (gated and linear). The
  sequences are identical on vulkan and rocm (fp32 sequential
  accumulation, same op order), so a drift flags a kernel or scheduler
  change. The 27B baseline stays covered by the env-gated real-model
  test.
- 2026-10-07: public generation API + runtime options (140/140 `ctest`
  on both builds). `Engine::Generate` runs greedy decode
  on the non-speculative path and returns the token ids. The CLI gains
  `--context` and `--draft-block` (no more hard-coded sizes) and drives
  generation through the public API, so it no longer includes core
  internals.
- 2026-10-07: byte-level BPE tokenizer (145/145 `ctest` on both
  builds). Current head. The GGUF parser retains the tokenizer
  definition arrays; `tessera::Tokenizer` builds from the vocab, merge
  rules and types, and applies the Qwen pre-tokenization split. It
  matches the reference tokenizer exactly on ASCII, Latin, CJK and
  code samples (real-model test), and decode round-trips. `Model::
  GetTokenizer` exposes it; the CLI adds `--prompt-text` and
  `Engine::Generate` takes prompt tokens. Fixed the `\s+(?!\S)`
  backtracking the reference relies on. Not applied: NFC, special-
  token matching, combining marks and emoji.

- 2026-10-07: chat-template renderer (157/157 `ctest` on both builds).
  Current head. `src/serve/jinja` implements the Jinja2 subset the model
  templates use (macros, namespace, filters, tests, slicing, loop vars,
  whitespace control, tuples). `Model::ChatTemplate` exposes the GGUF
  template; the 27B template renders byte-for-byte identical to the
  reference HF Jinja2 output (thinking on and off). `tests/jinja_test.cpp`
  covers the engine plus the real template (env-gated).

- 2026-10-07: HTTP serving layer (158/158 `ctest` on both builds).
  Current head. The HTTP writer supports buffered and chunked/SSE
  responses; the server adds API-key auth and CORS and a first endpoint
  set (health, metrics, models, props, tokenize/detokenize, slots,
  completions/chat/messages with SSE). `Engine::GenerateStreaming`
  drives token-by-token output. The CLI `serve` gains `--api-key` and
  `--allow-origin`. Unimplemented endpoints answer 501.

- 2026-10-07: cache constant host weights (158/158 `ctest` on both
  builds). Current head. The decode caches download the F32 norm
  vectors and SSM scalars once (`DownloadF32Cached`) instead of
  re-downloading them every step; the decode determinism tests cover
  the cached path. Output is unchanged.

- 2026-10-07: device elementwise kernels (160/160 `ctest` on both
  builds). Current head. Generic add (residual) and silu_mul
  (gated MLP) built-ins on vulkan (GLSL) and rocm (HIP) with host
  references in src/core/numerics/norm.*, contracts and
  device-vs-reference tests. First step toward a device-resident
  decode loop (the current loop is host-orchestrated and slow).

- 2026-10-07: device-to-device launch helpers (161/161 `ctest` on
  both builds). Current head. `src/core/decode_internal.hpp` gains
  ProjectDevice/RmsNormDevice/AddDevice/SiluMulDevice so a block
  forward can keep activations on the device; a chained
  gemm_q4k -> rmsnorm -> add device test matches the host refs.

- 2026-10-07: device-to-device copy (162/162 `ctest` on both
  builds). Current head. `Backend::CopyD2D` moves byte ranges
  between device buffers (vkCmdCopyBuffer / hipMemcpy D2D) for a
  device-resident decode and cache; a bounds-checked copy test
  covers it on both backends.

- 2026-10-07: device-resident vanilla decode (162/162 `ctest` on
  both builds). Current head. `src/core/decode_device.cpp` keeps
  the vanilla activations on the device: embedding via CopyD2D, a
  device KV cache, and chained gemm/rmsnorm/add/silu_mul/rope/
  attention kernels with a single sync per step and only the
  logits downloaded. `DecodeStep` dispatches vanilla configs to it;
  the hybrid path is still host-orchestrated (next).

- 2026-10-07: hybrid device launch helpers (163/163 `ctest` on
  both builds). Current head. `decode_internal.hpp` adds
  QGateSplitDevice, MropeDevice, SigmoidGateDevice, L2NormDevice,
  RmsNormGatedDevice, Conv1dDevice and DeltaStepDevice, the
  device-to-device building blocks for a device-resident hybrid
  decode. A DeltaStepDevice test matches the host reference.

- 2026-10-07: repeat_heads kernel (164/164 `ctest` on both builds).
  Current head. The gated-delta layers repeat query/key heads to
  the value heads before the scan; generic "repeat_heads" on
  vulkan (GLSL) and rocm (HIP) with a host reference and
  device-vs-reference tests, a building block for the
  device-resident linear path.

- 2026-10-07: ssm_gate and delta_step_heads kernels (166/166
  `ctest` on both builds). Current head. "ssm_gate" computes the
  gated-delta decay/write gates per value head; "delta_step_heads"
  runs the gated-delta step for all heads in one launch with the
  device state in place. Both on vulkan (GLSL) and rocm (HIP) with
  host references and device-vs-reference tests.

- 2026-10-07: conv1d_step kernel (167/167 `ctest` on both builds).
  Current head. Generic "conv1d_step" computes the current-step
  causal depthwise conv output (channels x width, newest first) as
  one contiguous vector on vulkan (GLSL) and rocm (HIP) with a host
  reference and device-vs-reference tests. The device-resident
  linear path uses it so the conv output feeds the next kernel
  without a strided gather.

- 2026-10-07: device-resident hybrid decode (167/167 `ctest` on
  both builds). Current head. `decode_hybrid.cpp` now keeps the
  activations, full-attention KV caches and the linear recurrent
  state on the device, chaining the qgate-split/QK-norm/mRoPE/
  attention/sigmoid-gate and conv/L2/norm-gated/delta kernels with
  one sync per step. Only the causal conv and the final logits
  touch the host. The hybrid greedy baselines are unchanged. The
  27B gated-delta decode drops from about 8.5 s to about 1 s per
  token. The host-only `hybrid_ops.hpp` helper is removed.

- 2026-10-07: MTP head (168/168 `ctest` on both builds; the MTP test
  is env-gated). Current head. `decode_mtp.cpp` runs the Qwen3.5
  multi-token-prediction head: RMSNorm the token embedding and the
  backbone hidden, concat, `nextn.eh_proj` to one hidden vector,
  one full-attention block, the shared head norm and the shared
  output weight. `DecodeStep` can emit the final hidden state; the
  CLI `--mtp <id>` prints a draft. Unverifiable against a public
  reference output, but the layout matches vLLM's qwen3_5_mtp
  (credited). The 27B drafts deterministically.

- 2026-10-07: batched attention verified (169/169 `ctest` on both
  builds). Current head. The attention kernel already handles
  several query rows (m > 1); a device-vs-reference test pins it,
  a prerequisite for the speculative verifier.

- 2026-10-07: DFlash2 grouped conv kernel (170/170 `ctest` on both
  builds). Current head. Generic "dflash_conv" implements the
  DFlash2 local dynamic convolution: a per-tap base kernel plus a
  per-token per-group offset, reset every block_size positions, on
  vulkan (GLSL) and rocm (HIP) with a host reference and
  device-vs-reference tests. Layout from vLLM qwen3_dflash2.py
  (credited). First kernel of the DFlash2 draft.

- 2026-10-07: DFlash2 candidate-selector edge score (171/171 `ctest` on
  both builds). Current head. Generic "selector_edge_score"
  implements the DFlash2 low-rank predecessor/successor transition
  score that re-ranks the unary top-K, on vulkan (GLSL) and rocm
  (HIP) with int32 token-id gather, a host reference and
  device-vs-reference tests. Second kernel of the DFlash2 draft.

- 2026-10-07: Sliding-window attention for DFlash2 (172/172 `ctest` on
  both builds). Current head. The generic "attention" built-in takes
  a window scalar: a nonzero window keeps only the recent window
  keys, so DFlash2 draft layers (causal sliding attention, window
  2048) run on vulkan (GLSL) and rocm (HIP). Host reference, contract
  and device-vs-reference tests updated.

- 2026-10-07: dflash_conv delta row stride (173/173 `ctest` on both
  builds). Current head. dflash_conv now takes a delta_row_stride
  scalar so it consumes one side of the DFlash2 kernel_projection
  output [rows, 2, taps, num_groups] directly, matching the
  reference Triton kernel's delta_stride_row. Host reference, contract
  and a strided device-vs-reference test updated.

- 2026-10-07: verifier vocabulary (174/174 `ctest` on both builds).
  Current head. `DecodeLogits` returns the full vocab row for one
  step and `ScoreTokens` scores a token sequence from a fresh cache
(row i is the distribution after tokens[0..i]); the hybrid and
  device paths share one forward through a logits helper, and the
  step functions now argmax that row (single implementation). This is
  the primitive both speculative paths verify with. A test pins the
  ScoreTokens rows against the greedy baseline.

- 2026-10-07: greedy speculative verification loop (175/175 `ctest` on
  both builds). Current head. `VerifyDraft` feeds a draft token only
  while it equals the target's greedy token, then stops, so the cache
  is left at the accepted prefix and no state is rolled back. The
  accepted tokens plus the bonus token equal plain greedy decoding for
  any draft. Tests run the loop on the full-attention and stateful
  linear fixtures with all-rejected and accepted drafts and pin the
  output to the greedy baseline. Sequential today (one target forward
  per accepted token); batched scoring is the next speed step.

- 2026-10-07: MTP speculative generation (177/177 `ctest` on both
  builds; the real-model test is env-gated). Current head.
  `Engine::GenerateSpeculative` drafts with the MTP head, verifies
  with `VerifyDraft`, accepts matches and falls back to the target's
  greedy token on rejection (truncating the MTP key/value row); the
  output is identical to plain greedy. The CLI gains `--speculate`.
  Verified on the 27B (866 tensors): speculative equals greedy on
  vulkan and rocm. Sequential scoring today; batching is the next
  speed step.

- 2026-10-07: preallocated KV caches (178/178 `ctest` on both builds;
  the real-model tests are env-gated). Current head. The full-attention
  KV caches (vanilla and hybrid, plus the MTP block) now grow
  geometrically instead of reallocating and copying the whole cache
  every step, turning the per-step O(n) append into amortized O(1).
  The two duplicate append helpers are unified into one templated
  `AppendKv` in `decode_internal.hpp`. A device test covers the growth
  boundary; the pinned baselines and the 27B MTP path are unchanged.

- 2026-10-07: explicit GPU selection (179/179 `ctest` on both builds).
  Current head. `EngineOptions::device_index` (CLI `--gpu <n>`) selects
  the GPU; the default is 0, the first device. Backends no longer
  assume device 0: vulkan picks `devices[index]` and rocm calls
  `hipSetDevice(index)`, both rejecting an out-of-range index as
  InvalidArgument. The backend logs the selected index and name, and
  the engine start line carries the device index. Verified on the
  host (vulkan sees 3 devices, rocm 2).

- 2026-10-07: prompt prefill skips the output head (181/181 `ctest` on
  both builds). Current head. `DecodeForward`/`HybridForward`/
  `DecodeStepDeviceForward` run a block forward without the
  vocab-sized output projection. `Engine::Generate` and
  `GenerateSpeculative` use it for every prompt token except the last,
  which removes a full vocab GEMM and a logits download per prompt
  token. A hybrid test asserts the forward leaves the cache identical
  to a full step, and an engine test prefills a multi-token prompt.

- 2026-10-07: list GPUs (182/182 `ctest` on both builds). Current head.
  `tessera::ListGpuNames()` enumerates the configured backend's devices
  in index order; the CLI command `tessera-cli --list-gpus` prints
  `gpu <i>: <name>` and exits. Vulkan and rocm each build a temporary
  device context (no engine). The vulkan instance setup is shared by
  Init and the listing. An engine test checks the non-empty result.

- 2026-10-07: plain fp32 GEMM (183/183 `ctest` on both builds).
  Current head. Generic "gemm_f32" computes C = A x W^T for
  unquantized fp32 weights (the layout bf16 weights convert to), on
  vulkan (GLSL) and rocm (HIP) with a host reference and a
  device-vs-reference test. `GemmKernelName` maps DType::F32 to it, so
  a projection whose weights are fp32 now selects a kernel instead of
  UnsupportedFeature.

- 2026-10-07: bf16-weight GEMM (184/184 `ctest` on both builds).
  Current head. Generic "gemm_bf16" computes C = A x W^T with bf16
  weights and fp32 sequential accumulation, on vulkan (GLSL) and rocm
  (HIP) with a host reference and a device-vs-reference test.
  `GemmKernelName` maps DType::BF16 to it. The safetensors parser and
  the tensor sizer already handled BF16, so bf16 weights (the DFlash2
  draft projections) now select a kernel. bf16 weights are read
  two-per-word and shifted into fp32 (exact).

- 2026-10-07: block-scaled fp8 GEMM (185/185 `ctest` on both builds).
  Current head. Generic "gemm_fp8_block" computes C = A x W^T with
  FP8 E4M3 weights and one fp32 scale per 128 x 128 block (the DFlash2
  draft quantization `weight_scale_inv`), on vulkan (GLSL) and rocm
  (HIP) with a host reference and a device-vs-reference test. Together
  with gemm_f32 and gemm_bf16 this covers every DFlash2 draft weight
  dtype.

- 2026-10-07: DFlash2 draft config parsing (188/188 `ctest` on both
  builds). Current head. The bounded JSON DOM moves from `src/serve`
  to `src/core/json.*` (namespace `tessera::core`), so core can parse
  model configs without depending on the serving layer. New
  `src/spec/dflash2_config.*` parses the draft `config.json`: the
  transformer geometry, `rope_theta`, and the `dflash_config` block
  (block size, conv group/taps, mask token id, selector rank/top-K,
  target layer ids, layer types), with validation. Tests cover a
  valid config, malformed fields, and the real Qwen3.8-27B DFlash2
  checkpoint (env-gated).

- 2026-10-07: DFlash2 grouped-conv layer stage (189/189 `ctest` on
  both builds). Current head. `src/spec/dflash2_conv.*` composes one
  DFlash2 layer stage: the kernel_projection GEMM (gemm_f32) then one
  side of the grouped dynamic convolution (dflash_conv). `dflash_conv`
  gains a `delta_offset` scalar so it selects the side
  (`side * taps * num_groups`); the device helper copies the requested
  base side into scratch and runs both on device. Host reference and a
  device-vs-reference test cover both sides.

- 2026-10-07: DFlash2 MLP layer half (190/190 `ctest` on both builds).
  Current head. `src/spec/dflash2_mlp.*` runs the MLP half of a draft
  layer: post_attention RMSNorm, the mlp_conv prepare (projection +
  side-0 grouped conv), the gated SiLU MLP, and the mlp_conv finish
  (side-1 conv reusing the prepare projection). Adds the finish conv
  stage (convolve with an externally computed delta) to
  dflash2_conv. Host reference and a device-vs-reference test.

- 2026-10-07: DFlash2 attention layer half (191/191 `ctest` on both
  builds). Current head. `src/spec/dflash2_attention.*` runs the
  attention half of a draft layer: input_layernorm RMSNorm, the
  attention_conv prepare (projection + side-0 grouped conv), QKV
  projections, per-head QK-RMSNorm, RoPE, grouped-query sliding
  attention, the output projection and the attention_conv finish
  (side-1 conv reusing the prepare projection). Fixes latent
  single-row assumptions: `RopeDevice` and `AttentionDevice` took a
  row count (the decode path passes 1) so multi-token blocks rotate
  and attend every row. Host reference and a device-vs-reference
  test.

- 2026-10-07: DFlash2 full draft layer (192/192 `ctest` on both builds).
  Current head. `src/spec/dflash2_layer.*` assembles a draft layer:
  the pre-norm residual (add the carried residual, or use the input on
  the first layer), the attention half, a second residual add, and the
  MLP half, returning the MLP output and the new residual. Weight
  handles use small structs. Host reference and a device-vs-reference
  test, with and without a carried residual.

- 2026-10-07: DFlash2 draft stack (193/193 `ctest` on both builds).
  Current head. `src/spec/dflash2_stack.*` runs a list of draft layers
  in order, chaining the residual and finishing with the final
  RMSNorm over (last output + residual). Host reference and a
  device-vs-reference test with two layers.

- 2026-10-07: DFlash2 candidate-selector block (194/194 `ctest` on
  both builds). Current head. `src/spec/dflash2_selector.*` projects
  the draft hidden states to the selector rank (gemm_f32) then scores
  the predecessor/successor transitions of the unary top-K
  (selector_edge_score). Host reference and a device-vs-reference
  test. The target-hidden fusion (`fc`) is a plain GEMM over the
  concatenated aux hidden, so it needs no new code.

- 2026-10-07: concat_features and the DFlash2 target fusion (195/195
  `ctest` on both builds). Current head. Generic "concat_features"
  stacks n tensors of rows x features along the feature axis (host
  reference, vulkan + rocm kernels, device test). `src/spec/
  dflash2_fuse.*` composes it with the `fc` projection, the DFlash2
  target-hidden fusion, with a host reference and device test.

- 2026-10-07: sampling parameters (197/197 `ctest` on both builds).
  Current head. `GenerateOptions` gains `sample`, a `SamplingOptions`
  block with the Qwen 3.8 27B defaults (temperature 0.6, top_p 0.95,
  top_k 20, min_p 0.0, presence_penalty 0.0, repetition_penalty 1.0)
  and a seed. `src/core/sampling.*` draws a token: repetition and
  presence penalties over the history, temperature (0 is greedy),
  min_p, top_k and top_p, then a seeded draw. The generation loop
  scores with DecodeLogits and picks greedy or sampled. Tests cover
  the filters and seeded determinism; the greedy baseline is
  unchanged. The CLI gains `--sample` and the parameter flags.

- 2026-10-07: DFlash2 context K/V projection (198/198 `ctest` on both
  builds). Current head. `src/spec/dflash2_context.*` turns the fused
  target hidden into the draft attention context: hidden_norm RMSNorm,
  the layer k_proj/v_proj, K-norm and RoPE. Host reference and a
  device-vs-reference test. This is the input the context-aware draft
  attention will consume.

- 2026-10-07: DFlash2 context attention (199/199 `ctest` on both builds).
  Current head. `DraftAttentionRef`/`Device` take an optional context
  K/V prefix: the queries attend over the context rows followed by the
  block's own keys, and the query positions start at `pos_base + ctx`
  (RoPE and causality/window). With no context the behaviour is
  unchanged. Host reference and a device-vs-reference test.

- 2026-10-07: DFlash2 context through the draft layer and stack (199/199
  `ctest` on both builds). Current head. A draft layer gains a shared
  hidden_norm and an optional context hidden input: when present it
  forms the layer's context K/V (DraftContextKv) and the attention
  attends over it; the stack passes the fused context hidden to each
  layer. The layer test now covers the residual x context combinations
  on both backends.

- 2026-10-07: DFlash2 draft block (200/200 `ctest` on both builds).
  Current head. `src/spec/dflash2_block.*` ties the draft forward
  together: fuse the target hidden states with `fc`, run the mask-query
  layer stack with that fused hidden as context, and project the final
  normed hidden to logits with the shared output weight. Host reference
  and a device-vs-reference test. Fixes a dropped `ctx` argument in the
  device stack call.

- 2026-10-07: bf16 and block-fp8 dequant to fp32 (201/201 `ctest` on
  both builds). Current head. `Bf16ToFloat` moves next to the other
  codecs in `quant.*` (gemm_bf16 shares it) and new
  `DequantizeBf16`/`DequantizeFp8Block` decode whole tensors to fp32
  (the latter with the DFlash2 128x128 block scales). Host tests
  cover both. These are the building blocks for loading the real
  bf16/fp8 DFlash2 draft weights into the fp32 draft forward.

- 2026-10-07: DFlash2 draft weight loader (202/202 `ctest` on both
  builds; the real-checkpoint test is env-gated). Current head.
  `src/spec/dflash2_weights.*` reads the draft safetensors file,
  converts each forward tensor from bf16 or 128x128 block-scaled fp8
  to fp32 (the block scales are bf16) and uploads it, binding the
  per-layer buffers plus the shared fc, hidden_norm and final norm.
  The real Qwen3.8-27B DFlash2 draft loads and binds.

- 2026-10-07: DFlash2 candidate extraction (203/203 `ctest` on both
  builds). Current head. `src/spec/dflash2_candidates.*` returns the
  top-K draft logits per row (ids and unary values, descending), the
  input the candidate selector re-ranks. Host test.

- 2026-10-07: DFlash2 drafter runs the real draft (204/204 `ctest` on
  both builds; the real-checkpoint test is env-gated). Current head.
  `src/spec/dflash2_drafter.*` loads the draft weights with the block
  kernels and runs one draft block (fc fusion, mask-query stack with
  context, final norm, head projection). The real Qwen3.8-27B DFlash2
  draft loads and produces finite logits on the device.

- 2026-10-07: target hidden capture (205/205 `ctest` on both builds).
  Current head. `HybridForward` takes an optional layer list and
  capture buffers: after each listed layer it copies the
  residual-stream hidden on the device. This is how the DFlash2 draft
  gets the target hidden at `target_layer_ids`. On the one-layer
  fixture the captured layer-0 hidden equals the final hidden.

- 2026-10-07: DFlash2 mask embeddings (206/206 `ctest` on both builds).
  Current head. `src/spec/dflash2_mask.*` gathers the target
  embedding row at `mask_token_id` (any quantized format, via the
  shared gather) and tiles it to the draft block query rows. Test on
  the gated fixture: every row equals the gathered row.

- 2026-10-07: DFlash2 speculative generation (207/207 `ctest` on both
  builds; the real-model test is env-gated). Current head.
  `spec::GenerateDFlash2` captures the target hidden at the draft's
  target_layer_ids during each target step, drafts a top-1 block with
  the loaded draft (mask-token queries, fused context, shared
  quantized head), verifies the block with `VerifyDraft`, and accepts
  the matching prefix. `Engine::GenerateDraft` exposes it; the CLI
  routes `--draft` generation through it. Output equals greedy on the
  27B. Fixed a double-emitted bonus token.

- 2026-10-07: DFlash2 candidate selector wired (207/207 `ctest` on both
  builds; the real-model test is env-gated). Current head. The draft
  weight loader also loads the candidate selector (projection and
  codebooks, fp32); generation takes the top-K draft candidates,
  re-ranks them with the predecessor/successor transition scores, and
  chains the best successor per position. Falls back to the unary top-1
  when the checkpoint has no selector. Output still equals greedy on
  the 27B.

- 2026-10-07: DFlash2 draft-block override (207/207 `ctest` on both
  builds). Current head. `GenerateOptions::draft_tokens` (CLI
  `--draft-block`) caps the DFlash2 draft to that many tokens per step
(at most the checkpoint block size); 0 uses the checkpoint size.

- 2026-10-07: fp16 keys/values attention (208/208 `ctest` on both builds).
  Current head. The "attention" built-in takes a kv_f16 flag: keys and
  values are read as fp16 when set, fp32 otherwise. `AttentionRefF16`
  decodes fp16 K/V for the host reference and `Fp16FromFloat` encodes
  fp32 to fp16. Device-vs-reference test on both backends. This is the
  kernel half of the fp16 KV cache (storage wiring next).

- 2026-10-07: fp32 -> fp16 cast kernel (209/209 `ctest` on both builds).
  Current head. Generic "cast_f32_f16" packs two fp32 into one fp16
  word (vulkan + rocm, host reference, device test). `Fp16FromFloat`
  encodes fp32 to fp16. With the fp16 attention flag this covers both
  halves of the fp16 KV cache; cache/storage wiring is next.

- 2026-10-07: q8 KV kernels (212/212 `ctest` on both builds). Current
  head. Generic "quantize_q8" symmetrically quantizes fp32 rows to
  packed int8 with a per-row absmax scale; "attention_q8" runs GQA
  over int8 keys/values with per-key-row scales. Host references and
  device-vs-reference tests. These are the kernels for the q8 KV cache
(storage/wiring next), alongside the fp16 cache.

## Next (in order)

1. **Speculative decoding performance**: MTP speculation runs end to
   end (`Engine::GenerateSpeculative`, CLI `--speculate`) and is output
   preserving, but scores one token per target forward. Next: batch the
   draft scoring so k drafts cost one forward.
2. **DFlash2**: runs end to end (`Engine::GenerateDraft`, CLI `--draft`):
   grouped dynamic convolution, sliding attention, candidate selector and
   the verification loop, output equal to greedy. Remaining: acceptance
   tuning (context beyond the last token) and batching.
3. **MoE, MLP, RMSNorm and embedding kernels** as the Qwen 3.8
   definition needs them. The generic RMSNorm kernel is done. The
   decode loop runs projections, RoPE and attention on the device.
   Norms and SiLU still run on the host. Models are data. No per
   model branches.
4. **Serving API (DEFERRED)**: do not extend the HTTP surface unless
   explicitly told. A first slice lives in `src/serve/` (`/health`,
   `/metrics`, `/v1/models`, `/props`, `/tokenize`, `/detokenize`,
   `/slots`, `/v1/completions`, `/v1/chat/completions`, `/v1/messages`,
   `/v1/messages/count_tokens`, SSE for completions/chat/messages,
   API-key auth, CORS). Not done and deferred: thread pool/engine
   queue, keep-alive, `/v1/responses`, render/derender/batch,
   `/tokenizer_info`, `/load` and LoRA, and the 501
   embedding/rerank/audio/pooling/classify/score surfaces.
5. **Runtime options**: context size, draft-block and the GPU index
   (`--gpu`) are CLI flags now, and the KV cache can be fp16 (`--kv-f16`).
   Still to wire: q4/q8 KV quantization, mmproj path for vision input, and
   batch caps (features that do not exist yet). No hard-coded paths or
   sizes.
6. **Multimodal (mmproj)**: load the vision projector next to the
   model, encode images to embeddings, prepend them to the prompt
   sequence. Covers the mmproj file in the model directory.

7. **Multi-GPU (deferred)**: today `--gpu` selects one device and there is
   one `Backend` per engine. Two researched routes: tensor parallelism
   (shard attention heads and MLP rows across GPUs with an all-reduce per
   layer; vLLM tensor parallelism, Megatron-LM TP) or layer/pipeline
   split (assign blocks to GPUs and hand off activations; Megatron-LM
   pipeline parallelism). Both need several `Backend` instances, weight
   sharding in the loaders, and cross-device collectives or peer copies.
   Not started.

## Notes and decisions

- HTTP serving is deferred. `src/serve/` already serves a first
  endpoint set; do not add endpoints, SSE variants, auth changes, or
  concurrency there unless the project owner explicitly asks. The
  decode loop is host-orchestrated and slow (chat prefill on the 27B
  exceeds minutes), so performance work comes before more endpoints.
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
