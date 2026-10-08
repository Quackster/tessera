# Progress

This file tracks tessera development. Update it as work progresses.
After each milestone, update "Done" and refresh "Next" (see AGENTS.md,
Working Principles).

## Current status

The boilerplate is complete and passes on both backends.
`ctest` passes 229/229 on both builds.
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

- 2026-10-07: q8 KV cache (213/213 `ctest` on both builds). Current
  head. `GenerateOptions::kv_type` (CLI `--kv-f16`/`--kv-q8`) selects
  the full-attention KV storage: fp32, fp16 or symmetric int8 with a
  per-row scale (`quantize_q8` + `attention_q8`). The baseline stays
  fp32; the fp16 and q8 paths are deterministic on the fixtures and run
  on the 27B (same leading tokens).

- 2026-10-07: q4 KV cache (216/216 `ctest` on both builds). Current
  head. `KvCacheType::Q4` (`--kv-q4`) stores the full-attention KV in
  symmetric 4-bit with a per-row scale (`quantize_q4` + `attention_q4`,
  eight nibbles per word). Default fp32; fp16/q8/q4 are opt-in. The q4
  path is deterministic on the fixtures and runs on the 27B (a lossy
  mode, so the greedy tokens differ from fp32 by design).

- 2026-10-07: vision (mmproj) config (218/218 `ctest` on both builds;
  the real-file test is env-gated). Current head. `tessera::
  LoadVisionConfig` parses the mmproj GGUF (CLIP vision encoder +
  merger): image/patch size, embedding/ffn/block/head counts,
  projection dim, spatial merge, layer-norm eps, image mean/std,
  projector type. The real Qwen3.8 mmproj parses; a non-CLIP GGUF is
  MalformedFile. This is the first multimodal step; the encoder and
  merger forward follow.

- 2026-10-07: LayerNorm and GELU kernels (220/220 `ctest` on both
  builds). Current head. Generic "layernorm" (row-wise, weight and
  bias, biased variance) and "gelu" (elementwise tanh approximation)
  on vulkan and rocm with host references and device-vs-reference
  tests. The CLIP vision encoder needs both (it uses LayerNorm and
  GELU, not RMSNorm/SiLU).

- 2026-10-07: image patchify (221/221 `ctest` on both builds). Current
  head. Generic "image_patchify" normalizes an [h, w, 3] image by the
  per-channel mean/std and splits it into patch x patch patches in the
  CLIP conv weight layout (host reference, vulkan + rocm, device test).
  This is the CLIP vision encoder input step.

- 2026-10-07: non-causal attention (222/222 `ctest` on both builds).
  Current head. The "attention" built-in takes a causal flag (0 attends
  all n keys, 1 the causal prefix); the CLIP vision encoder is
  bidirectional. Push constants widen to 128 bytes and kMaxScalars to
  16 for the ninth scalar. Host reference and non-causal device-vs-
  reference test.

- 2026-10-07: CLIP vision transformer block (223/223 `ctest` on both
  builds). Current head. `src/core/vision_block.*` runs one CLIP block:
  LayerNorm, fused QKV (split per row), non-causal multi-head attention,
  output projection, residual; LayerNorm, GELU MLP, residual. Adds the
  generic "bias_add" (row-broadcast) kernel. Host reference and a
  device-vs-reference test.

- 2026-10-07: CLIP vision stack (224/224 `ctest` on both builds).
  Current head. `src/core/vision_stack.*` runs the vision encoder body:
  patch embedding (bias) + position embedding, the transformer blocks,
  and the final LayerNorm. Adds a shared `LayerNormDevice`/`BiasAddDevice`
  helper. Host reference and a device-vs-reference test (two blocks).

- 2026-10-07: vision merger (225/225 `ctest` on both builds). Current
  head. `src/core/vision_merger.*` runs the Qwen3VL merger: 2x2 spatial
  merge (`spatial_merge` kernel) then mm.0 + bias -> GELU -> mm.2 +
  bias into the language hidden space. Host reference and a
  device-vs-reference test.

- 2026-10-07: vision model load + encode (226/226 `ctest` on both
  builds; the real-file test is env-gated). Current head.
  `tessera::VisionModel` loads the mmproj (bf16 tensors converted to
  fp32, F32 norms kept), binds the patch/position/post-norm/block/
  merger weights, and `Encode` runs patchify -> stack -> merger to
  image embeddings. The real Qwen3.8 mmproj loads and encodes a
  synthetic image to finite embeddings.

- 2026-10-07: image-embedding injection (227/227 `ctest` on both
  builds). Current head. `DecodeForward`/`DecodeLogits` (and the hybrid
  forward) take an optional device embedding buffer that replaces the
  gathered token embedding, so a precomputed image embedding can be fed
  as a prompt token. Test: feeding a gathered embedding buffer equals
  decoding the same token.

- 2026-10-07: image load and resize (228/228 `ctest` on both builds).
  Current head. Dependency-free binary PPM (P6) loader
  (`tessera::LoadPpm`) to fp32 RGB [0, 1] and bilinear resize
  (`ResizeBilinear`). Host tests cover a tiny PPM and resize. This is
  the image input step before the vision encoder.

- 2026-10-07: multimodal generation (229/229 `ctest` on both builds).
  Current head. `Engine::GenerateMultimodal` feeds image embeddings at
  the `image_token_id` placeholder positions during prefill (the first
  `image_tokens` placeholders consume consecutive embedding rows) and
  then generates. Test: deterministic on the gated fixture. The
  vision encoder + image loader + injection now connect end to end at
  the engine level.

- 2026-10-07: multimodal CLI (229/229 `ctest` on both builds). Current
  head. `tessera-cli run --mmproj <dir> --image <ppm> --image-token <id>`
  loads the vision projector, encodes the image, prepends its token
  count as placeholders and calls `GenerateMultimodal`. The prefill of
  the ~576 image tokens is token-by-token (host-glue, ~1 s each), so a
  full image run takes minutes; batched prefill would fix it.

- 2026-10-07: CLI progress logging (229/229 `ctest` on both builds).
  Current head. `GenerateOptions::progress_every` (default 64) makes the
  engine log prefill progress every N tokens and a final
  `generated N token(s) in X ms (Y ms/token)` line. The CLI logs the KV
  cache type, the sampling settings, the vision projector load, the
  image load/encode, and the encoded token count; `--quiet` hides all
  of it.

- 2026-10-07: CLI --log-tokens (229/229 `ctest` on both builds).
  Current head. `tessera-cli --log-tokens` decodes the input and output
  token ids to text with the model tokenizer and logs them; the default
  logs only `generated N token(s)` (the engine still logs the timing).

- 2026-10-07: coherent 27B decoding (230/230 `ctest` on vulkan). Greedy
  generation on the real Qwen 3.8 27B GGUF answers coherently now
  ("The capital of France is **Paris**."; "Once upon a time, in a land
  far away, ..."). Two linear-attention bugs remained after the earlier
  decay and conv fixes. First, the gated-delta query/key heads repeat to
  the value heads with the interleaved GQA map (`value head h` uses
  `key head h % num_k_heads`, not `h / factor`); the grouped map paired
  the wrong key head with each value head. Second, the recurrence scales
  the query by `1/sqrt(head_k_dim)`; without it the delta output was
  ~11x too large, which changed the RMSNormGated eps behavior and the
  gated norm. `RepeatHeadsRef` and both `repeat_heads` kernels use the
  interleaved map; `l2norm` takes a scale scalar on both backends, with
  the host reference and `BackendTest.L2NormDeviceMatchesRef` updated.
  Confirmed against llama.cpp's per-node graph at layer 0: the predicted
  delta output equals the reference `attn_output` exactly. The synthetic
  linear-fixture pins moved to `{21, 28, 24, 6, 9, 15, 18, 24}`. (The
  earlier fused Q+gate "all queries then all gates" change was wrong and
  is reverted; the gate is per head `[q, gate]`.)

- 2026-10-07: batched speculative scoring (232/232 `ctest` on both
  builds). `VerifyDraft` now scores a multi-token draft in one target
  forward. `HybridForwardBatch` runs the whole hybrid trunk with the GEMMs
  and attention at m = draft length (causal within the block) and records
  per-token linear-attention state snapshots; verification accepts the
  matching prefix and rolls the cache back (full-attention KV rows,
  recurrent state, conv history and position). `qgate_split` and
  `ssm_gate` gained a `rows` scalar on both backends; the gemm, rmsnorm,
  l2norm, mrope and attention helpers were already row based.
  `Engine::GenerateSpeculative` drafts up to `draft_tokens` tokens by
  chaining the MTP head (the shared-head-norm output feeds the next step)
  and verifies them in one forward; on the 27B this is about 1.5x faster
  than greedy and output equal. The DFlash2 block draft uses the same
  batched verifier.

- 2026-10-07: batched prefill (233/233 `ctest` on both builds).
  `Engine::Generate`/`GenerateStreaming` and `GenerateSpeculative` prefill
  the whole prompt in one batched trunk forward (`PrefillTokens`), with
  the output head on the last row only, instead of one forward per token.
  A 27-token prompt drops from about 38 s to 8.9 s; the last logits and
  the retained hidden match the sequential prefill. `GenerateMultimodal`
  passes one embedding per row (text rows gathered, image rows substituted)
  to the same batched prefill, so the image prompt is one forward too.

- 2026-10-07: image-placeholder token mapping (235/235 `ctest` on both
  builds). `Tokenizer::SpecialTokenId` looks up a special token by its
  literal string. The CLI defaults `--image-token` to the model's
  `<|image_pad|>` id so image prompts work without passing it.

- 2026-10-07: O(n*d) attention kernel (235/235 `ctest` on both builds).
  The attention kernel recomputed the query/key dot product once per
  output dimension (O(n * head_dim^2) per query). It now runs one
  workgroup per (query row, head) with a tiled online softmax: the query
  is staged in shared memory, each thread scores one key per tile, and
  the tile weights feed the value accumulation, so the cost is
  O(n * head_dim) per query/head. head_dim is capped at 256. Both
  backends; the attention_q8/attention_q4 variants keep the old shape.

- 2026-10-07: architecture module interface (235/235 `ctest` on both
  builds). Added the `Architecture` interface
  (`include/tessera/architecture.hpp`), the registry
  (`src/models/registry.cpp`) and the first module `src/models/qwen3_5/`
  (registered for `general.architecture` qwen35). `Model::Arch()` builds
  the module at load; `MtpDraftStep` dispatches through it. The hybrid
  trunk (`src/core/decode_hybrid*.cpp`) still moves to the module next.

- 2026-10-07: hybrid trunk moved into the Qwen3.5 module (235/235
  `ctest` on both builds). The `Architecture` interface gained Forward,
  Logits, ForwardBatch and Verify; the hybrid trunk
  (`src/models/qwen3_5/trunk.cpp`, `trunk_batch.cpp`) and its helpers
  (`internal.hpp`) moved out of `src/core` behind it, and
  `src/core/decode.cpp` dispatches by `model.Arch()`. The generic
  fallback (null module) keeps the device path. The synthetic hybrid
  fixtures now declare `general.architecture` qwen35.

- 2026-10-07: hybrid decode state moved into the Qwen3.5 module (235/235
  `ctest` on both builds). `core::DecodeCache` holds an opaque `ArchState`
  (subclassed by `Qwen35State` in src/models/qwen3_5/state.hpp); the MTP
  key/value rollback goes through new `Architecture::DraftRows` and
  `DraftTruncate`. `src/core` no longer mentions the hybrid state, and no
  core file names a model: dispatch, state creation and the draft cache
  all go through `model.Arch()`. The generic transformer path and the
  GGUF config parse stay in core (format-generic, not architecture
  specific).

- 2026-10-07: DFlash2 draft context width (235/235 `ctest` on both
  builds). The draft now conditions on a configurable window of target
  positions: `DraftBlockDevice` gets the real `pos_base`, and the
  generate keeps a rolling history of the captured target hidden and
  builds `aux` for the window (default width 1, set with
  `TESSERA_DFLASH2_CTX`). Wider context does not raise acceptance (it
  stays about 0 to 15 percent), so the draft block quality, not the
  context, is the limit.

- 2026-10-08: MXFP4 multimodal tensors parse (237/237 `ctest` on both
  builds). `TensorShape::kMaxRank` is raised from 4 to 6 so the rank-5
  vision conv weights in the Qwen3.8 MXFP4 checkpoint (for example
  `model.visual.patch_embed.proj.weight`, `[1152, 3, 2, 16, 16]`) no
  longer reject the whole safetensors map as malformed. Covered by
  `EngineTest.LoadMxFp4Rank5Tensor`. See item 0: the MXFP4 directory
  still does not run end to end.

- 2026-10-08: DFlash2 draft 1+N query layout (237/237 `ctest` on both
  builds). The draft now matches vLLM's `_prepare_dflash_inputs_kernel`:
  the anchor (bonus) token embedding is query 0, the N mask tokens are
  queries 1..N, only the mask rows predict, and the draft context
  excludes the anchor (it is the prefix before it). New
  `spec::QueryEmbeddings` builds the 1+N query block (covered by
  `HybridDecodeTest.QueryEmbeddingsAnchorThenMask`); `GenerateDFlash2`
  runs `block + 1` rows and reads the mask rows at offset 1. Acceptance
  still cannot be judged until the target is the MXFP4 model (item 0).

- 2026-10-08: MXFP4 E2M1 codec fix (242/242 `ctest` on both builds). The
  OCP MX E2M1 nibble `0x7`/`0xF` is `+-6.0`, not NaN; `F4E2M1ToFloat`,
  `Fp32ToF4E2M1Nibble` (max now 6.0), the Vulkan `gemm_mxfp4` shader and
  the ROCm kernel are corrected, and the codec/GEMM tests cover 6.0. This
  was the cause of the all-NaN MXFP4 logits. The Quark MXFP4 on-disk
  layout (low nibble first, `[rows, cols/32]` E8M0 scales, dequant scale
  `2**(byte-127)`, max 6.0) was confirmed against the amd-quark Triton
  reference (`quark/torch/kernel/mx/triton.py`), so the layout matches
  ours and is not the bug. The MXFP4 target now decodes without NaN, but
  the residual hidden still diverges from the GGUF from layer 0 onward
  (layer 7 jumps to about 32000); the SSM tensors were verified correct
  against the GGUF (`dt_bias`, `ssm_a`, `alpha`, `beta`, `conv1d` value
  block reorder), so the divergence is elsewhere (item 0).

- 2026-10-08: MXFP4 coherent output (Gemma RMSNorm unit offset). Qwen3.5
  uses `GemmaRMSNorm`, whose gain is `1 + weight` (vLLM aliases it as
  `Qwen3_5RMSNorm` in `qwen3_5.py`). The GGUF stores the offset already
  applied, but the HuggingFace checkpoint stores the offset, so the plain
  `x * w` kernel left every norm off by one and the residual diverged from
  layer 0 (the "layer 7 explosion" was a downstream symptom, not a bug).
  `WeightConversion` gained `add_one`; the Qwen3.5 module sets it for
  `attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm` and
  `output_norm`. The gated SSM `ssm_norm` and the GGUF path are unchanged.
  The MXFP4 target now matches the GGUF greedy continuation token for token
  and passes 244/244 `ctest` on both builds. Verified with the new
  element-wise tensor dump, which showed the loader's embedding and
  value-head reorder match the GGUF at correlation 0.99.

- 2026-10-08: linear-attention conv on the device (item 1, 246/246
  `ctest` on both builds). A new `conv1d_state` built-in (Vulkan and ROCm)
  assembles the window from the fresh qkv and the history, convolves with
  the reversed tap order, applies the SiLU, writes the q/k/v split
  directly, and shifts the history in place. The history for each linear
  layer now lives in the device buffer `Qwen35State::LinearState::conv_hist`;
  the batched rollback snapshots it with a device copy instead of a host
  copy. Both trunk paths call the kernel, so the host `conv_hist`,
  `conv_mixed` and the five per-layer transfers are gone. A
  device-vs-reference test covers the kernel. Decode fell from 789 ms/token
  to 513 ms/token on the MXFP4 target, and the GGUF from 1100 ms/token to
  492 ms/token, with the same tokens (`11751, 13, 198, 760`). The batched
  path is output preserving: `EngineTest.DFlash2MatchesGreedyOnModel`
  still passes.

- 2026-10-08: DFlash2 acceptance investigation. `EngineTest.DFlash2MatchesGreedyOnModel`
  now runs a realistic 5-token prompt for 8 tokens (it used a single-token
  prompt, which gave the misleading 0 of 24). Acceptance is still only 2 to
  3 of 40 draft tokens (about 6 percent). Two suspects are ruled out: the
  candidate selector (bypassing it and taking the unary top-1 gives the
  same rate) and the draft context window (`TESSERA_DFLASH2_CTX` of 1, 8
  and 32 all agree). So the raw draft mask-row logits are wrong, and the
  fault is in the draft forward or the aux hidden capture.
  Two vLLM mismatches were found and fixed, and the rest of the draft
  forward was verified equal to vLLM stage by stage. First, the prefill
  did not capture the aux target hidden states (it used `DecodeForward`),
  so the draft context only ever held the last prompt position; the
  prefill now captures every position (`DecodeForward` gained capture
  arguments) and `TESSERA_DFLASH2_CTX` can widen the window. Second, the
  draft self-attention is non-causal: the config sets `is_causal: false`,
  which vLLM's `_dflash_layer_causal` uses directly (it overrides the
  per-layer-type default), so the whole block and the whole context are
  visible to every query. The draft config now carries `attn_causal`
  (parsed from `is_causal`, then `dflash_config.causal`, then the layer
  type) and threads it to the draft `attention` call.
  Verified equal to vLLM: the aux capture point (vLLM adds 1 to
  `target_layer_ids` and captures after that layer, so `[5,19,33,47,61]`
  means after layers `[5,19,33,47,61]`, which is what `capture_layer`
  does); the fp8 block dequant; the draft norms (plain, no unit offset);
  the RoPE theta and NeoX style; the `fc` concat layout and the
  layer-major aux order; the grouped dynamic convolution (position = row
  mod block_size, base + delta, prepare side 0 and finish side 1); the
  pre-norm residual structure and the final `norm(hidden + residual)`; the
  context `KV = k/v_proj(rms_norm(fc))` with K-norm and RoPE on K only.
  The head is the target's `output.weight` (using the embedding is worse:
  0 of 56).
  A per-step diagnostic (env `TESSERA_DFLASH2_RANK`, since removed) pinned
  the failure. The draft's first mask row is often exactly right (step 0:
  the target's token is the draft's argmax), but the later mask rows
  collapse to a short repeating pattern (for example `13,0,3,6,3,6,3,6`)
  and are all wrong. The grouped conv is net positive (row 0 drops from
  rank 1 to rank 7 when the conv is bypassed, and step 1 goes from rank 8
  to rank 202377), so the conv is most likely correct; the collapse is
  that every mask row has the same input, so the depthwise conv yields the
  same value for rows with the same tap pattern, and the row differences
  come only from the RoPE positions. The bug is in how the mask rows are
  differentiated (the attention, or the mask/position layout), not in the
  weights. The next step remains a host or vLLM reference of one draft
  step that can compare per-row hidden states.
  The query-row count now matches vLLM: the config's block_size is the
  total block (anchor plus masks), so the draft proposes `block_size - 1`
  masks with `block_size` query rows (the README's block size 8 means 7
  draft tokens); this also removes the conv position wrap on the last row.
  Acceptance is unchanged (2 of 35 over 5 steps).

- 2026-10-08: MXFP4 HF tokenizer. `LoadHfTokenizer` (`src/core/loaders/
  hf_tokenizer.{hpp,cpp}`) parses `tokenizer.json` at MXFP4 load and
  builds the byte-level BPE `Tokenizer`: it inverts `model.vocab`,
  reads `model.merges`, and marks `added_tokens` as control (type 3)
  tokens. `LoadHfChatTemplate` reads the `chat_template` string from
  `tokenizer_config.json`, else `chat_template.jinja`. `Model::Load`
  wires both into the MXFP4 model (previously it passed no tokenizer and
  an empty template). The parsed tokenizer reproduces the GGUF
  reference ids for ASCII, code and CJK
  (`TokenizerTest.RealTokenizerMatchesReference` run against the MXFP4
  target), and the text-only CLI now prompts the MXFP4 target end to end
  (`The capital of France is` gives ` Paris.`). `TokenizerTest.
  HfTokenizerJsonParses`, `HfTokenizerNonBpeIsEmpty` and
   `HfTokenizerRejectsMalformed` cover the parser. 250/250 ctest on
   vulkan and rocm.

- 2026-10-08: **Fixed the OCP FP8 E4M3 codec.** `Fp8E4M3ToFloat` used an
  exponent bias of 8 (`exp - 8`, subnormal `2^-10`) instead of the OCP
  bias of 7 (`exp - 7`, subnormal `2^-9`), so every FP8 value decoded to
  half. The fix lands in all four copies: the core
  `Fp8E4M3ToFloat`, the ROCm `Fp8E4M3ToFloatDev`, and the Vulkan
  `fp8_at` in `gemm_fp8.comp` and `gemm_fp8_block.comp`. The inverse
  `Fp32ToFp8E4M3Bits` (used only by tests) was rewritten for bias 7, and
  `BackendTest.FpCodecsHitKnownPatterns` now asserts the PyTorch values
  (`0x38 -> 1.0`, `0x40 -> 2.0`, `0x76 -> 224`, `0x7E -> 448`). Found by
  comparing the DFlash2 draft's FP8 weights against the raw tensor times
  `weight_scale_inv`: the C++ dequant was exactly half. This is the
  dominant reason the draft draft was poor, though the DFlash2
  acceptance only moved from 2 to 3 of 35, so a second draft bug remains.
  The MXFP4 target (E2M1) is unchanged (`The capital of France is` still
  gives ` Paris.`). 250/250 ctest on vulkan and rocm.

- 2026-10-08: **CLI streams output tokens.** `tessera-cli run` used the
  blocking `Engine::Generate`, which returns only after every decode
  step. The default fills the remaining context (about 4000 tokens),
  so the console showed the prefill line and then nothing for a long
  time. The plain text path now uses `Engine::GenerateStreaming` and
  writes each decoded token to stdout at once with a flush. It also
  logs the planned token budget before the stream starts. The other
  paths (speculative, draft, multimodal) still run to completion and
  then write the full decoded text to stdout. A new
  `EngineTest.GenerateStreamingEmitsIncrementally` test pins the
  streaming contract: the streamed ids match `Generate`, and a false
  return stops the run. 251/251 ctest on vulkan.

- 2026-10-08: **Generation stops at declared stop tokens.** The decode
  loops ran to `max_tokens`, so a chat prompt streamed the answer and then
  kept emitting `<|im_end|><|endoftext|><|im_start|>...` without end. The
  GGUF declares `tokenizer.ggml.eos_token_id` and the MXFP4 target declares
  `eos_token_id` in `generation_config.json`; neither was read. The model
  now exposes `Model::StopTokens()` (parsed by `GgufStopTokens` and
  `LoadHfStopTokens`) and `GenerateOptions` gained `stop_tokens`.
  `GenerateStreaming`, `GenerateMultimodal`, `GenerateSpeculative` and the
  DFlash2 loop all end before emitting a stop token, so the output is the
  answer alone. Verified with the CLI: `Explain gravity in one sentence.`
  stops at token 248046 (`<|im_end|>`) and prints one sentence.
  `EngineTest.GenerateStopsAtStopToken`,
  `EngineTest.LoadGgufModelReadsStopTokens` and the
  `TokenizerTest.LoadHfStopTokens*` cases cover it. 255/255 ctest on
  vulkan and rocm. AGENTS.md gained rule 19 (garbage is a bug) and rule 20
  (probe device and host memory before every command).

- 2026-10-08: **DFlash2 acceptance root-caused to a NaN draft block.**
  With the FP8 codec fixed, the draft forward matches an independent
  Python port stage by stage (attention-conv h1, q/k/v, the attention
  output and o_proj all at correlation 1.0; the context K matches
  exactly). But the draft block's hidden state is **NaN**: at layer 0 the
  activations grow from the token embedding (absmax 0.06) through
  input_layernorm (3.1), the attention conv (28), v_proj (145) and o_proj
  (8695) to a post residual of 21255, and the MLP then overflows. The
  draft logits are garbage, so acceptance is near zero (0 to 3 of 35 on
  the 5-token fixture, and 0 of 49 against the GGUF target too, so it is
  not the MXFP4 aux capture). vLLM runs the same weights without
  exploding (the checkpoint card reports greedy acceptance 4.2 at block
  8), so the port misses a stabilizer; the dynamic per-token FP8
  activation quantization the card names is the prime candidate. Also
  fixed a real CLI bug: `--draft-block` defaulted to 4, but the
  checkpoint is trained for block 8; it now defaults to 0 ("use the
  checkpoint's block", accepted by the strategy as a sentinel). Tests
  can target a free GPU with `TESSERA_TEST_GPU` because a vLLM server was
  holding device 0 and caused a GPU context loss. 255/255 ctest on
  vulkan.

- 2026-10-08: **DFlash2: the FP8 draft block itself is numerically
  unstable.** A faithful reference exists at
  `~/git/radiance-vllm-mxfp4/paroquant/drafter/train_drafter.py` (plain
  Torch; the same equations as vLLM). Running it on the dumped inputs with
  the `Qwen3.8-27B-DFlash2-FP8` weights reproduces the same behaviour as
  the C++: the block's activations grow layer over layer (layer 0 reaches
  ~1.6e6 from a 0.06-magnitude input) and intermittently overflow to
  NaN/inf, so the logits are garbage and acceptance is near zero. The C++
  and the reference agree to correlation 0.99999 on q/k/v, so the port is
  faithful and the checkpoint/quantization is the variable. The C++
  dequantizes the FP8 blocks to normal magnitudes (rms ~0.1), and the
  channel layout is the plain 128x128 block scale, not an AWQ fold. Next:
  try the alternate draft checkpoints (`~/models/malicz/...-DFlash2-FP8`,
  or the radiance MXFP4 drafter from `quantize_dflash_mxfp4.py`) and, if
  they are stable, treat the `tcclaviger` FP8 quant as bad; if all are
  unstable, find the stabilizer vLLM applies (the dynamic per-token FP8
  activation quantization is the remaining candidate). Temporary
  diagnostic dumps were used to reach this and removed; the run is
  nondeterministic once the block NaNs, so comparisons must capture the
  inputs and the output in one run.

- 2026-10-08: **DFlash2 correction: the draft forward is faithful and
  numerically stable.** The two entries above are wrong. The "NaN draft
  block" was an artifact of a bug in the throwaway Python reference: its
  `fc` projection used `W @ x.t()` instead of `x @ W.t()`, so the context
  was the transpose and the forward "exploded" with a huge/NaN hidden.
  With the reference fixed to mirror radiance's `train_drafter.py`
  (`~/git/radiance-vllm-mxfp4/paroquant/drafter/train_drafter.py`) exactly,
  the C++ draft forward matches the reference at correlation 0.99976 on
  the same inputs: reference absmax 31.5, C++ absmax 31.06, no NaN. The
  C++ draft hidden is finite and stable across runs (absmax about 31, no
  NaN or inf). Also verified this session: the query embeddings are
  byte-exact with the target `embed_tokens` (row 248070 matches
  `model.language_model.embed_tokens.weight`; rows 1..7 are the identical
  mask row, row 0 the anchor), the context is the target hidden at the
  capture layers, `RunFullBlock`/`RunFfn` both add into `h.x` before the
  capture, and the context window is used (widening it changes the
  acceptance: `TESSERA_DFLASH2_CTX=64` gives 1 of 42). Shifting the
  capture to layers `target_layer_ids - 1` (vLLM's `_maybe_add_hidden_state`
  captures after 0-based layer `T-1`, see `qwen3_next.py`) does not change
  the acceptance either (still 3 of 35). The acceptance is 3 of 35 on the
  fixture. The remaining suspects are the aux hidden **values** (which
  need an independent target forward to compare) and the FP8 draft
  weights; both targets (MXFP4 and GGUF) give the same low acceptance, and
  the `fc` orientation fix confirms the port itself is correct. Temporary
  dumps removed; 255/255 ctest.

## Next (in order)

- **PERF (DEFERRED)**: make MXFP4 inference fast. Targets: the whole load
  under 60 s (met, about 54 s), and 35 to 40 tokens/s decode without MTP.
  Count the decode rate on generation only: the model load is a one-time
  cost and is not part of tokens/s.
  Deferred for now, because the speed is gated by the outstanding to-do
  items, not by one knob. The causal conv is now on the device (done), but
  the norms and SiLU still run on the host (item 2) and the GEMM throughput
  is item 3. Each of those adds synchronous submit-and-fence round-trips or
  host work. Land them first, then re-measure and do the remaining PERF
  work.
  Context: the load reads the
  19 GB safetensors, dequantizes and packs every fp4 blob, and converts
  large BF16 tensors to F32 (the `lm_head` alone becomes a 5 GB F32
  matrix). The reference runtime loads the same file in 8 s because it
  keeps fp4 and dequantizes in the kernel. Measure on
  `EngineTest.MxFp4GeneratesWhenProvided` and
  `EngineTest.DFlash2MatchesGreedyOnModel`.
  Progress: the load is about 54 s now, under the 60 s target, and
  `EngineTest.MxFp4GeneratesWhenProvided` fell from 287 s to 68 s. The
  changes: pack blob||scales in one allocation (appending reallocated the
  whole blob for every tensor, 180 s); copy the value-head reorder in runs;
  memory-map the checkpoint instead of `ReadFile` (removed an 18.6 s copy
  and the 18 GB anonymous allocation); vectorize the bulk BF16-to-F32
  conversion; upload the F32 tensor without an extra copy; skip the second
  fp4 copy when there is no reorder.
  Decode fell from about 2.6 s/token to 789 ms/token (and prefill from
  5.5 s to 1.8 s) with the same output. Two changes did it. `gemm_mxfp4`
  now decodes a 16-entry nibble table and loads one 32-bit word (eight
  nibbles) per eight elements instead of one byte per two; a per-kernel
  pass had shown this kernel was 5.9 s of the ~7.3 s fence-wait total, and
  the earlier version of this change was committed but never recompiled,
  so it read the wrong word offset. `rmsnorm` now uses one workgroup per
  row with a shared-memory reduction instead of one thread per row (the
  old form serialized a 5120-element row on a single thread). The
  remaining cost is no longer the GEMV arithmetic. A per-kernel pass put
  the total fence wait at about 1.4 s over the run (gemm_mxfp4 870 ms) while
  decode alone is about 3.2 s, so most of a token is the per-op submit and
  fence round-trip: on the order of 700 copies and 500 kernel launches per
  forward, each blocking on its own fence. Reusing the command buffer and
  fence in `LaunchKernel` did not change the time, so it is the latency of
  the synchronous submit, not the allocation. The fixes are to batch a
  step's ops into one command buffer and wait once, and to remove ops. The
  device conv (done) removed the per-layer host round-trip and took decode
  from 789 ms to 513 ms/token (and the GGUF from 1100 ms to 492 ms). A
  tiled or split-K GEMV would then help the remaining kernel time.

0. **DFlash2**: runs end to end (`Engine::GenerateDraft`, CLI `--draft`)
   and output equals greedy. Batching and the draft context width are
   done (see the Done entry). Corrected 2026-10-08: the draft block is
   NOT NaN and NOT unstable. The forward is faithful to radiance's
   `train_drafter.py` (correlation 0.99976) and stable (absmax ~31); the
   query embeddings are byte-exact with the target. The low acceptance is
   therefore not the block math; the remaining suspects are the aux hidden
   values and the FP8 draft weights (see the correction Done entry).
   Note: DFlash2 has a slower boot (it loads
   the 5-layer draft) and each step costs more than one greedy step, so it
   is only a win at a high enough acceptance; the target is steady-state
   tokens/s, not the first few tokens. Measure tokens/s over a long
   generation. Measured on the 27B over 32 tokens: greedy 0.86 s/token,
   DFlash2 about 4.8 s/token (acceptance 5 of 108) even at steady state,
   so the draft quality, not the boot, is the blocker. The draft accepts
   about 0 to 15 percent. Align it with
   vLLM's DFlash handling (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py`
   and `.../dflash2/speculator.py`, `vllm/model_executor/models/qwen3_dflash.py`
   and `qwen3_dflash2.py`): the draft block has a 1+N query layout (query 0
   is the bonus/anchor token, queries 1..N are mask tokens, and only the
   mask positions predict; `num_query_per_req = 1 + num_steps`,
   `is_bonus = query_off == 0`, `sample_off = 1`), the mask id is
   `dflash_config.mask_token_id`, and the draft context K/V is the target
   hidden at the context positions. Exact vLLM layout
   (`_prepare_dflash_inputs_kernel`): the context is the prefix up to the
   anchor (`last_valid_pos`), it does NOT include the bonus; query 0 is
   the bonus token at `P+1` (`input_id = bonus_token`), queries 1..N are
   mask tokens at `P+2..`; only masks predict (`sample_off=1`). The
   context K/V is `KV_proj(rms_norm(fc(concat of aux target-layer
   hiddens)))` (`_project_context_kv`, `_get_dflash_fc_input_size` =
   target_hidden_size * num_aux_layers). The 1+N layout and the
   anchor-out-of-context rule are implemented (see the 2026-10-08 Done
   entries). The Qwen3.8 27B MXFP4 target
   (`~/models/Qwen3.8-27B-MXFP4-MTPFP8/`) now loads and decodes end to
   end through the Qwen3.5 module: the `Architecture` hooks
   `ParseConfigJson`, `MapWeightName` and `ConvertWeight` read config.json,
   rename the HuggingFace tensors, and apply the value-head reorder
   (HF value head `p = f + 3*kh` goes to internal row `16*(p%3) + p//3`:
   `dt_bias`, `A_log` (then `-exp` to F32 `ssm_a`), `in_proj_a/b`,
   `in_proj_z`, the value block of `in_proj_qkv`/`conv1d` and the input
   of `ssm_out` all reorder); the loader packs each F4E2M1 blob with its
   E8M0 scale. Acceptance can now be measured against the correct target.
   The garbled output was the Gemma RMSNorm unit offset. Qwen3.5 uses
   `GemmaRMSNorm`, whose effective gain is `1 + weight`
   (`vllm/model_executor/layers/layernorm.py` and
   `vllm/model_executor/models/qwen3_5.py`, which aliases it as
   `Qwen3_5RMSNorm`). The GGUF bakes the offset into its norm tensors, so
   the plain `x * w` kernel was correct there, but the HF checkpoint
   stores the offset, so every norm was off by one and the hidden state
   diverged from the first layer. `WeightConversion` gained `add_one` and
   the Qwen3.5 module sets it for `attn_norm`, `post_attention_norm`,
   `attn_q_norm`, `attn_k_norm` and `output_norm` (the gated SSM `ssm_norm`
   uses the plain weight, and the GGUF path does not run this hook, so it
   is unchanged). With the fix the MXFP4 target is coherent and its greedy
   continuation matches the GGUF reference token for token (`Paris` `.`
   newline newline `The`). A per-tensor element-wise dump
   (`EngineTest.DumpTensorWhenProvided`, env `TESSERA_DUMP_MODEL`,
   `TESSERA_DUMP_TENSOR`, `TESSERA_DUMP_OUT`) confirmed the loader: the
   embedding and the linear-attention value-head reorder match the GGUF to
   correlation 0.99, so the layout and permutation are right. Acceptance
   can now be measured against the correct target.
   One gap remains: the MXFP4 GEMM is very slow (about 2.5 s/token on the
   27B, versus 0.9 s/token for the GGUF; the `gemm_mxfp4` kernel needs
   item 3 work). The `gemm_mxfp4` kernel now reads the scale from the
   packed weight tail, so its contract, host reference and device test
   changed.
   The HF tokenizer is now parsed. vLLM never derives the tokenizer from
   the weights: `get_tokenizer` (`vllm/tokenizers/registry.py`) calls
   `AutoTokenizer.from_pretrained` on the model repo path, so the
   tokenizer comes from the checkpoint directory; Quark's
   `--model_export hf_format` carries the source tokenizer files next to
   the quantized safetensors, which the local MXFP4 target has
   (`tokenizer.json`, `tokenizer_config.json`, `vocab.json`,
   `merges.txt`, `chat_template.jinja`). `LoadHfTokenizer`
   (`src/core/loaders/hf_tokenizer.cpp`) reads `tokenizer.json` at MXFP4
   load and builds the byte-level BPE tokenizer: it inverts `model.vocab`
   (byte-level tokens to ids), reads `model.merges` ("left right" in rank
   order) and marks `added_tokens` as control tokens. `LoadHfChatTemplate`
   reads the `chat_template` string from `tokenizer_config.json`, else
   `chat_template.jinja`. Weight loading is untouched. The tokenizer
   covers ids 0..248076 (248044 BPE entries plus 33 added tokens); the
   head is 248320 wide, so the last 243 rows are unused padding.
   `TokenizerTest.RealTokenizerMatchesReference` now passes against the
   MXFP4 target (the same ids as the GGUF reference across ASCII, code
   and CJK), and the text-only CLI prompts the MXFP4 target end to end
   (`The capital of France is` gives ` Paris.`).
- **Hybrid SSM device path (done)**: the causal conv1d, the SiLU and the
  q/k/v split now run on the device (`conv1d_state`), with the history in
  `linear[l].conv_hist`, in both trunk paths. See the Done entry.
2. **MoE, MLP, RMSNorm and embedding kernels** as the Qwen 3.8
   definition needs them. The generic RMSNorm kernel is done. The
   decode loop runs projections, RoPE and attention on the device.
   Norms and SiLU still run on the host. Architecture specific behavior
   moves behind the `Architecture` module interface (see item 0).
3. **GEMM throughput**: the model runs far below memory bandwidth
   (about 20 GB/s of 16 GB weights per 0.76 s/step), so the GEMM kernels
   are bound by the per-element byte-wise weight reads, not by the
   multiply-accumulate. Two changes: (a) read and dequantize a weight
   block once and apply it to a tile of rows (the batched prefill
   dequantizes each weight m times for an m-token batch); (b) vectorize
   the byte reads (read 4 bytes at a time and extract). Applies to
   gemm_q4k/q5k/q6k/q3k/q8_0/iq4xs/iq4nl/fp8 on both backends. The
   attention_q8/attention_q4 kernels still recompute the dot product per
   output dimension.
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
   (`--gpu`) are CLI flags now, and the KV cache can be fp16 (`--kv-f16`)
   (--kv-q8) or 4-bit (`--kv-q4`). Still to wire: mmproj path for vision
   input and batch caps (features that do not exist yet). No hard-coded
   paths or sizes.
6. **Prefill optimisation**: make the batched prefill faster. Each GEMM
   still dequantizes each weight once per prompt row. Read each weight
   block once and reuse it across a tile of rows. Keep one sync per
   prefill and preallocate the KV caches up front. Measure prompt
   tokens per second on a text prompt and on an image prompt.
7. **Multimodal (mmproj)**: config, weights, encoder+merger, image
   load/resize, image-embedding injection, the CLI wiring and the
   `<|image_pad|>` placeholder default are done
   (`Engine::GenerateMultimodal`). Still to do: deepstack feature
   injection and the image prefill speed (attention kernel).

8. **Multi-GPU (deferred)**: today `--gpu` selects one device and there is
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
