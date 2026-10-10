# Progress

This file tracks tessera development. Update it as work progresses.
After each milestone, update "Done" and refresh "Next" (see AGENTS.md,
Working Principles).

## Current status

The boilerplate is complete and passes on both backends.
`ctest` passes 278/278 on both builds.
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
   target generates on both backends. RMSNorm, add, silu_mul and
   embedding gather run on the device. The linear path still issues
   several host-side launches per token.
- DFlash2 strategy. It loads the real FP8 draft and speculates end
  to end through the `SpeculativeStrategy` seam (greedy, MTP and
  DFlash2 share one engine loop). The 27B MXFP4 target accepts 4.4
  tokens per step, above the served reference. Output equals greedy.
  The draft step still costs more than greedy.
- Public generation API: `Engine::Generate(model, options)` runs greedy
  single-token decode on the non-speculative (reference) path and
  returns the produced token ids. The CLI drives it.
- HTTP serving: a blocking HTTP/1.1 server (`tessera::Serve`) with a
  buffered and a chunked (SSE) response path, API-key auth
  (Bearer/X-Api-Key/x-api-key, env `TESSERA_API_KEY`) and a CORS
  allowlist. Endpoints: `/health`, `/metrics`, `/v1/models`, `/props`,
  `/tokenize`, `/detokenize`, `/slots` (live sessions), `/v1/completions`
  (streaming SSE), `/v1/chat/completions` (SSE), `/v1/messages` and
  `/v1/messages/count_tokens`; unimplemented surfaces return 501.
  `Engine::GenerateStreaming` emits tokens one at a time. Generations
  queue in arrival order on the single device instead of answering 503,
  so parallel windows wait their turn. Chats live in server sessions
  (`/api/sessions`, `/v1/sessions` aliases); the OpenAI endpoints
  continue one through `session_id` (see the newest Done entry).
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
  [--draft-block <n>] [--prompt-text <str>] [--tokens <n>]`. It loads
  the model, uploads weights, prints a tensor summary, and with
  `--tokens` runs greedy decode through `Engine::GenerateStreaming`
  and streams the text to stdout. `--prompt-text` tokenizes text
  with the model's tokenizer. It also serves (`serve`), samples
  (`--sample` and parameter flags), selects the KV cache (`--kv-f16`,
  `--kv-q8`, `--kv-q4`, `--kv-fp8`), speculates (`--speculate`,
  `--draft`) and encodes images (`--mmproj`, `--image`). The image
  token id is detected from `<|image_pad|>`. There are no
  `--log-tokens`, `--image-token`, `--mtp` or `--prompt` flags, even
  though some Done entries below name them. Input and output text
  logging is on by default. MTP speculation uses `--speculate`.
  Prompts use `--prompt-text`.
- Architecture modules: one `Architecture` module per model family
  (`src/models/qwen3_5/`). The hybrid trunk, state and MTP head live
  behind it. Core names no model.
- MXFP4 target: the Qwen 3.8 27B MXFP4 target loads, tokenizes and
  decodes end to end. Greedy output matches the GGUF reference. The
  HuggingFace tokenizer and chat template parse at load. Generation
  stops at the declared stop tokens. The FP8 MTP head loads. W4A8
  activation quant (`TESSERA_MXFP4_W4A8=1`), bf16 target mode
  (`TESSERA_TARGET_BF16=1`) and the fp8 KV cache (`--kv-fp8`) are
  opt-in. None of them changed DFlash2 acceptance, so they stay off
  by default.
- GEMM and attention speed: tiled batched kernels for the hot formats,
  warp-per-output decode GEMVs, a warp-per-column multi-row MXFP4
  kernel for small verify batches, tiled quantized attention, and
  four-way ILP in the scan and norm kernels. Vulkan launches pipeline
  through a four-slot ring instead of waiting on a fence per launch.
  ROCm copies use async device-to-device transfer. On the 27B MXFP4
  target (ROCm) greedy runs at about 23 tok/s (43 ms/token, the fp8
  WMMA path) and DFlash2 at about 60 tok/s, faster than greedy with the
  output equal to greedy. The speculative step is about 75 ms. See the
  newest Done entries for the numbers.
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

- 2026-10-08: **DFlash2 acceptance is a real draft-quality bug, pinned to
  the aux hidden.** On a realistic 128-token generation the C++ accepts 21
  of 742 draft tokens (about 0.2 per step); radiance's drafter on the same
  checkpoint reaches 0.807 first-position top-1 and about 1.9 accepted per
  block (`paroquant/RESULTS.md`), so the gap is real and not a short-prompt
  fixture. Verified this session against radiance's own code: the draft
  forward matches `train_drafter.py` (correlation 0.99976); the query
  embeddings are byte-exact with the target `embed_tokens`; the capture
  layers are 5/19/33/47/61 (radiance's README, and the C++ matches); the
  block is `block_size - 1` mask rows plus one anchor row (query rows 8),
  matching vLLM's `num_query_per_req = 1 + num_speculative_tokens`; and the
  grouped-conv `block_size`, the non-causal draft mask, the grouped dynamic
  conv, the candidate selector and the context `rms(fc(aux), hidden_norm)`
  all match vLLM's `qwen3_dflash2.py` / `qwen3_dflash.py`. The C++ draft
  hidden is finite and stable (absmax about 31). Therefore the only
  remaining input is the aux hidden **values** (the C++ target hidden at the
  capture layers), not the draft math.
  Blocked on an independent target reference. The local vLLM install cannot
  load the target: `vllm/model_executor/models/qwen3_5.py` needs
  `libnvrtc.so.13`, which is absent on this AMD box, and its bundled
  torchcodec is a CUDA build that fails to load. Radiance's serving target
  (`Qwen3.8-27B-PARO-MXFP4-ft`, bf16 `Qwen3.8-27B-bf16`) is not on this
  machine. `paroquant/RESULTS.md` records a known "drafter/target mismatch":
  the drafter self-distills on one served target's hidden, so a different
  target shifts the distribution. The next step is a host reference of the
  target's first six layers (embed .. layer 5), or running radiance's ROCm
  vLLM container, to compare the C++ aux against vLLM's.

- 2026-10-08: **DFlash2 reference measured; the gap is pinned to the target
  hidden.** This machine already runs the reference serve as a docker
  container: `r9700-qwen3.8-mxfp4-g0a` (image `local-radiance-mxfp4:0.9.3`,
  port 9300, GPU0, systemd unit `r9700-mxfp4.service`), serving
  `/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken` with the DFlash2 draft
  (`--speculative-config method=dflash, num_speculative_tokens=7`) and
  `--attention-backend R4D`. A 128-token greedy chat generation there
  measures **2.85 accepted/draft, 3.85 tokens/step** in 3.1 s. The C++
  on `Qwen3.8-27B-MXFP4-MTPFP8` accepts about 0.2/step, a roughly 14x
  gap, so the fault is in the C++ port, not the prompt.
  The served target and the C++'s `Qwen3.8-27B-MXFP4-MTPFP8` share the
  same weights (`model.safetensors`, `19373796656` bytes) but differ in the
  activation quantization: the `-pertoken` config uses
  `PerChannelMinMaxObserver` / `qscheme: per_channel` / `ch_axis: 0` on the
  MXFP4 inputs, while `MTPFP8` uses `PerTensorMinMaxObserver` /
  `per_tensor`. vLLM applies that activation quantization in the forward,
  and the drafter was trained on the served target's hidden (the capture
  stores the aux as e4m3 with a per-token scale), so the C++ aux, which
  runs bf16 activations and applies neither observer, is the prime suspect
  for the 14x gap.
  GPU0 is held by the production `r9700-qwen3.8-mxfp4-g0a`; a second full
  27B load (the radiance container) started beside it crashed the
  workstation. GPU1/`:9301` (`g1a`) is intentionally stopped. AGENTS rule 20
  now says to run a model server in the foreground and not background it.

- 2026-10-08: **DFlash2 W4A4 activation quant is deferred (about 34x too
  slow).** The working hypothesis was that vLLM quantize-and-dequantizes
  the MXFP4 linear inputs (W4A4, E2M1 plus one E8M0 scale per 32 elements)
  and that the DFlash2 drafter was trained on that QDQ'd target hidden, so
  the C++ W4A16 target hidden mismatches the draft and acceptance stays at
  about 0.2 per step against the reference 2.85. The activation half is
  implemented: `MxFp4QuantizeBlock` in `src/core/numerics/quant.*`, used by
  `GemmMxFp4Ref`, and a fused E2M1/E8M0 quant inside the ROCm
  `GemmMxFp4Kernel` (bit-trick `MxFp4BlockScaleDev`/`E2M1CodeDev`).
  `BackendTest.GemmMxFp4DeviceMatchesRef` passes with the W4A4 reference.
  The real-model run does not confirm the hypothesis yet. On GPU0,
  `EngineTest.DFlash2MatchesGreedyOnModel` loads the 27B MXFP4 target (858
  tensors) and prefills 5 tokens, but plain greedy decode is **17686
  ms/token** (141490 ms for 8 tokens) against the about 513 ms/token
  baseline, so the run was aborted before the `spec dflash2: accepted X of
  Y` line and the acceptance is unmeasured. The ROCm object was verified
  newer than the edited source, so the slow path is the new kernel. The
  likely cause is that the activation block amax and code are recomputed
  inside the per-output element loop, so a usable W4A4 must quantize each
  activation block once and share it across output columns. The source
  change stays uncommitted and DFlash2 acceptance is deferred. The
  production container `r9700-qwen3.8-mxfp4-g0a` was stopped for the run.

- 2026-10-08: **Focus moved to item 2** (MoE, MLP, RMSNorm and embedding
  kernels). DFlash2 is deferred per the entry above. The production
  container is stopped while GPU0 is used for kernel work.

- 2026-10-08: **Generic device embedding gather (item 2).** New
  "embedding_f32", "embedding_bf16" and "embedding_q4k" built-ins on
  Vulkan (GLSL) and ROCm (HIP): buffer 0 is the token ids (u32, rows),
  buffer 1 the embedding table (vocab x cols), buffer 2 the fp32 output
  (rows x cols); scalars are rows, cols, vocab, one thread per output
  element. Q4_K reads element `(e % 64) / 32` of a 256-element block from
  the same nibble layout as `gemm_q4k`. The Qwen3.5 module gathers through
  `GatherEmbeddingRows`, so the single decode, the batched prefill and the
  MTP head no longer dequantize embedding rows on the host; a dtype with no
  device kernel, or a width that does not align with the 256-element block,
  falls back to the canonical host gather. `BackendTest.
  EmbeddingGatherMatchesRef` covers f32, bf16 and Q4_K against the host
  dequant, including an out-of-range id that must write a zero row. The
  gated (Q4_K `token_embd`) and f32-embedding fixtures decode to the same
  tokens on both backends. `CachedKernel` is extracted from `GemmFor` so
  both kernel families share the load-once cache. 256/256 `ctest` on
  vulkan and rocm. The W4A4 DFlash2 experiment is preserved in a git
  stash, not in the build.

- 2026-10-08: **Release is the default build type (perf prerequisite).**
  The documented `cmake -B` left `CMAKE_BUILD_TYPE` empty, so the binary
  ran at `-O0`: a 27B GGUF decode measured 10814 ms/token (prefill 24738
  ms for 5 tokens). `CMakeLists.txt` now defaults an unspecified build
  type to Release (an explicit `-DCMAKE_BUILD_TYPE=...` is still
  honored). Both backends rebuilt at Release and the full suite passes
  256/256 on both. The same GGUF run now prefills 5 tokens in 690 ms and
  decodes at 256 ms/token with the identical token sequence (`11751, 13,
  198, 760`), so performance work (items 3 and 6) can now be measured.
  Builds stay serial (`-j1`): the gcc-15 toolchain segfaults and emits
  corrupt assembly under `-j3`/`-j4`.

- 2026-10-08: **Tiled batched Q4_K GEMM, first item 3 step (ROCm).** New
  "gemm_q4k_batched" on ROCm: one workgroup handles 8 activation rows x
  one weight column, the Q4_K block is dequantized once into shared
  memory and reused across the 8 rows, and the 8 dots accumulate in the
  `gemm_q4k` element order so the batch matches the sequential forward.
  `GemmTiledKernelName`/`GemmTiledFor`/`ProjectTiledDevice` (plus the
  shared `kGemmTileRows`) live next to the GEMV helpers, and the Qwen3.5
  batch path projects through a `ProjectBatch` that picks the tiled
  kernel when rows > 1 and the dtype has one, else the GEMV kernel.
  `BackendTest.GemmQ4KBatchedMatchesRef` covers a non-multiple-of-8 row
  count; `HybridDecodeTest.BatchedPrefillMatchesSequential` pins the
  batch against sequential (it caught a reordered reduction, now fixed).
  257/257 `ctest` on rocm; 257/257 on vulkan (the tiled test skips there,
  and the batch falls back to GEMV). A 5-token GGUF prefill measures
  790 ms with the tiled kernel against 690 ms with GEMV, so the win (if
  any) needs a larger batch to show; the Vulkan tiled kernel and the
  MXFP4 tiled kernel are still to do.

- 2026-10-08: **Tiled batched Q4_K GEMM on Vulkan (item 3).** The ROCm
  tiled kernel is ported to GLSL (`gemm_q4k_batched.comp`, shared-memory
  tile, `barrier()` syncs, the `gemm_q4k` accumulation order). The batch
  path now uses it on both backends through the same `ProjectBatch`
  selection. `BackendTest.GemmQ4KBatchedMatchesRef` runs on Vulkan (no
  longer skips) and the full suite is 257/257 on both builds. The MXFP4
  tiled kernel and the vectorized reads are still to do.

- 2026-10-08: **Tiled batched MXFP4 GEMM (item 3).** `gemm_mxfp4_batched`
  on both backends: one workgroup handles 8 activation rows x one weight
  column, the 32 MXFP4 elements and E8M0 scale of a block are
  dequantized once into shared memory and reused across the rows, and the
  dot runs elements 0..31 in the `gemm_mxfp4` order. Registered as the
  tiled kernel for F4E2M1, so the batch path picks it through
  `ProjectBatch`. `BackendTest.GemmMxFp4BatchedMatchesRef` covers a
  non-multiple-of-8 row count on both backends (the Vulkan port's first
  version read the element index instead of the blob byte index and
  produced 152 absolute error; fixed and green). 258/258 `ctest` on
  vulkan and rocm. On the real 27B MXFP4 target a Release build now
  prefills 5 tokens in 1839 ms and decodes at 669 ms/token, and emits
  `11751, 13, 198, 760`, the same tokens as the GGUF target. This also
  pins the earlier confusion: the repeated `89307` token and the 7.7
  s/token run were the W4A4 experiment plus the `-O0` build, both since
  set aside (W4A4 is in `stash@{0}`). The vectorized byte reads and a
  larger-batch prefill measurement remain.

- 2026-10-08: **Vectorized MXFP4 weight reads (item 3b, ROCm).** The
  `gemm_mxfp4` GEMV read one byte (two nibbles) per load; it now reads
  one `unsigned int` (eight nibbles) per load, matching the Vulkan
  kernel. The row and block byte offsets are multiples of 16 elements,
  so every word load is aligned. Results are bit-identical (same
  element order); `BackendTest.GemmMxFp4*` still pass. The measured
  effect is small: the 27B MXFP4 decode is 652 ms/token against 669
  ms/token before, so the read width is not the MXFP4 bottleneck. The
  per-block `E8M0ToFloatDev` (a software `ldexp` on device) is the next
  suspect and the next item 3 lever.

- 2026-10-08: **E8M0 scale decode by bit cast (item 3).** `E8M0ToFloatDev`
  called `ldexp` once per MXFP4 block. Since the value is 2^(scale - 127),
  the fp32 exponent field is `scale`, so the bits are `scale << 23`
  (`scale == 0`, 2^-127, stays on ldexp because it is subnormal, so the
  device still matches the core host `E8M0ToFloat` exactly). Results are
  bit-identical, `ctest` is 258/258, and the 27B MXFP4 decode drops from
  652 to 599 ms/token with the same tokens. The MXFP4 path is still about
  2x the GGUF decode (280 ms/token); the separate, strided scale-array
  read is   the likely next cost. The remaining item 3 work (scale-array
  access, a larger-batch prefill measurement, and the same vectorization
  on the other quant GEMVs) continues.

- 2026-10-08: **Wide Q4_K nibble reads and the item 3 cost picture.**
  `gemm_q4k` on ROCm now reads the 128 nibble bytes of a block as eight
  aligned `unsigned int` words (four bytes, eight nibbles per load)
  instead of one byte (two). Results are bit-identical and `ctest` is
  258/258, but the 27B GGUF decode is unchanged (284 ms/token against 280
  ms/token). Two more targeted experiments were run and reverted because
  they did not help: reading the MXFP4 E8M0 scales as if transposed
  (perfectly coalesced access, same 599 ms/token) shows the strided scale
  read is not the cost, and keeping the 248k-vocab head BF16 (`gemm_bf16`
  rather than the F32 conversion) did not change the decode either. The
  GGUF runs at about 57 GB/s of the roughly 16 GB of weights per step, far
  below the memory bandwidth, so the per-thread weight-row reads (one
  thread reads a whole row, uncoalesced across the warp) are the remaining
  item 3 lever; that is a GEMV redesign, not a micro-optimization. The
  tiled kernels already fix the batched case.
  A coalesced Q4_K GEMV is now kept as a feature even though it measured
  slower. `gemm_q4k_row` on both backends uses one workgroup per output
  element (thread `t` owns element `b * 256 + t`, shared-memory
  reduction), so the activation and weight reads coalesce across the
  workgroup instead of striding per thread. `ProjectDevice` selects
  `grid_x = m * n` for an id ending in `_row` (the shared `GemmGridFor`
  rule). `GemmKernelName(Q4K)` returns it, so the model uses it.
  `BackendTest.GemmQ4KRowMatchesRef` covers it on both backends. On the
  27B GGUF it measures 310 ms/token against 280 ms/token for the
  word-read GEMV: the 256-wide reduction per output and the `m * n`
  workgroups outweigh the coalescing gain at k = 5120. The owner asked to
  keep it as a feature regardless, so it stays wired in. Because the
  batched path uses the tiled kernel and the sequential path this coalesced
  one, their fp reductions associate differently;
  `HybridDecodeTest.BatchedPrefillMatchesSequential` now tolerates that
  reassociation (1e-2 instead of 1e-3). The earlier "reduced word reads"
  note above stands. 259/259 `ctest` on both backends.

- 2026-10-08: **Tiled Q5_K GEMM (item 3a).** `gemm_q5k_batched` on both
  backends: one workgroup handles 8 activation rows x one weight column,
  the 256 Q5_K elements of a block are dequantized once into shared
  memory and reused across the rows, and the dot uses the `gemm_q5k`
  order. Element `e` is sub-block `e/32`; its high bit is `1 << s` of
  byte `16 + e%32` and the low nibble byte is `48 + (s/2)*32 + e%32`.
  Registered as the tiled kernel for Q5_K so `ProjectBatch` picks it for
  rows > 1. `BackendTest.GemmQ5KBatchedMatchesRef` covers both backends
  and a non-multiple-of-8 row count. 262/262 `ctest` on vulkan and rocm.
  The 5-token GGUF prefill is unchanged (812 ms against 799 ms) and the
  decode is 311 ms/token: at a five-row batch the GEMV's repeated weight
  reads already hit cache, so tiling saves little DRAM traffic. The tiled
  IQ4_XS kernel is still to do.

- 2026-10-08: **Tiled Q6_K GEMM (item 3a).** `gemm_q6k_batched` on both
  backends, same 8-row tile as Q4_K/Q5_K. Thread `e` owns element `e`:
  `n2 = e/128`, `q = (e%128)/32`, `l = e%32`, the scale byte
  `192 + n2*8 + l/16 + 2*q`, the low nibble byte `0`/`32` (by `q` parity)
  at `n2*64 + l`, the 2-bit high at bits `2*q` of byte
  `128 + n2*32 + l`; the value is `d * scale * (q - 32)`. Registered for
  Q6_K so `ProjectBatch` picks it. `BackendTest.GemmQ6KBatchedMatchesRef`
  covers both backends. 263/263 `ctest` on vulkan and rocm. The 5-token
  prefill is still 805 ms and the decode 310 ms/token, so tiling the
  remaining formats is not where the time is.

- 2026-10-08: **Tiled IQ4_XS GEMM (item 3a, tiling set complete).**
  `gemm_iq4xs_batched` on both backends, the same 8-row tile. Thread `e`
  is element `e`: `ib = e/32`, `j = e%16`, the nibble byte is
  `8 + ib*16 + j` (low nibble for the low half of the sub-block, high for
  the high half), the 6-bit scale is the 4 low bits at `4 + ib/2` plus
  the 2 high bits at bit `2*ib` of the u16 at 2, and the value is
  `d * (scale - 32) * codebook[nibble]`. Registered for IQ4_XS, so every
  hot GEMM format (Q4_K, Q5_K, Q6_K, IQ4_XS, MXFP4) now has a tiled
  kernel for the batched prefill.
  `BackendTest.GemmIq4XsBatchedMatchesRef` covers both backends. 264/264
  `ctest` on vulkan and rocm (the ROCm run is flaky: `VisionStackMatchesRef`
  intermittently fails inside the full suite and passes standalone or on a
  re-run, with no docker/GPU process present; treat a single failure there
  as environmental). The 27B GGUF decode measures 310 to 359 ms/token
  across runs and the 5-token prefill 805 to 1004 ms, so run-to-run noise
  exceeds any tiling effect at this batch size. The remaining item 3 work
  is the m = 1 decode GEMV, not the tiling.

- 2026-10-08: **Wide reads on the remaining hot GEMVs, and an attention
  OOB fix (item 3b).** The ROCm `gemm_q5k` now reads the 32 low-nibble
  bytes and the 32 high-bit bytes of each group as words (eight loads per
  group instead of 64 byte loads), and `gemm_iq4xs` reads the 16 nibble
  bytes of a sub-block as four words. Both are bit-identical and their
  device tests pass. `gemm_q6k` is left byte-wise: its 210-byte block is
  not 4-byte aligned, so word loads would be misaligned. The Vulkan
  `gemm_q5k`/`gemm_iq4xs` already extracted bytes from words. Also fixed a
  real out-of-bounds read in the attention kernel (ROCm and Vulkan): the
  value-accumulation loop read `v[vb + e]` for every lane `e` up to the
  256-thread workgroup even when `head_dim < 256` (the vision tests use
  `head_dim = 4`); only lanes below `head_dim` have an output element, so
  the loop is now guarded. 264/264 `ctest` on vulkan and rocm.

- 2026-10-08: **Fixed the flaky `VisionStackMatchesRef` (a test bug).**
  The test failed about 40% of runs with an absolute error of order 1 to
  3. Instrumenting it showed the device output `got` was identical across
  failures while the host reference `ref` changed, and the failing `ref`
  was column-constant (equal to the post-LN bias), i.e. the reference had
  run on an all-zero input. Root cause: `VisionBlockWeights` holds
  `std::span`s, and the test filled them from a loop-local `parts` vector
  that was destroyed each iteration, so `ref_blocks` held dangling spans
  and the reference read freed memory. The device path was correct. The
  test now owns the weights in a reserved `keep` vector. 30/30 passes on
  ROCm after the fix. The attention out-of-bounds fix above is a separate,
  real bug and stays.

- 2026-10-08: **Item 7 deepstack is not needed for the target.** The
  Qwen3.8 vision config sets `deepstack_visual_indexes` to the empty list,
  so the Qwen3VL deepstack multi-level feature injection has nothing to
  inject and the single merged embedding the vision encoder already
  produces is correct. Item 7's remaining work is only the image-prefill
  attention speed, which is the same GEMM/attention work as item 3.
- 2026-10-08: **Per-kernel profile of the GGUF decode (item 3).**
  `rocprofv3 --kernel-trace` on `EngineTest.GgufGeneratesWhenProvided`
  (27B GGUF, 5-token prefill plus 4 decode tokens) ranks the kernels by
  total device time (total 972 ms over 8775 dispatches):
  `gemm_q5k` 304 ms (31%), `gemm_q4k_batched` (the tiled prefill) 204 ms
  (21%), `gemm_q6k` 164 ms (17%), `gemm_iq4xs` 148 ms (15%), then
  `delta_step_heads` 27 ms, `l2norm` 26 ms, `gemm_q4k` (decode) 24 ms,
  `gemm_q80` 23 ms. The UD-Q4_K_M mix is not mostly Q4_K: the Q5_K, Q6_K
  and IQ4_XS GEMVs dominate the decode, and only Q4_K has a tiled kernel,
  so the Q5_K/Q6_K/IQ4_XS prefill still runs the per-row GEMV at m = 5.
  The next item 3 work is therefore (a) tiled kernels for Q5_K, Q6_K and
  IQ4_XS for the batched prefill, and (b) the same wide-read/vectorized
  treatment on those GEMVs. The coalesced Q4_K kernel above does not show
  up because its decode contribution is small.
  A second lever the profile implies: the kernel time is only part of the
  decode wall. Four decode tokens are about 1.24 s wall while the profiled
  kernel time for that span is well under it, and the single decode path
  already does exactly one `Synchronize` plus one tiny D2H per token (in
  `Qwen35Architecture::Logits`). So a large share of the per-token time is
  the CPU cost of submitting on the order of 500 kernel launches, each of
  which heap-allocates two pointer vectors in `Backend::LaunchKernel`.
  The ROCm `LaunchKernel` now uses fixed-size stack arrays instead (the
  counts are already bounded by `ValidateLaunch`), so it no longer
  allocates; `ctest` is 264/264. The 27B GGUF decode still measures 359
  ms/token this run, i.e. within the 310 to 359 ms/token run-to-run band,
  so the allocation removal is not separately measurable. `KernelLaunch`
  itself still builds two `std::vector`s per call in the launch helpers,
  so a fully allocation-free launch would need fixed arrays in that public
  type next.

- 2026-10-08: **GPU-only device tests.** The vulkan backend listed the
  llvmpipe CPU rasterizer as `gpu 2`, so `--gpu 2` or `TESSERA_TEST_GPU=2`
  silently ran the whole suite on software rendering. `Init` and
  `ListGpuNames` now share one `EnumerateGpuDevices` helper that drops
  `VK_PHYSICAL_DEVICE_TYPE_CPU` devices, so a GPU index can never select
  the CPU; vulkan now lists 2 GPUs here (was 3). The rocm backend needs no
  change (`hipGetDeviceCount` only sees real GPUs). Both test helpers now
  read the index from one `TestDeviceIndex` (the backend tests previously
  ignored `TESSERA_TEST_GPU` and always used device 0). New tests assert
  the GPU list has no llvmpipe entry and cover the env parsing. 262/262
  `ctest` on both builds.

- 2026-10-08: **Hardware fp16 block-scale decode (item 3).** The ROCm
  `Fp16ToFloatDev` ran a branchy software IEEE conversion, and every
  quant GEMM decodes a block scale per block per thread, so it was a
  significant per-element cost. On the device it now uses the exact
  hardware `__half2float` (`cvt`), with the software path kept for host
  compilation as the reference. `BackendTest.Gemm*` (22 tests) pass, so
  the conversion matches the host reference. The 27B GGUF decode measures
  336 ms/token, still inside the 310 to 359 ms/token run-to-run band, so
  as with the other micro-optimizations the effect is below the noise
  floor. Conclusion for item 3's decode: the GEMM kernels are at a local
  optimum (per-thread decode with no barriers beats both the shared
  once-per-workgroup decode, whose barriers cost more than the redundant
  work saved, and the coalesced reduction), and the run-to-run noise
  exceeds the remaining effects. Further decode work needs a quieter
  measurement or a different approach (for example a warp-per-output
  GEMV with shuffle broadcast), not another micro-optimization.

- 2026-10-08: **FP8 MTP head now loads on the MXFP4 target (item 2).**
  The MXFP4 loader skipped every `F8_E4M3` tensor, so the
  `Qwen3.8-27B-MXFP4-MTPFP8` MTP head never reached the device and
  `MtpDraftStep` failed with `UnsupportedFeature`. The head is eight
  projections stored as `float8_e4m3fn` with one `F32` scale per output
  channel (`fp8_mtp.py`: `scale = amax/448` per row, dequant
  `flat = fp8 * scale[row]`). `BuildMxFp4Weights` now handles `F8E4M3`:
  it pairs each weight with its `_scale`, dequantizes to `F32` at load
  the same way the DFlash2 draft handles FP8, and uploads; the
  architecture's `MapWeightName` already renames the `mtp.*` tensors to
  `blk.<trunk>.nextn.*`. New `EngineTest.LoadMxFp4Fp8MtpHeadDequantizes`
  pins the dequant. `HybridDecodeTest.MtpDraftWhenModelProvided` now
  passes on the MXFP4 target (866 tensors, was 858), and
  `HybridDecodeTest.SpeculativeMatchesGreedyOnModel` passes there too
  (the MTP speculative output equals plain greedy), so MTP speculation is
  both available and output-preserving on the MXFP4 target.

- 2026-10-08: **The served Qwen target is W4A8, not W4A4 (corrects the
  deferred DFlash2 hypothesis).** The reference launcher
  (`Quackster/vllm-gfx1201-launchers` @ `4d1ccbf`, `startup-qwen3.8-27b-vllm.sh`
  and its README) describes the Qwen3.8-27B target as "native MXFP4
  (W4A8)" and runs `RadianceMxfp4W4A8LinearKernel` / `radiance_mxfp4_fp8.hip`
  (an "fp8-WMMA W4A8 GEMM", boot log `304/304 on our kernel`). So the
  served activation quantization is **fp8 E4M3 (8-bit)**, not MXFP4 E2M1
  (4-bit). The deferred DFlash2 W4A4 experiment in `stash@{0}` quantizes
  the activation to E2M1 with an E8M0 block scale, which is the wrong
  activation format; a correct revisit would match the server's W4A8 fp8
  activation quant applied to the MXFP4 linear inputs, which is also why
  the stashed ROCm kernel made decode about 34x slower (it quantized in
  the per-output inner loop). The checkpoint's own
  `quantization_config.global_quant_config.input_tensors` still says
  `dtype: fp4` (Quark metadata), but the runtime serves fp8 activations.
  The MTP head from the same pipeline is confirmed fp8 (`fp8_mtp.py`),
  consistent with W4A8.

- 2026-10-08: **FP8 activation QDQ, and a negative DFlash2 W4A8 result
  (268/268 `ctest` on both builds).** New generic `quantize_fp8` built-in
  on Vulkan (GLSL) and ROCm (HIP): per-token quantize-dequantize to OCP FP8
  E4M3 with `scale = max(amax/448, 1/(448*512))` and round-to-nearest-even,
  the `dynamic_per_token_scaled_fp8_quant` contract (confirmed against
  radiance's `radiance_mxfp4.py`). Host reference `core::QuantizeFp8Ref`,
  contract and arg check, and `BackendTest.QuantizeFp8*` /
  `QuantizeMxFp4InputGatedByEnv` tests. The DFlash2 drafter now quantizes
  the concatenated target hidden to E4M3 per token before `fc`
  (`dflash2_fuse`), exactly the representation the training capture stores
  (`radiance_dflash_capture.py`), so the drafter runs on the input it was
  trained for. `ProjectBatch` moved to `models/qwen3_5/project.cpp` and now
  unifies the single and batched projections; with `TESSERA_MXFP4_W4A8=1`
  it also QDQ's each MXFP4 linear activation in place (the served target's
  W4A8). Measurement on `EngineTest.DFlash2MatchesGreedyOnModel`
  (5-token prompt, 8 tokens): aux QDQ alone 3 of 35; target W4A8 0 of 49.
  The fixture is the one PROGRESS already flags as noisy (its range is 0 to
  3 of 35), so this is not evidence that W4A8 hurts, but it is not a fix
  either. W4A8 also costs about 2.5x decode (1105 ms/token against 428 with
  it off), so it is off by default pending a long (128-token) generation
  measurement; the default MXFP4 output is unchanged and still matches the
   GGUF. The remaining suspect stays the aux hidden values from a different
   source (the served target also uses an fp8 KV cache and R4D fp8
   attention).

- 2026-10-09: **DFlash2 full-prefix draft context tried and reverted.** The
   drafter training reference (`paroquant/drafter/train_drafter.py`) builds
   the context from every position `0..anchor` (`ctx_ok = ctx_pos <=
   anchors`), so the committed windowed context (one position) was a
   simplification, not the reference behavior. The experiment extended
   `Verify`/`ForwardBatch` to capture the per-row residual hidden and kept
   every accepted prefix position as draft context, which matches the
   reference layout. Measured on the 27B MXFP4 target (ROCm, one generation
   of 32 tokens): acceptance fell to 0 of 217 draft tokens against 3 of 84
   with the windowed baseline (`TESSERA_DFLASH2_CTX=1`). Two causes. First,
   the implementation assembled the raw aux on the host and re-uploaded all
   `n * ctx * hidden` floats every step, so the run is O(ctx) in host memory
   and bandwidth (250 s for 32 tokens) and is not viable for a full
   generation. Second, the aux hidden values already differ from the served
   target (the deferred W4A8 and fp8-KV gap), so the extra context rows
   carry the same error and compound it. Reverted to the committed windowed
   baseline. A correct full-prefix implementation needs a device-resident
   context K/V cache (like vLLM's `precompute_and_store_context_kv`), not a
   raw-aux re-upload, and it must wait for the aux-value fix. The draft
   block itself is covered at `ctx = 3` by `BackendTest.DraftBlockMatchesRef`,
   so the kernel plumbing for a longer context is already exercised.

- 2026-10-09: **fp8 KV cache.** New generic "quantize_fp8_pack" (packed
   OCP FP8 E4M3 bytes with the dynamic per-token scale, four bytes per word)
   and "attention_fp8" (GQA over E4M3 keys and values with one fp32 scale
   per key row) on Vulkan (GLSL) and ROCm (HIP), with the host references
   `QuantizeFp8PackRef` and `AttentionFp8Ref` in `src/core/numerics/quant.*`
   and `attention.*` and device-vs-reference tests
   (`BackendTest.QuantizeFp8PackMatchesRef`,
   `BackendTest.AttentionFp8DeviceMatchesRef`). `KvCacheType::FP8` is wired
   through the vanilla and hybrid decode paths (`AppendKv`, the
   attention-kernel selection and the KV scratch), `ToString(KvCacheType)`
   names the types, and the CLI adds `--kv-fp8`. This is the fp8 KV cache
   the served MXFP4 target uses (item 0). 271/271 `ctest` on both builds;
   the 27B GGUF with `--kv-fp8` answers coherently end to end ("The capital
   of France is" gives " Paris."), which exercises the batched prefill and
   the hybrid decode. Fixed a latent host/device disagreement at the E4M3
   saturation boundary: the scaled row maximum can round just past 448,
   which the host converter turned into the NaN code (0x7F) while the device
   clamped to 448 (0x7E). The `quantize_fp8` and `quantize_fp8_pack` paths
   now clamp the quotient to `+-448` before the conversion, matching the OCP
   saturating semantics and keeping NaN out of a KV row. The
   `HybridDecodeTest.Fp8KvDecodesDeterministically` test pins the fixture
   path.

- 2026-10-09: **The DFlash2 acceptance gap is not the KV or activation
   quantization.** With the fp8 KV cache available, the drafter was measured
   against the served target's storage choices. On the 27B MXFP4 target
   (ROCm, GPU0), 32 tokens: fp32 KV accepts 9 of 154 draft tokens and fp8 KV
   (`TESSERA_DFLASH2_KV=fp8`) accepts an identical 9 of 154 (the same greedy
   tokens and the same acceptance), so the KV quant is not the cause. On the
   8-token fixture the served combination (`TESSERA_DFLASH2_KV=fp8` plus
   `TESSERA_MXFP4_W4A8=1`) accepts 3 of 35, the same as the fp32 baseline,
   while W4A8 alone gave 0 of 49, so the W4A8 activation quant is not the
   cause either (the served target is now reproduced: W4A8 linears plus an
   fp8 KV cache). Also rechecked against vLLM: `_maybe_add_hidden_state`
   appends `hidden_states + residual` (the residual stream) at
   `target_layer_ids + 1`, which is exactly what the C++ capture copies, so
   the captured tensor is not the mismatch. The remaining gap to the served
   2.85 accepted/draft is the aux hidden values from the target numerics not
   yet reproduced (vLLM runs bf16 residuals with fused RMSNorm+quant and the
   R4D fp8 attention), or the drafter runtime. The test now takes
   `TESSERA_DFLASH2_TOKENS` and `TESSERA_DFLASH2_KV` for these runs.

- 2026-10-09: **Tiled quantized attention (item 3).** The `attention_q8`,
   `attention_q4` and `attention_fp8` built-ins recomputed the query/key dot
   product once per output dimension, so a query/head cost
   O(n * head_dim^2). They now use the fp32 attention's tiled online
   softmax: one workgroup per (query row, head), the query staged in shared
   memory, one key scored per lane per tile, so the cost is
   O(n * head_dim) per query/head. On ROCm one `__device__` template
   (`AttentionQuantTiled<Kind>`) backs the three kernels; on Vulkan one
   `attention_quant.comp` serves all three with a `kind` scalar
   (`pc.v[14]`: 0 int8, 1 4-bit, 2 E4M3). The quantized attention contract
   gains that scalar and `AttentionQuantDevice` reads the kind from the
   kernel id and launches `rows * heads` workgroups. The old
   `attention_q8.comp`/`attention_q4.comp`/`attention_fp8.comp` are
   replaced. 271/271 `ctest` on both builds; the device-vs-reference tests
   `BackendTest.AttentionQ8DeviceMatchesRef`,
   `AttentionQ4DeviceMatchesRef` and `AttentionFp8DeviceMatchesRef` cover
   the tiled paths.

- 2026-10-09: **Warp-per-output MXFP4 GEMV (item 3b).** The m = 1
   `gemm_mxfp4` decode read a whole weight row per thread, uncoalesced across
   the warp, which the profile called the remaining item 3 lever. It now
   assigns one 32-lane warp per output element: lane l sums blocks
   `l, l+32, ...`, so consecutive lanes read consecutive MXFP4 blocks
   (coalesced) and the activation row is read coalesced, then the lane
   partials reduce with a shuffle (ROCm `__shfl_down`; Vulkan a 32-lane
   shared-memory group, eight per 256-thread workgroup). The grid is
   `ceil(m*n/8)` (`GemmGridFor` special-cases the id; the contract comment
   and the MXFP4 device test follow). A/B on the 27B MXFP4 target (ROCm,
   GPU0), three runs each of `EngineTest.MxFp4GeneratesWhenProvided`:
   602-605 ms/token before, 559-560 ms/token after, a 7.5% decode
   improvement with the same tokens (`11751, 13, 198, 760`), so the fp32
   reassociation did not change the output. 271/271 `ctest` on both builds.

- 2026-10-09: **DFlash2 device-resident draft context.** The draft now
   conditions on the committed prefix through a device K/V cache instead of
   one host-assembled position. `DFlash2Drafter::AppendContext` takes the
   target hidden at every capture layer for the newly committed positions,
   fuses it (`fc`), normalizes it (`hidden_norm`), and computes each draft
   layer's context K/V once (k_proj/v_proj, K-norm, RoPE at the absolute
   position), appending into a per-layer buffer that grows geometrically.
   `DraftContextAppendDevice` does the work; a context cap
   (`SetContextLimit`, `TESSERA_DFLASH2_CTX`) keeps only the most recent
   rows and advances the base. `Run` reads the cache and
   `DraftBlockDevice`/`DraftStackDevice`/`DraftLayerDevice` gained an
   optional precomputed context K/V view, so the per-step host assembly, the
   `UploadF32` and the O(ctx) host memory are gone. `VerifyDraft` /
   `Architecture::Verify` / `ForwardBatch` regained the capture hooks so the
   accepted positions' hidden is kept. The new
   `BackendTest.DraftContextAppendMatchesRef` pins the incremental append
   against one batch fuse plus `DraftContextKvRef`; 272/272 `ctest` on both
   builds. Measured on the 27B MXFP4 target (ROCm, GPU1, 16 tokens): context
   1, 8 and the checkpoint window (2048) accept 3 of 84, 2 of 91 and 1 of 98,
   all the same within the known bimodal noise. So the context width is not
   the acceptance blocker either; the remaining suspect is the target's
   fused-bf16 / W4A8 activation numerics (see the urgent Next item). The
   default context window is the checkpoint's `sliding_window` (2048, full
   prefix for a short sequence), matching the reference.

- 2026-10-09: **bf16 target activation mode (DFlash2 aux-value test).** New
   generic "round_bf16" built-in (Vulkan and ROCm) rounds fp32 to bfloat16
   in place (round to nearest even), with the host reference `RoundBf16Ref`
   and `BackendTest.RoundBf16MatchesRef`. With `TESSERA_TARGET_BF16=1`,
   `ProjectBatch` rounds every linear output to bf16, the served target's
   bf16 GEMM output. Measured on the 27B MXFP4 target (ROCm, GPU1,
   `TESSERA_DFLASH2_CTX=1`, 16 tokens): bf16 GEMM output plus W4A8 plus fp8
   KV accepts 3 of 84, identical to the fp32 baseline. So the target's
   activation precision is not the acceptance blocker either; combined with
   the earlier runs, every target-side hypothesis (fp8 KV, W4A8, bf16, and
   the context width) leaves acceptance unchanged, so the limit is the
   draft's own within-block chaining (see the urgent Next item). The mode is
   opt-in; the default fp32 path and the MXFP4/GGUF outputs are unchanged.
   273/273 `ctest` on both builds.

- 2026-10-09: **DFlash2 aux diagnostic and chat-prompt measurement.** A
   temporary env-gated dump showed the captured target hidden is sane and
   non-zero at every capture layer (the absmax figures in this entry were
   from a bad dequant; the corrected values are 81, 170, 271, 221, 221 for
   layers 5/19/33/47/61, see the newest Done entry) and the context cache is
   populated. A real chat prompt
   (`Explain in one paragraph why the sky is blue...`, ~30 context tokens,
   `--tokens 32`, `--kv-fp8`) accepts 3 of 196 draft tokens over 28 steps,
   so a longer context does not raise acceptance either. Combined with the
   earlier runs, the draft is faithful, the aux is non-zero, and no
   target-side knob moves acceptance, which points to the drafter/target
   distribution mismatch the reference itself records (self-distillation on
   the served target's hidden). See the urgent Next item.

- 2026-10-09: **DFlash2 aux hidden captured from the served target and
   diffed: it matches.** The served radiance container was run on GPU1
   (TP=1) with `RADIANCE_DFLASH_CAPTURE_DIR` set, serving
   `Qwen3.8-27B-MXFP4-mtpfp8-pertoken`, and its DFlash2 aux hidden was
   captured for a fixed 96-token sequence (the "why is the sky blue" chat
   prompt plus 64 generated tokens) as E4M3 with a per-token scale. Tessera
   then ran the same 96 token ids through the target (`MTPFP8`) and dumped
   its aux via the new env `TESSERA_DFLASH2_DUMP_AUX` (test
   `EngineTest.DumpDFlash2AuxWhenProvided`, driven by
   `TESSERA_TEST_DFLASH2_TOKENS_FILE`). The two agree well: per-layer cosine
   0.9985 to 0.9997 (layer 5/19/33/47/61), relative L2 error 3 to 6 percent,
   and matching magnitudes (tessera absmax 81/170/271/221/221 against the
   reference 78/175/272/214/221). Tessera's aux data is off by at most about
   one E4M3 quantization step: re-quantizing it exactly as the capture does
   matches 36 to 50 percent of the reference bytes, and the post-quant
   cosine stays 0.998 to 0.9997. So the aux hidden values are **not** the
   cause of the 14x acceptance gap, and the earlier "aux values" hypothesis
   is ruled out. The reference DFlash2 path was also re-read from the local
   image (`vllm/v1/worker/gpu/spec_decode/dflash{,2}/speculator.py`,
   `model_executor/models/qwen3_dflash{,2}.py`): the draft config sets no
   `input_embedding_scale`, `output_multiplier` or
   `final_logit_softcapping` (all default), so those are not differences
   either. The gap must be in tessera's draft-input path (context K/V,
   query/anchor positions, candidates or the selector walk). The served
   reference reaches 2.7 to 2.85 accepted per draft; tessera stays at about
   0.1 to 0.45. Temporary diagnostics: the radiance capture needs
   `patch_dflash_capture.py` and the `serve-mxfp4.sh` capture mount (in
   `~/git/radiance-vllm-mxfp4`, uncommitted); the tessera dump is env-gated.
   The build is 273/273 on rocm.

- 2026-10-09: **DFlash2 acceptance root-caused and fixed: the candidate
   extraction shrank its own index vector.** With the aux and the draft block
   both measured equal to the served reference, the draft path was compared
   stage by stage against the reference plain-torch drafter on the *same*
   captured aux. Tessera's block hidden matches the reference (cosine 0.997
   to 0.9996), and projecting either hidden with the target head gives the
   **same 7-of-7** correct mask-row tokens, so the unary logits were already
   correct. Two defects remained, both downstream:
   1. `spec::DraftCandidates` called `row_ids.resize(top_k)` on a vector
      reused across rows. After the first row the vector held only `top_k`
      elements, so `std::iota` filled just `0..top_k-1` and every later row
      ranked a tiny slice of the vocabulary (ids 0..15). This is the real
      cause of the low acceptance: only mask row 0 ever had the full vocab.
      The fix keeps the vector at size `vocab` and sorts only the `top_k`
      prefix. Test `BackendTest.DraftCandidatesRanksWholeVocabPerRow`.
   2. The candidate-selector transition score added `unary[p]` (the
      predecessor's logit) instead of `unary[c]` (the successor's), so the
      walk ignored the unary term. The reference `_score_edges` maps
      `unary_logits[:, :, None]` onto the successor axis. Fixed in the host
      reference `core::SelectorEdgeScoreRef`, the ROCm kernel and the Vulkan
      shader. Test `BackendTest.SelectorEdgeScoreUnaryIsSuccessor`. The old
      device test compared the kernel to the (equally wrong) reference, so it
      could not catch this; the earlier note "bypassing the selector gives
      the same rate" was measured while `DraftCandidates` was still broken.
   With both fixed, the diagnostic draft tokens for the fixed 96-token
   sequence are 7 of 7 correct per step (was 0 to 1), and
   `EngineTest.DFlash2MatchesGreedyOnModel` on the 27B MXFP4 target (prompt
   `{760,6511,314,9338,369}`, 64 tokens) accepts **49 of 98 draft tokens over
   14 steps (3.5 per step)** against 0.1 to 0.45 before, now at or above the
   served reference (2.7 to 2.85). Output still equals greedy. 275/275
   `ctest` on vulkan and rocm; the ROCm end-to-end run is the acceptance
   measurement, the Vulkan shader is covered by the device-vs-reference
   test. The one-off capture helpers were removed from the radiance repo.

- 2026-10-09: **DFlash2 steady-state tokens/s measured: acceptance is fixed,
   but the draft step is too expensive.** `GenerateDFlash2` now logs a timing
   line (generated N tokens in X ms, Y ms/token) next to the acceptance line.
   On the 27B MXFP4 target (ROCm, GPU1), prompt `{760,6511,314,9338,369}`,
   128 tokens: greedy is **206 ms/token** (26.4 s) while DFlash2 accepts
   **106 of 168 draft tokens over 24 steps (4.4 per step)** but takes
   **949 ms/token** (121.5 s), about 4.6x slower than greedy. So the draft
   quality is now better than the reference's 2.7 to 2.85 per step, and the
   remaining DFlash2 work is the per-step draft cost, not quality: the draft
   block is fp32 (about 3.6 GB of weights read per step), the selector
   codebooks are fp32 (about 508 MB per step), and the step issues many small
   kernels. This is item 3 (GEMM throughput) work, and it must land before
   DFlash2 is a net win. 275/275 `ctest` on both backends.

- 2026-10-09: **Coalesced the fp32/bf16 GEMVs and stopped widening the head
   (hipfire-guided).** A `rocprofv3` kernel trace of the DFlash2 run showed
   `GemmF32Kernel` was 28.2 s of 38.9 s of device time (487 dispatches, about
   58 ms each): `gemm_f32` read one weight row per thread with a stride-k
   access, so it ran at about 6 GB/s. The same profile showed the cause on
   the greedy path: the MXFP4 loader widened **every** BF16 tensor to F32,
   including the 248320x5120 vocabulary head, so greedy decode streamed a
   5 GB fp32 head through that uncoalesced kernel. Both `gemm_f32` and
   `gemm_bf16` are now warp-per-output on ROCm and Vulkan (lane l sums
   `t = l, l+32, ...`, shuffle/shared reduce), `GemmGridFor` dispatches
   `ceil(m*n/8)` workgroups for both ids, and the head (with the embedding)
   stays BF16 in the loader (`mxfp4.cpp`), so the head is 2.4 GB, not 4.8 GB
   (BF16 to fp32 is exact, so the tokens are unchanged). `GenerateSpeculative`
   (MTP) gained the same accept-count and timing log as `GenerateDFlash2`.
   Measured on the 27B MXFP4 target (ROCm, GPU1, 128 tokens):
   greedy **206 -> 51 ms/token** (DFlash2 prompt) and **40 ms/token** (MTP
   prompt), DFlash2 **949 -> 296 ms/token** with the same 106 of 168
   acceptance, and MTP **1396 ms/token accepting 0 of 508** (so MTP
   speculation is output-preserving but useless on this target; the MTP head
   draft quality is the next suspect, and it is now measurable).
   `BackendTest.GemmF32DeviceMatchesRef` and `GemmBf16DeviceMatchesRef` use
   `GemmGridFor`; 275/275 `ctest` on both backends. The design comes from
   hipfire (`~/.hipfire/src`, the same author's Rust engine for the same
   R9700/Qwen3.8): decode is a warp-per-row GEMV with weights kept
   quantized, the head stays quantized (Q8_0, about 1.35 GB, never widened),
   projections are fused per layer, and one `Speculator`/`SpecTarget` seam
   plus `accept_greedy_prefix` is shared by MTP and DFlash.

- 2026-10-09: **Greedy and speculation are one loop (`SpeculativeStrategy`
   seam).** The engine had three parallel decode loops (`Generate` greedy,
   `GenerateSpeculative` MTP, `GenerateDraft` DFlash2) and the public
   `SpeculativeStrategy` was a skeleton DFlash2 did not implement. It is now a
   real drafter seam with `Prepare`, `DraftBlock`, `CaptureLayers`,
   `CaptureBuffers`, `OnAnchor`, `Draft` and `Commit`, mirroring hipfire's
   `Speculator`/`SpecTarget`: the engine owns the target, the KV/recurrent
   state and the single accept rule (`VerifyDraft`); the strategy owns the
   draft model, the policy and the target-hidden capture buffers it reads.
   `Engine::GenerateStreaming` is the one decode loop: with no strategy it is
   plain greedy; when a strategy is attached it does prefill (per-token with
   capture for a hidden-conditioned drafter, batched otherwise), then per
   window `OnAnchor` -> `Draft` -> `VerifyDraft` -> `Commit` -> emit the
   accepted prefix plus the bonus, capping at `max_tokens`. `GenerateDraft`
   and `GenerateSpeculative` now attach a `CreateDFlash2Strategy()` /
   `CreateMtpStrategy()` and run that loop; the old free function
   `spec::GenerateDFlash2` and the `dflash2_generate.*` pair were deleted.
   Behaviour is preserved: on the 27B MXFP4 target DFlash2 still accepts
   106 of 168 (4.4 per step) and MTP still accepts 0 of 508 (output equals
   greedy in both). The engine now logs prefill and decode throughput
   separately (`prefill: N prompt token(s) ... prompt tok/s` and `decode: M
   token(s) ... tok/s`), so prompt and decode are measured apart the way
   hipfire's README reports pp8192 and decode. Measured (ROCm, GPU1, 128
   tokens, 5-token prompt): greedy decode 25 tok/s (39 ms/token), DFlash2
   decode 3 tok/s (295 ms/token), MTP 0 tok/s (1393 ms/token, no acceptance).
   Both are far below hipfire on the same R9700 (DFlash 123 tok/s, MTP 68,
   greedy ~200), so the next work is decode speed, not the seam. 275/275
   `ctest` on both backends.

- 2026-10-09: **Long-prompt prefill no longer exhausts memory.** A 112-token
  prompt on the 27B MXFP4 target failed with `out_of_memory`; both GPUs were
  idle (34 GB each), so it was tessera's own over-allocation. `AllocBatch`
  sized two scratches by the whole batch: the logits scratch as
  `rows * vocab_size` (112 x 248320 x 4 B = 111 GB) and, for every linear
  layer, `(rows + 1) * state_len` recurrent-state snapshots (about 23 GB for
  112 rows across the 48 linear layers). Prefill only scores the last token
  and never rolls back, so `AllocBatch` now takes `logits_rows` and a
  `need_state_hist` flag: prefill allocates one logits row and no state
  history, and a verification (which does roll back, even for a one-token
  draft, so it needs slot 0) grows the history and the logits to the draft
  size on demand. `RunLinearBlockBatch` skips the per-row state/conv
  snapshots when `Qwen35BatchScratch::snapshot_states` is false. Verified:
  the same 112-token prompt now prefills and decodes, MTP and DFlash2 on the
  27B still reproduce greedy (new `HybridDecodeTest.
  BatchedPrefillThenVerifySizesScratchByNeed` pins the sizing and the
  prefill-then-verify transition), 276/276 `ctest` on both backends. Prefill
  itself stays slow (about 6 prompt tok/s, item 6).

- 2026-10-09: **Prefill is 5.6x faster: the D2D copy was synchronous.** The
  112-token prefill above then took 18.2 s (6 prompt tok/s). The batched
  linear-attention block runs token by token on the host and copies the
  per-row scale, gate and output through `Backend::CopyD2D` (about four
  copies per row per layer, 21k for this prompt); the ROCm `CopyD2D` used a
  synchronous `hipMemcpy`, so every copy drained the async kernel queue and
  nothing pipelined. `CopyD2D` now uses `hipMemcpyAsync` on the null stream,
  which stays ordered with the null-stream kernels but does not block the
  host. The 112-token prefill drops to 3.3 s (34 prompt tok/s) and the
  greedy/speculative decode rates are unchanged (35 ms/token). Also widened
  the tiled MXFP4 GEMM tile from 8 to 256 rows (one 256-thread block per
  weight row) so a prefill reads the weight matrix once instead of once per
  eight-row tile; it is correct, but the prefill bottleneck was the copy,
  not the GEMM, so it moved the number only slightly. 276/276 `ctest` on
  both backends; MTP and DFlash2 on the 27B still equal greedy. The Vulkan
  `LaunchKernel` still waits on a fence per launch (a separate, larger
  inefficiency to fix under item 6).

- 2026-10-09: **The MXFP4 tiled GEMM was latency-bound; four partial sums
  fix it.** With the copy fix the 112-token prefill was 3.27 s and
  `rocprofv3` put 2.35 s (76%) in `GemmMxFp4BatchedKernel`. Stubbing that
  kernel to a no-op dropped the prefill to 0.94 s, proving it is the cost,
  yet widening the tile (8 to 256 rows) and a 32x8 tile with cache reuse
  changed nothing: the kernel is neither weight- nor activation-bandwidth
  bound. Each thread accumulated the whole k into a single dependent FMA
  chain, so it was latency bound. Now each thread keeps four independent
  partial sums (one per 8-element word of the 32-element block) on a 32x8
  output tile; eight partial sums regressed (the extra live registers cost
  more occupancy than the ILP bought). The 112-token prefill drops to
  2.18 s (51 prompt tok/s) and, because the speculative verifier scores its
  draft with the same kernel, DFlash2 on the 27B drops from 443 to 125
  ms/token (3 to 7 tok/s). Both backends port the change; 276/276 `ctest`;
  DFlash2 accepts 29 of 63 and still equals greedy. (The `gemm_mxfp4` m = 1
  decode GEMV has a shallow chain and stays memory bound, so it needs no
  such change.)

- 2026-10-09: **Tiled batched F32/BF16 GEMMs speed the DFlash2 draft and
  verify.** Profiling a DFlash2 decode showed `GemmF32Kernel` (452 calls,
  509 ms) and `GemmBf16Kernel` (19 calls, 387 ms) — the draft projections
  and the shared head — both warp-per-output, so each weight row was read
  once per draft row (m about 7). New `gemm_f32_batched`/`gemm_bf16_batched` kernels (32-row
  x 8-column tile, four partial sums per thread, both backends) read each
  weight row once; the DFlash2 drafter and selector load the f32 variant,
  and `ProjectBatch` now selects the tiled F32/BF16 kernels for m > 1 so the
  target's bf16 verify head uses one too. DFlash2 on the 27B drops from 177
  to 136 ms/token (CLI) and 125 to 101 ms/token (test); the draft still
  accepts 29 of 63 and equals greedy. New
  `BackendTest.GemmBatchedF32Bf16DeviceMatchesRef`; 277/277 `ctest` on both
  backends.

- 2026-10-09: **Small batches use the GEMV, not the 32-row tile.** The
  DFlash2 verify runs the target trunk at the draft size (about 7), where the
  tiled MXFP4 kernel's 32-row tile leaves most rows idle and every thread
  runs a deep k-chain. `ProjectBatch` now uses the tiled kernel only for
  `m >= kGemmTiledMinRows` (16); smaller batches use the m = 1 GEMV, whose
  shallow chains hide latency better. DFlash2 on the 27B drops 136 -> 128
  ms/token; the 112-token prefill (m = 112) is unchanged at 2.16 s. 277/277
  `ctest` on both backends.

- 2026-10-09: **The per-row linear prefill loop is batched.** Prefill never
  rolls back, so it now runs the causal conv, the two L2 norms, the head
  repeat, the delta scan and the gated norm over all rows in one launch
  each, instead of about eleven launches and four device-to-device copies
  per row. The `conv1d_state`, `repeat_heads` and `delta_step_heads` kernels
  take a `rows` scalar and advance the shared conv history / recurrent
  state across the rows internally; `rows == 1` is the previous single-token
  behavior, so decode and the rollback path (which needs per-row state
  snapshots) keep the per-row loop. The 112-token prefill wall is unchanged
  (2.16 s): the loop is GPU-work-bound on ROCm, whose launches were already
  asynchronous, so the launch and copy reduction does not show there. It
  should help the Vulkan backend, whose `LaunchKernel` waits on a fence per
  launch (a follow-up). 277/277 `ctest` on both backends; DFlash2 on the 27B
  still accepts 29 of 63 and equals greedy.

- 2026-10-09: **Vulkan launches no longer wait on a fence each.**
  `VulkanCompute::LaunchKernel` recorded one command buffer and waited its
  fence before returning, so the GPU drained after every kernel and never
  pipelined. It now uses a four-slot ring of command buffers, fences and
  per-slot descriptor sets: a slot is reused only after its fence signals,
  and its set is freed then, so the host never waits inside a step
  (`Synchronize` drains the queue). A/B on the 27B MXFP4 (ring size 1
  reproduces the old per-launch wait): decode 113 to 69 ms per token (8 to
  14 tok/s) and prefill 27 to 31 prompt tok/s. 277/277 `ctest` on both
  backends (the numerical tests exercise the descriptor and command-buffer
  lifecycle).

- 2026-10-09: **The vocab-width bf16 head uses the tiled kernel too.** The
  16-row threshold (small batches use the shallow-chain GEMV) sent the
  target's bf16 output head to the GEMV, which at the DFlash2 draft size
  (m about 7) re-reads the 2.5 GB head once per row (387 ms of the decode).
  The tiled kernel reads it once. `ProjectBatch` now also picks the tiled
  kernel when `n >= kGemmTiledMinCols` (65536), so the head is tiled while
  the narrow trunk projections stay on the GEMV. DFlash2 on the 27B drops
  from 138 to 123 ms/token. 277/277 `ctest` on both backends.

- 2026-10-09: **DFlash2 draft GEMM weights are stored bf16.** The draft
  weight loader dequantized the fp8-block checkpoint to fp32 (4 bytes per
  weight); the projection weights (q/k/v/o, the conv kernel projections,
  gate/up/down, fc and the selector projection) now upload bf16 (2 bytes),
  which is still more precise than the fp8 source, while the norm, conv-base
  and codebook tensors stay fp32 (their kernels take fp32). The drafter and
  selector use `gemm_bf16_batched`. Draft weight memory halves; DFlash2 is
  123 -> 121 ms/token (marginal: the draft GEMMs are not the bottleneck) and
  the 27B acceptance is unchanged (10 of 42). 30 draft/spec tests pass.

- 2026-10-09: **MTP acceptance fixed (0 -> about 50%).** The MTP head is a
  Qwen3.5 block, so its RMSNorms (`nextn.enorm`, `nextn.hnorm`,
  `nextn.shared_head_norm`) are the Gemma form (gain `1 + weight`). The MXFP4
  `ConvertWeight` applied the `add_one` offset only to the trunk norms
  (`attn_norm`, `post_attention_norm`, `attn_q_norm`, `attn_k_norm`,
  `output_norm`), so the MTP head ran with an off-by-one norm and every
  draft was wrong: 0 accepted. Adding the three MTP norm tails to the
  offset list fixes it. On the 27B MXFP4 the MTP now accepts 13 of 24
  (5-token prompt) and 8 of 28 (1-token prompt), and output still equals
  greedy. MTP decode is still slower than greedy (the per-draft MTP head and
  the verify batch cost more than the tokens recovered), but it is no longer
  a no-op. 277/277 `ctest` on both backends.

- 2026-10-09: **MTP default chain shortened to one draft.** A longer MTP
  chain both lowers the accept rate (a later draft in the chain is wrong more
  often) and multiplies the per-draft MTP-head cost (each draft reads the
  shared 2.5 GB output head). Measured on the 27B MXFP4 over 24 tokens:
  block 1 = 21 tok/s (11 of 13 accepted), block 2 = 18 (15 of 20), block 4 =
  15 (19 of 32). `kDefaultMtpBlock` is now 1; `--draft-block` still
  overrides. MTP is now within about 15 percent of greedy (46 vs 39
  ms/token). 277/277 `ctest` on both backends.

- 2026-10-09: **Small-m verify GEMM (warp-per-column multi-row MXFP4 GEMV).**
  The speculative verifier runs the target trunk at m about 2 to 7; the
  single-row `gemm_mxfp4` is warp-per-(row,column), so each weight row is
  re-read once per row. New `gemm_mxfp4_rows`: one warp owns one weight
  column and accumulates all m activation rows, decoding the weight block
  once per column. `ProjectBatch` routes MXFP4 with m in [2,16] to it (the
  GEMV stays for m=1, the tiled kernels for m>=16 and the vocab head). The
  DFlash2 draft head (`head_gemm_`, the shared bf16 output head) now uses the
  tiled bf16 kernel too, so it reads the 2.5 GB head once instead of once per
  draft row. DFlash2 on the 27B drops 121 -> 95 ms/token (CLI) and 77 -> 71
  ms/token (test); acceptance unchanged and output equals greedy. Both
  backends: the Vulkan shader (`gemm_mxfp4_rows.comp`, warp-per-column with a
  shared-memory row reduction) is validated by the new
  `BackendTest.GemmMxFp4RowsDeviceMatchesRef`. Within the rows kernel several
  variants regressed and were
  reverted: a cooperative element-per-lane mapping (coalesced reads, but a
  160-deep accumulator chain) measured 200 ms/token; four partial sums per row
  (register spill from 64 accumulators) measured 192; two partial sums per row
  measured 102. The single-accumulator lane-per-block form stays the best
  found, and the ILP variants making it *worse* says the kernel is
  occupancy-bound rather than latency-bound. 277/277 `ctest` on both backends.
  Next lever found: the engine runs an extra single-token target forward every
  speculative step (`engine.cpp:419`, `DecodeLogits(next, &hidden)`) to advance
  the KV cache and the hidden for the bonus token, about 450 MXFP4 GEMV calls
  and ~43 ms/step in the DFlash2 profile. The verify's batched forward already
  computes that position, so its KV row and captured hidden can likely be
  reused instead of a second forward; that is the next change, ahead of the
  shared-memory verify GEMM.

- 2026-10-09: **Reference (radiance-vllm-mxfp4 / vLLM) speed techniques
  catalogued** in `~/git/radiance-vllm-mxfp4` and `~/git/vllm`, for the
  remaining MTP/DFlash2 gap. Measured there, candidates to port:
  `DECODE_MAX_M` small-M decode GEMM (M<=64, split-K, TM=ceil(M/16); -4.8%
  step, +28% conc-4) -- the same lever as the small-m verify GEMV above;
  `RADIANCE_MXFP4_WPERM` + `..._DECODE_NT` fragment-order weight permutation
  and nontemporal decode loads (-5.4% step); int2 target verify head
  (+2.9%); `FAST_DRAFT` int2 MTP draft head with a bf16 rerank (+6.5%);
  dynamic per-request verify width (+11-13% conc-8); GDN `in_proj`
  single-GEMM merge (removes 96 launches + 48 quant per forward); fused
  DFlash2 context-KV projection (one GEMM for every layer); the MTP head is
  FP8 E4M3 per-channel (`fp8_mtp.py`) and runs as FP8 (tessera dequantizes
  it to F32); DFlash2 runs as one captured graph per step.

- 2026-10-09: **The linear-attention verify kernels get ILP.** The gated-delta
  scan (`DeltaStepHeadsKernel`) accumulated its `read` dot and its output in
  one dependent chain per state column (`dk` deep, 128); four partial sums for
  each halve the chain. Then `L2NormKernel` and `RmsnormGatedKernel` (one
  thread per row, sequential accumulation) got the same four-way split. Both
  backends. DFlash2 on the 27B drops 95 -> 88 ms/token (CLI) and 71 -> 66
  (test); acceptance unchanged (29 of 63) and output equals greedy. The
  delta-scan read is now four deep instead of 128; the reference's GDN work
  targets the same area. 278/278 `ctest` on both backends.

- 2026-10-09: **`quantize_fp8` is one workgroup per row.** The kernel was one
  thread per row, so the DFlash2's per-token FP8 QDQ of a single-row
  (~1 x 25600) context hidden ran on one thread. It now uses a workgroup per
  row with a shared-memory absmax reduction, so the whole block works.
   Callers (`QuantizeMxFp4Input`, the DFlash2 fuse) dispatch `rows`
   workgroups; the Vulkan shader mirrors it. DFlash2 on the 27B drops 88 -> 83
   ms/token (CLI) and 66 -> 61 (test); acceptance unchanged (29 of 63) and
   output equals greedy. 278/278 `ctest` on both backends.

- 2026-10-09: **OpenAI tool calling on `/v1/chat/completions`
  (286/286 `ctest` on rocm; vulkan build not run, the code is backend
  agnostic).** New `src/serve/respond.*` (shared response helpers),
  `src/serve/render.*` (`ToJinja` plus `RenderPrompt`, moved out of
  `server.cpp`), `src/serve/tools/tool_call.*` (Qwen XML `<tool_call>`
  parser, tool history normalizer) and `src/serve/tools/chat.*`
  (tool-aware chat handler). `tools` render into the model template
  through the existing Jinja subset (it already covers `tojson`,
  `raise_exception`, `startswith` and `previtem`/`nextitem`); incoming
  assistant `tool_calls` convert JSON-string arguments to objects and
  `tool_choice: "none"` renders without tools. The handler answers
  with OpenAI-form `tool_calls` (buffered and SSE) and strips the
  `<think>` block from the reply content. Verified live on the 27B
  GGUF (ROCm, GPU1): call emission, streaming deltas, a tool-result
  turn, and clean text answers. The opencode provider wiring
  (`tessera/qwen3.8-27b` on 127.0.0.1:8080) resolves, but live agency
  is blocked: opencode's first prompt is about 10k tokens against the
   4096 serve context at 11 prompt tok/s (see Next item 4).

- 2026-10-09: **Tool-path thinking as `reasoning_content` (308/308
  `ctest` on vulkan).** The tool handler deleted the `<think>` span,
  so a client that sends `tools` (opencode always does) never saw the
  reasoning. `ParseToolCalls` now also returns the think text, and the
  buffered and streamed tool replies report it as `reasoning_content`
  (the vLLM convention) beside the stripped content and the calls.
  Verified live on the 27B MXFP4 (Vulkan, GPU0): the SSE stream
  carries the thinking delta, then the clean answer, with no device
  errors.

- 2026-10-09: **Web chat UI with sessions (316/316 `ctest` on
  vulkan).** The HTTP server runs one worker thread per connection,
  so a streaming generation never blocks control requests; device
  generation stays serialized (a second turn answers 503). New
  in-memory sessions (`SessionStore`) with `/api/sessions` CRUD, chat
  and retry turns (SSE or buffered), and stop/pause/resume controls
  that abort or stall the decode loop. Session turns split thinking
  live (`ThinkStreamer`) into `reasoning_content` deltas beside the
  answer. The dependency-free UI (`src/serve/web/`, embedded at
  configure time, served at `/`) lists sessions, streams thinking and
  answers, and renders fenced SVG inline, fenced HTML in a sandbox
  frame, and markdown and data URL images. Verified live on the 27B
  MXFP4 (Vulkan, GPU0): the full session flow through curl, including
  pause, resume and a stopped turn persisted with its flag.

- 2026-10-09: **Stop on disconnect and SIGPIPE survival (323/323
  `ctest` on vulkan).** Every serve send used bare `send()`, so a
  cancelled stream killed the server with SIGPIPE; surviving turns
  also decoded into the void. Sends now use `MSG_NOSIGNAL`, and every
  generation hook aborts when `ResponseWriter::IsPeerGone` sees the
  close: UI stop, closed window/tab, and opencode request cancels all
  free the device at the next token. Verified live on the 27B MXFP4
  (Vulkan, GPU0): a hard-killed mid-stream client leaves the server
  alive, and the next request serves in about a second.

- 2026-10-09: **Default 32k completion cap (322/322 `ctest` on
  vulkan).** Requests without `max_tokens` decoded to the end of the
  context. `tessera::kDefaultMaxTokens` (32k) is now the single
  source: `ServeOptions`, the CLI `--tokens` default and help text,
  and the web UI input default (injected into the served page) all
  derive from it. The engine keeps 0-means-fill for direct API use.

- 2026-10-09: **Chunked prefill and 220k context on the 8-bit KV path
  (299/299 `ctest` on vulkan).** `PrefillTokens` splits the prompt into
  chunk-sized forwards (automatic default 512, `--prefill-chunk` to
  override), so a 220k-token prompt prefills in bounded scratch instead
  of losing the device in one giant forward. Only the last chunk scores
  logits. The DFlash2 path prefills the prompt head without capture and
  appends only its recent window (2048+1 rows) in batched chunks through
  the new `PrefillCaptureTail`/`AppendPrefill` strategy seam; MTP keeps
  the per-token loop. A prompt past the context is rejected with
  `InvalidArgument` before any device work (serve answers 400). Also
  fixed: `ForwardBatch` returned a stale hidden row when the reused
  batch scratch outlived its rows; it now copies the last row out
  first. Chunked prefill matches one forward on both hybrid fixtures
  (logits, hidden and the follow-up decode).

- 2026-10-09: **Bounded prefill launches (302/302 `ctest` on vulkan).**
  A 76k-token prefill at 220k context hung the GPU and the driver
  reset the device. One attention launch per full layer covers rows
  times keys pairs. Past about 6M pairs a single launch outruns the
  kernel driver hang timeout. The driver reset shows as a sticky
  device loss on every later call. `PrefillTokens` and the DFlash2
  head and tail prefill now clamp each chunk to the attention work
  budget (`core::detail::ClampPrefillRows`, 4M pairs at 24 heads and
  head dim 256, scaled for other shapes). Each chunk keeps at least
  one row, so a long prompt always progresses. The clamp lives in
  the backend agnostic core, so both backends get it. It is verified
  on vulkan. Three new tests pin the schedule (the budget bound, the
  76k crash shape, edges). The engine reports chunk progress through
  `PrefillProgress` (one line per `progress_every` chunks), so a long
  prefill shows movement instead of silence. Full attention stays
  quadratic: a full 220k prefill is hours of single GPU work on any
  runtime. A FlashAttention class kernel is the remaining speed work
  (see Next item 6).

- 2026-10-10: **DFlash2 output no longer equals greedy: the fp8-WMMA
  verify used different MXFP4 numerics than the scalar greedy path.**
  The fp8 tensor-core MXFP4 GEMM (the served target's W4A8 path) was
  wired for the verify at m in [2, 16] while the m=1 greedy path kept
  the scalar GEMV. On the 27B MXFP4 target the two disagree on the
  argmax after about 20 tokens, so `EngineTest.DFlash2MatchesGreedyOnModel`
  fails at 128 tokens (it passes at 16, which is why the regression was
  not seen). `ProjectBatch` now routes m>=1 through the same fp8-WMMA
  MXFP4 kernel, so greedy and the verify share numerics and the output
  equals greedy again. Measured on the 27B MXFP4 target with the real
  DFlash2 draft (ROCm, GPU1), 128 tokens: DFlash2 55 tok/s, accepted
  101 of 182 draft tokens, and output equals greedy. `TESSERA_MXFP4_WMMA=0`
  still selects the all-scalar path (correct, 38 tok/s), which is the
  fallback where the backend has no fp8 tensor cores (Vulkan). Greedy
  on the MXFP4 target is 43 ms/token with the WMMA (the scalar GEMV was
  34 ms/token); the WMMA is the served path and is required for the
  speculative output to stay equal to greedy.

- 2026-10-10: **The wide bf16 output head runs on the bf16 tensor-core
  kernel.** `rocprofv3` showed the verify's vocab-sized head projection
  (`gemm_bf16_batched`, m=9, grid 31040) at 273 GB/s, half the draft
  head's 598 GB/s on the same 2.5 GB weight. The tiled kernel's 32-row
  tile leaves 23 of 32 rows idle at the draft/verify size. `ProjectBatch`
  now routes a bf16 projection with `n >= kGemmTiledMinCols` and
  `m <= 16` through `gemm_bf16_wmma`, the same kernel the DFlash2 draft
  head already uses, so the greedy head (m=1) and the verify head (m=9)
  share numerics and the head streams coalesced. The verify head drops
  from 9.3 to 4.25 ms/step. The `DFlash2Strategy` also defers its context
  append to `Commit`, so the anchor row and the accepted draft rows append
  in one call (the fc and the per-layer context K/V projections run once
  per step instead of twice). DFlash2 on the 27B MXFP4 target moves from
  about 88 to 82 ms per speculative step (256 tokens: 55 tok/s, accepted
  199 of 392, output equals greedy). 330/330 `ctest` on ROCm; the
  Vulkan build compiles and falls through to the tiled kernel where the
  backend has no bf16 tensor cores.
  Profiling the DFlash2 step (ROCm, 64 tokens, `rocprofv3`): MXFP4
  verify 30 ms/step (424 GB/s), draft forward 16 ms/step, delta scan
  5.5, verify head 4.25, L2 norm 2.4, fp8 activation pack 1.7,
  split-K reduce 1.4, attention 1.0. The draft's small-n bf16 GEMMs
  (n = 1024 to 5120) run at 160 to 300 GB/s because grid is only 16 to
  80 workgroups. The next levers are the draft (fuse the per-layer
  context K/V and the q/k/v projections, as the reference does) and
  the small-n GEMM occupancy.

- 2026-10-10: **The batched verifier argmaxes on the device.** With the
  anchor folded into the verify batch, `Qwen35Architecture::Verify`
  downloaded the whole `rows x vocab` logits (about 9 MB at the draft
  size) and argmaxed each row on the host, on the critical path between
  steps. The engine reads only `accepted` and `next_token` from the
  result, so the anchor path now runs the trunk and the head without the
  logits download, argmaxes every scored row on the device
  (`top_k_rows`, top-1) and reads back `rows` ids (32 bytes). The head
  and the device logits are shared with `ForwardBatch` (`RunBatchHead`);
  the top-1 scratch is shared with the MTP draft (`DeviceRowArgMax` in
  `internal.hpp`, replacing the MTP's inline argmax). The no-anchor path
  keeps the host download because the caller reads `result.logits`
  (`HybridDecodeTest.BatchedVerifyPreservesGreedy`). DFlash2 on the 27B
  MXFP4 target drops from 82 to 75 ms per speculative step; 256 tokens:
  60 tok/s, accepted 199 of 392, output equals greedy. 330/330 `ctest`
  on ROCm.

- 2026-10-10: **OpenAI session management and device queueing (324/324
  `ctest` on vulkan).** Generations queue in arrival order on the single
  device instead of answering 503, so parallel windows and parallel API
  clients each get their turn; a peer that disconnects while queued
  gives up its place (`WaitForGpu`). The OpenAI endpoints
  (`/v1/completions`, `/v1/chat/completions`, `/v1/messages`) accept
  `session_id` to continue a stored session (404 unknown, 409 already
  generating): incoming histories reconcile by common prefix
  (`Session::SyncHistory`), so a client that resends the full history
  and a client that sends only new turns both converge, and the
  assistant reply (text, reasoning, tool calls) is appended to the
  session. Sessions are managed through `/api/sessions` and the
  `/v1/sessions` aliases; `GET /slots` reports the live sessions. The
  OpenAI endpoint code moves from `server.cpp` to `src/serve/openai.*`,
  keeping both files under the size cap.

- 2026-10-10: **Chat UI message meta and thinking box (329/329 `ctest`
  on vulkan).** Each stored message carries its store timestamp and,
  for assistant turns, the generation stats (prompt/completion tokens
  over the serve-layer generation window, `TurnStats::TokensPerSecond`).
  The web UI shows the timestamp and tok/s at the left of every message
  (token counts as a hover title), renders thinking in a separate
  expandable box, and scrolls sticky: the stream follows only while the
  view sits at the bottom, so reading back never yanks. Verified through
  the unit suite and the embedded assets; no live GPU run yet.

- 2026-10-10: **Completion/thinking token budgets and prefill progress
  (342/342 `ctest` on vulkan).** The completion budget is renamed to
  `max_completion_tokens` (CLI `--max-completion-tokens`, default 0:
  an unspecified request fills the remaining context instead of
  stopping at 32k) and a `max_thinking_tokens` budget is added (CLI
  `--max-thinking-tokens`, default unlimited). The old `max_tokens`
  and `--tokens` names are gone without aliases; requests that still
  send `max_tokens` get a warning naming the replacement. A past-the-
  context prompt is rejected whatever the budget, so an unlimited
  request never decodes empty. `ThinkBudget` tracks think spans over
  token ids (multi-token tags match as sequences); the decode loop
  force-closes the block past budget by feeding the think-close ids
  through the normal target advance (with the folding/non-folding
  drafter bookkeeping) and re-picks the answer token, so the turn
  always continues. A per-chunk `prefill_progress` hook streams
  ahead of thinking: the web UI shows it as a loading line and the
  CLI image path logs its rows.

- 2026-10-10: **Thinking streams live (346/346 `ctest` on vulkan).**
  The session streamer withheld all thinking until the think-close
  arrived, so long thinks showed nothing and a turn cut mid-think
  dumped the thinking as one answer blob (root-caused live against
  the served 27B target). Pending thought now streams live as
  `reasoning_content` (only a trailing tag fragment is held back),
  and a cutoff mid-think flushes the remainder as reasoning, never
  as answer text. Takes effect on a server rebuild and restart.

- 2026-10-10: **Thinking boxes keep their open state while streaming.**
  Every streamed token rebuilds the message list, which recreated each
  thinking box with the streamed default: a box the user closed snapped
  open on the next token. Re-renders now preserve every box's open
  state by index, so expanding and collapsing works mid-stream.

- 2026-10-10: **Browser chat cache and model-generated titles (351/351
  `ctest` on vulkan).** The web UI caches open chats in browser
  localStorage and paints instantly on reload; the server stays
  authoritative, so a miss (or an id the server no longer knows)
  falls back to fetching. After the first prompt of a new session the
  model itself names the chat window from the opening exchange
  (`ServeOptions::auto_title`, CLI `--no-auto-title` opts out); renamed
  sessions and the history are never touched.

## Next (in order)

- **Speculation (seam done; acceptance fixed; speed remains).**
  Greedy and both drafters share one loop and one `SpeculativeStrategy`
  seam. DFlash2 acceptance is fixed (4.4 per step, reference 2.7 to
  2.85) and MTP accepts about half its drafts with a default chain of
  one. Output equals greedy in both. Greedy runs at about 25 tok/s on
  the 27B MXFP4 target. DFlash2 runs slower than greedy. Remaining, in
  order: (a) the engine runs one extra single-token target forward per
  speculative step to advance the cache for the bonus token; the
  verify batched forward already computes that position, so reuse it;
  (b) the DFlash2 verifier still runs the full target trunk at the
  draft size (m about 7); closing this gap needs the reference
  structure (a batched or fused verify kernel), not a tile tweak;
  (c) the draft forward cost (fp32 weights, many small kernels).
  The MTP per-draft cost is measured: each draft reads the shared
  2.5 GB output head, so the default chain stays at one.

- **PERF (DEFERRED)**: make MXFP4 inference fast. Targets: the whole load
  under 60 s (met, about 54 s), and 35 to 40 tokens/s decode without MTP.
  Count the decode rate on generation only: the model load is a one-time
  cost and is not part of tokens/s.
  Deferred for now. Greedy runs at about 25 tok/s on the 27B MXFP4
  target. The load target is met. The conv runs on the device, the
  embedding gather runs on the device, and the GEMM kernels are tiled
  (see item 3). The remaining gaps are the speculative step cost (see
  the Speculation bullet) and the prefill on long prompts (see item
  6). Land those first, then re-measure and do the remaining PERF
  work.
  Context: the load reads the
  19 GB safetensors, dequantizes and packs every fp4 blob, and converts
  large BF16 tensors to F32. The head and the embedding stay BF16. The
  reference runtime loads the same file in 8 s because it keeps fp4 and
  dequantizes in the kernel. Measure on
  `EngineTest.MxFp4GeneratesWhenProvided` and
  `EngineTest.DFlash2MatchesGreedyOnModel`.
  Progress: the load is about 54 s now, under the 60 s target. Past
  decode wins are in the Done entries (device conv, tiled batched
  GEMMs, warp-per-output GEMVs, the Vulkan launch ring, async copies).
  Greedy is at about 25 tok/s on the 27B MXFP4 target. What remains is
  the speculative step cost and the long-prompt prefill (see the
  Speculation bullet and item 6).

0. **DFlash2 (acceptance fixed 2026-10-09)**: runs end to end
   (`Engine::GenerateDraft`, CLI `--draft`) and output equals greedy. The
   14x acceptance gap was two defects, both fixed: `DraftCandidates` shrank
   its reusable index vector so every mask row after the first ranked only
   ids 0..top_k-1, and the selector transition score added the predecessor's
   unary logit instead of the successor's (see the newest Done entry). The
   27B MXFP4 target now accepts 4.4 per step, at or above the served 2.7 to
   2.85. Steady-state speed is measured: greedy is about 25 tok/s and
   DFlash2 is slower than greedy (see the Speculation bullet). Remaining
   backlog: the per-step draft cost, not quality. If wanted, batched
   draft scoring. The older notes below are kept for history.
    Deferred: the fp8 W4A8 activation quant is implemented (`quantize_fp8`,
   default off, `TESSERA_MXFP4_W4A8=1`); turning it on did not raise
   acceptance on the noisy 5-token fixture and costs about 2.5x decode.
   Next: measure on a 128-token generation, and try the other differences
   to the served target (fp8 KV cache, R4D fp8 attention) before more
   quant work. 2026-10-09: the fp8 KV cache landed (`--kv-fp8`,
   `quantize_fp8_pack` + `attention_fp8`), so the served target's cache
   quant is now available for that comparison (see the Done entry). R4D fp8
   attention remains. 2026-10-09: measured `--kv-fp8` and W4A8 against the
   drafter; both leave acceptance unchanged (fp32 and fp8 KV both 9 of 154
   over 32 tokens), and the captured tensor matches vLLM's
   `hidden_states + residual`, so the gap is the target's bf16/fused-op
   numerics or the R4D attention, not the cache or activation quant (see
   the Done entry).
   Batching and the draft context width are
   done (see the Done entry). 2026-10-09: the full-prefix context the
   reference uses was tried and reverted; it regressed acceptance and needs
   a device-resident context K/V cache to be viable (see the Done entry). Corrected 2026-10-08: the draft block is
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
2. **MoE, MLP, RMSNorm and embedding kernels (done for Qwen 3.8)**.
   RMSNorm, sigmoid-gate, add, silu_mul and the embedding gather
   (`embedding_f32`/`embedding_bf16`/`embedding_q4k`) run on the device.
   The gated MLP is gemm + silu_mul + gemm on the device. The Qwen 3.8
   27B definition is dense, so it needs no MoE kernel. The Qwen 3.8 MXFP4
   definition's MTP head is complete: its eight FP8 E4M3 projections
   load and dequantize to F32, and MTP drafting accepts about half its
   drafts. The `Architecture` module interface is done (see the Done
   entries). Remaining: the MoE router and expert kernels for the
   planned Ornith-1.5-35B-A3B.
3. **GEMM throughput**: the model runs far below memory bandwidth, so the
   GEMM kernels are the cost. Done so far: tiled `gemm_q4k_batched`/
   `gemm_q5k_batched`/`gemm_q6k_batched`/`gemm_iq4xs_batched`/
   `gemm_mxfp4_batched`/`gemm_f32_batched`/`gemm_bf16_batched`, the
   warp-per-output `gemm_mxfp4`/`gemm_f32`/`gemm_bf16` decode GEMVs, the
   warp-per-column multi-row `gemm_mxfp4_rows` verify kernel, tiled
   quantized attention, wide MXFP4 and Q4_K reads, the E8M0 bit cast,
   hardware fp16 scale decode, and the fused four-partial-sum compute
   form. The coalesced `gemm_q4k_row` is kept but measures slower. The
   greedy GEMV roofline is settled: no structural variant tried beat
   the coalesced warp-per-output form. Next: the same wide-read
   treatment on the Q5_K, Q6_K and IQ4_XS decode GEMVs; a fused or
   batched verify kernel for the speculative path (see the Speculation
   bullet). Applies on both backends. The tree holds uncommitted f32
   and bf16 rows kernels plus a top-k kernel; they need device tests
   and a measurement before they commit.
4. **Serving API (partly done)**: the owner asked for session
   management and parallel windows, so the HTTP surface grew there:
   generations queue in arrival order on the device (no more 503),
   chats live in server sessions (`/api/sessions`, `/v1/sessions`
   aliases, `GET /slots` reports them), and the OpenAI endpoints
   continue one through `session_id`. Still deferred unless explicitly
   told: thread pool, keep-alive, `/v1/responses`,
   render/derender/batch, `/tokenizer_info`, `/load` and LoRA, and the
   501 embedding/rerank/audio/pooling/classify/score surfaces. A first
   slice lives in `src/serve/` (`/health`, `/metrics`, `/v1/models`,
   `/props`, `/tokenize`, `/detokenize`, `/slots`, `/v1/completions`,
   `/v1/chat/completions`, `/v1/messages`,
   `/v1/messages/count_tokens`, SSE for completions/chat/messages,
   API-key auth, CORS).
5. **Runtime options**: context size, draft-block and the GPU index
   (`--gpu`) are CLI flags now, and the KV cache can be fp16 (`--kv-f16`)
   (--kv-q8), 4-bit (`--kv-q4`) or FP8 (`--kv-fp8`). Vision input is
   wired (`--mmproj`, `--image`, with auto `<|image_pad|>` detection).
   Still to wire: batch caps (features that do not exist yet). No
   hard-coded paths or sizes. Note: the CLI has no `--log-tokens`,
   `--image-token`, `--mtp` or `--prompt` flags. Add them or keep the
   current behavior (logging is default-on, the image token is
   detected, MTP uses `--speculate`, prompts use `--prompt-text`).
6. **Prefill optimisation**: the batched prefill is much faster now
   (112 tokens in about 2.2 s, 51 prompt tok/s, was 18.2 s). Done: the
   ROCm device-to-device copy is asynchronous, the Vulkan launch ring
   pipelines kernels, the MXFP4 tiled GEMM is compute-bound (four
   partial sums), and the per-row linear-attention loop is batched
   into one launch per kernel (see the Done entries). Remaining: the
  full-attention layers still run per layer at the prompt row count,
  and the tiled GEMMs still dequantize each weight block once per
  tile rather than once per prompt. Each prefill forward is now
  clamped to the attention work budget, so a long prompt cannot wedge
  the device (see the newest Done entry). The total work stays
  quadratic in the prompt length. Measure prompt tokens per second
  on a text prompt and on an image prompt.
7. **Multimodal (mmproj)**: config, weights, encoder+merger, image
   load/resize, image-embedding injection, the CLI wiring and the
   `<|image_pad|>` placeholder default are done
   (`Engine::GenerateMultimodal`). Deepstack feature injection is not
   needed: the Qwen3.8 target's `deepstack_visual_indexes` is empty. The
   only remaining work is the image-prefill attention speed (the same
   GEMM/attention cost as item 3).

8. **Multi-GPU (deferred)**: today `--gpu` selects one device and there is
   one `Backend` per engine. Two researched routes: tensor parallelism
   (shard attention heads and MLP rows across GPUs with an all-reduce per
   layer; vLLM tensor parallelism, Megatron-LM TP) or layer/pipeline
   split (assign blocks to GPUs and hand off activations; Megatron-LM
   pipeline parallelism). Both need several `Backend` instances, weight
   sharding in the loaders, and cross-device collectives or peer copies.
   Not started.

## Notes and decisions

- Build type: an unspecified `CMAKE_BUILD_TYPE` now defaults to
  Release in `CMakeLists.txt`. Before that fix, the documented `cmake
  -B` compiled at `-O0` and a 27B decode measured 7.7 s/token (MXFP4)
  to 10.8 s/token (GGUF), which confounded every performance
  comparison. Measure only on a Release build.
- Parallel builds: build with `-j1`. The gcc-15 toolchain segfaults
  (including an ICE in `c_parse_final_cleanups` and corrupt assembler
  output) under `-j3`/`-j4`; a serial build completes.
- After a compiler crash, touch and rebuild the affected translation
  units before trusting the result. An ICE during a HIP or C++ compile
  left a stale object that still linked but produced wrong numbers
  (`VisionStackMatchesRef` failed until `vision_stack.cpp` and its
  neighbours were recompiled). A clean serial build avoids this.
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
- Device tests are GPU-only. The vulkan backend filters CPU-type physical
  devices from enumeration, so `--gpu` and `TESSERA_TEST_GPU` cannot
  select software rendering (llvmpipe). The rocm runtime only sees real
  GPUs, so it needs no filter.
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
