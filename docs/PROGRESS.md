# Progress

This file tracks tessera development. After each milestone, update
"Current status" and "Next" so both match reality. See AGENTS.md.

Latest suite: 421/421 `ctest` on vulkan. 421/421 `ctest` on ROCm.
Verified on AMD Radeon AI PRO R9700 (vulkan through RADV GFX1201,
rocm through the system ROCm) and on AMD Radeon RX 7900 XTX (vulkan
through RADV GFX1100, rocm through the system ROCm).

## Current status

- Project layout and CMake. One backend per binary, selected at
  configure time (`TESSERA_BACKEND=vulkan|rocm`).
- Public API: `Engine`, `Model`, `Backend`/`Buffer`,
  `SpeculativeStrategy`, `Diagnostics`.
- GGUF v2/v3 parser. It parses the header and tensor manifest.
  It checks all bounds. Small metadata arrays stay retained.
  Bulk arrays stay dropped, except tokenizer definition arrays.
- Safetensors layout checks for MXFP4 model directories. A bounded
  schema-strict JSON reader parses the tensor map with no third
  party dependency. MXFP4 blobs pair with E8M0 scales by name.
  MTP FP8 weights map natively. BF16 weights map natively.
- Vulkan and ROCm backends: device init, buffer alloc/free,
  H2D/D2H copy, D2D copy, synchronize. `CopyD2HAt` reads byte
  slices for embedding rows.
- Model load reports one timing line per phase through the engine
  diagnostics (`model: phase 'parse gguf' done in 25 ms (866
  tensors)`, then config, device allocation, weight upload,
  tokenizer). The ROCm batched weight upload stages through a
  bounded pinned window (256 MiB) with one stream sync per chunk
  and releases the window after the transfer. Before, hipMemcpyAsync
  received the pageable mapped file directly, the runtime's pinned
  set grew to the whole checkpoint, and a 16.4 GB GGUF load ran past
  five minutes on an RX 7900 XTX. The same file now loads in about
  8 s, with the upload at about 2.1 GiB/s. Covered by a load test
  that checks the phase lines and a test for the shared transfer
  summary; the existing copy tests cover the staged upload on both
  backends.
- Kernel launch plumbing on both backends. `LaunchKernel` binds
  buffers and 64-bit scalars in parameter order. Vulkan defers the
  submission: every dispatch and device copy of a decode step records
  into one command buffer (with an inter-operation barrier), so a step
  that issues about two thousand small launches pays one queue submit
  instead of thousands; host readbacks flush and wait. ROCm submits
  on one stream whose synchronization policy is
  `hipSyncPolicyBlockingSync`, so host waits sleep instead of busy
   waiting in the runtime. Covered by the launch and copy device
   tests on both backends.
- Quantized KV append is row-parallel. `quantize_q8`, `quantize_q4`
   and `quantize_fp8_pack` are one workgroup per row: the workgroup
   reduces the row absmax in shared memory and packs the codes
   together. `QuantizeRowDevice` launches a 256-thread workgroup, not
   one thread. Before, a single lane walked each row serially, so an
   fp8 KV verify step paid 16 full-attention layers times 2 tensors
   times the block width in single-thread launches; the profiler put
   `QuantizeFp8PackKernel` at 443 of 1115 ms of kernel time at 231 us
   per row. On the 27B MXFP4 + DFlash2 target, fp8 KV decode went
   from 132.6 to 76.3 ms/step (29 to 50 tok/s) and greedy fp8 from 50
   to 43 ms/token, matching the fp32 KV path. The code, scale and
   acceptance are unchanged. Covered by the quantize device tests on
   both backends, including `QuantizeKvRowsWideMatchesRef` (a row
   wider than the workgroup).
- Generic GEMM kernels with dequantization and fp32 accumulation:
  Q4_K, Q5_K, Q6_K, Q8_0, IQ4_NL, IQ4_XS, IQ3_S, F32, BF16, per-row
  scaled FP8 E4M3, block-scaled FP8, and MXFP4 (32-element E8M0
  blocks). Tiled batched kernels cover the hot formats.
  Warp-per-output GEMVs cover decode. A warp-per-column
  multi-row MXFP4 kernel covers small verify batches. Tensor-core
  paths cover the MXFP4 verify (fp8 WMMA) and the wide BF16 head.
  All verified against host references with per-backend tolerance.
- Generic kernels with host references on both backends: GQA
  attention (causal flag, sliding window, fp16/int8/4-bit/FP8 KV
  variants), RoPE, multimodal RoPE, RMSNorm, LayerNorm, GELU,
  L2 norm, gated RMS norm, sigmoid gate, add, silu_mul, bias_add,
  cast f32 to f16, quantize q8/q4/fp8, fused qgate split,
  repeat_heads, ssm_gate, causal conv1d, conv1d_step,
  gated delta-rule scan, delta_step_heads, dflash_conv,
  selector_edge_score, concat_features, spatial_merge,
  image_patchify, top_k_rows device argmax.
- Single-token decode loop in the core, backend agnostic. It runs
  vanilla GGUF models and hybrid models (gated full attention and
  recurrent linear attention). Activations stay on the device.
  Only the causal conv state and final logits touch the host.
- Public generation API: `Engine::Generate` (greedy reference
  path) and `Engine::GenerateStreaming` (token by token).
  Sampling supports temperature, top_k, top_p, min_p, repetition
  and presence penalties, with a seed. Greedy is the reference
  baseline. Output with any strategy on equals greedy output.
- Speculative decoding behind one `SpeculativeStrategy` seam, off
  by default. MTP drafting and DFlash2 drafting share one engine
  loop with greedy. The CLI attaches the chosen strategy to the
  engine, so `--speculate` (MTP) and `--draft` (DFlash2) both drive
  `run` and every served turn. DFlash2 loads the real FP8 draft,
  captures target hidden states at the draft layer ids, drafts a
  mask-token block, re-ranks candidates with the selector, and
  verifies with a batched target forward. MTP chain default is one.
- Measured on the 27B MXFP4 target (ROCm, GPU1): greedy about
  23 tok/s (43 ms/token, fp8 WMMA path). DFlash2 about 60 tok/s
  (75 ms/step, 256 tokens, 199 of 392 accepted). Output equals
  greedy. `TESSERA_MXFP4_WMMA=0` selects the all-scalar fallback
  for backends without fp8 tensor cores. Model load is about 54 s,
  under the 60 s target.
- GGUF load is now fast. The 27B Q4_K_M GGUF (866 tensors, 16.4
  GB, MTP head included) loads in 2.6 s on ROCm and 5.4 s on
  Vulkan, down from 12.3 s and 17.2 s. The loader memory-maps the
  file instead of reading it into a zero-filled host vector (that
  alone was 7 s of memset), and a batched `CopyH2DBatch` upload
  drains all tensors with one synchronization per backend chunk
  instead of one per tensor.
- KV cache types: fp32 (default), fp16, symmetric int8, symmetric
  4-bit, FP8. Selected by option (`--kv-f16`, `--kv-q8`,
  `--kv-q4`, `--kv-fp8`). Caches are preallocated and grow
  geometrically. Prompt prefill skips the output head. Prefill is
  chunked and each forward is clamped to an attention work budget.
- Byte-level BPE tokenizer from GGUF or HF checkpoint files. It
  matches the reference on ASCII, Latin, CJK, and code. Decode
  round-trips. `Model::GetTokenizer` exposes it. Not applied: NFC
  normalization, special-token matching, combining marks, emoji.
- Chat-template renderer: a compact Jinja2-subset engine that runs
  the model GGUF or HF chat template. It matches reference HF
  Jinja2 output on the 27B template, thinking on and off.
- HTTP serving: blocking HTTP/1.1 server with buffered and chunked
  (SSE) responses, API-key auth, CORS allowlist. Endpoints:
  `/health`, `/metrics`, `/v1/models`, `/props`, `/tokenize`,
  `/detokenize`, `/slots`, `/api/sessions` and `/v1/sessions`
  aliases, `/v1/completions`, `/v1/chat/completions`,
  `/v1/messages`, `/v1/messages/count_tokens`. Generations queue
  in arrival order on the single device. Chats live in server
  sessions. OpenAI endpoints continue a session through
  `session_id`. Tool calling and live thinking streams work.
  The `/v1/chat/completions` reply splits the model's thinking into
  `reasoning_content` (the chat template opens `<think>` in the
  prompt, so the streamer recovers the reasoning before the closer)
  and the answer into `content`, matching the reference runtime; the
  streamed path emits live `reasoning_content` and `content` deltas.
  The web UI keeps one live stream per chat, so switching chats
  while one generates shows the open chat and queues another turn.
  Completion and thinking token budgets are enforced. The reply
  `finish_reason` is accurate: "stop" when the model emits a declared
  stop token, "length" when the completion budget runs out, and
  "tool_calls" on a tool call (the engine returns the reason via
  `GenerateOutcome`; the streamed path ends with a final
  `finish_reason` chunk before `[DONE]`).
  The server binds before the model finishes loading: `/health`
  answers 503 with a JSON `status`/`detail` body until the model is
  loaded and its kernels warmed, then 200. The web UI polls it and
  keeps the composer disabled until ready. Unimplemented surfaces
  return 501.
- Chat-template arguments pass through from a request:
  `chat_template_kwargs` (any key) plus a top-level `reasoning_effort`
  reach the renderer unchanged, so a caller sets the difficulty the
  model template expects. `/props` reports `reasoning_efforts`, the
  levels the template validates (for example `xhigh`, `medium`,
  `low`), and the web UI builds its difficulty selector from that
  list. The template's own default is left in place: `xhigh` on the
  Qwen3.8 template injects a "think exhaustively" system message that
  drives greedy decode into a repetition loop, so selecting `medium`
  gives the short reference-style thinking. Covered by
  `ServeTest.RequestTemplateKwargs*` and
  `ServeTest.ReasoningEffortsDetectedFromTemplate`.
- Vision: mmproj config and weight load, CLIP encoder stack,
  merger into language space, PPM load and resize, embedding
  injection at `<|image_pad|>` placeholders,
  `Engine::GenerateMultimodal`, CLI wiring.
- CLI: `run` (model load, tensor summary, greedy and sampled
  decode, `--prompt-text`, `--max-completion-tokens`,
  `--max-thinking-tokens`, `--context`, `--draft-block`,
  `--gpu`, `--kv-*`, `--speculate`, `--draft`, `--mmproj`,
  `--image`, `--quiet`), `serve`, `--list-gpus`. Input and output
  text logging is on by default. There are no `--log-tokens`,
  `--image-token`, `--mtp`, `--prompt`, `--tokens`, or
  `--max-tokens` flags. Opt-in env toggles: `TESSERA_MXFP4_W4A8=1`
  (off, no acceptance gain), `TESSERA_TARGET_BF16=1` (off).
- Calibration (docs/CALIBRATE.md). A `calibrate` subcommand loads
  the model once, warms it, sweeps the hardware dependent settings
  (the MXFP4 fp8 split-K target and cap on ROCm, the prefill chunk on
  both backends, the single-token attention split at long context,
  and the draft block and draft context when a draft is attached),
  applies the pick rule (keep a candidate only when it beats the
  default by more than MIN_GAIN 0.03) and confirms a winner with an
  interleaved re-measurement (three rounds each, median). The prefill
  candidates are ascending multiples of 256 from the 512 default
  (512, 768, 1024, 1536, 2048); decode settings sweep a 64-token
  prompt, the attention split and the prefill chunk a long prompt, so
  the run stays within its cap. The tiled-GEMM dispatch thresholds
  and the prefill attention pair budget are exposed as environment
  overrides for measurement but are not swept: the thresholds are
  coupled and the budget is a driver-hang safety clamp. The command defaults the KV cache to int8 (kv8) and reports the
  maximum context length the device allows for that cache type
  (free memory minus a 2 GiB reserve, divided by the per-token KV
  bytes over the full-attention layers). It writes a JSON calibration
  file keyed by backend, device, model, context, KV type and
  strategy, and reports the backend used. It prints a formatted
  report (the identity, the sweep, the chosen settings and an example
  `run` command that reproduces them). `run` and `serve` accept
  `--calibration <file>` (or `TESSERA_CALIBRATION`): a saved entry is
  applied unless an explicit flag overrides it. The sweep, candidate
  lists, pick rule, hardware key, context bound and file reader/writer
  live in one module (`src/core/calibrate/`) behind
  `include/tessera/calibrate.hpp`; the CLI is a thin front end. The
  MXFP4 split-K target and cap moved from environment-only reads to
  `GenerateOptions`, carried through the request tuning into the
  architecture state (`TESSERA_MXFP4_SPLIT`/`SPLITCAP` still win as a
  diagnostic override). Measured on the 27B MXFP4 target (ROCm,
  GPU0, int8 KV, the default three-prompt set, 12m19s): the default
  split target 320, split cap 4, prefill chunk 512 and attention
  split 16 were all kept (every candidate inside the noise floor;
  prefill held at about 73 tok/s for every chunk size), 23.9 decode
  tok/s, 73.7 prefill tok/s. The reported memory bound for the int8
  cache is 359872 tokens. A calibrated run
  produces the same tokens as the default. Covered by `CalibrateTest`
  (stand-in engine sweep, candidate/pick/key/file/context units, and
  a device test that confirms a calibrated split matches the default
  within the per-backend tolerance).
- Architecture modules: one `Architecture` module per model
  family (`src/models/qwen3_5/`). Core names no model. The Qwen3.5
  module handles HF config parse, weight rename, value-head
  reorder, Gemma norm offset, and the MTP and verify paths.
- Single GoogleTest target. Device tests skip cleanly with no
  device. Numerical checks use per-backend tolerance.

## Done (grouped summary)

- Foundation: layout, CMake, public API, GGUF parser, safetensors
  checks, backend device plumbing, DFlash2 skeleton, CLI, tests.
  Suite green on both backends from the first week.
- Launch plumbing: SPIR-V pipelines and push constants on Vulkan,
  HIP kernels on ROCm, verified by a fill read-back test.
- GEMM coverage: Q4_K through Q8_0 plus IQ formats, F32, BF16,
  FP8 row-scaled and block-scaled, MXFP4. Exact codec unit tests.
  Fixed Q8_0 block size, Q3_K scale hoisting, and scalar argument
  order along the way.
- Attention and RoPE: GQA and NeoX RoPE driven by model metadata,
  verified against host references on both devices.
- Norms and gates: RMSNorm, sigmoid gate, add, silu_mul, L2 norm,
  gated RMS norm, qgate split, repeat_heads, ssm_gate. These wire
  the gated full-attention and linear-attention paths.
- Recurrence: causal conv1d, conv1d_step, gated delta scan step
  and multi-head step, chained-state tests on both devices.
- Multimodal RoPE: sectioned pairs with a global frequency
  schedule. Text rows reduce to plain RoPE.
- Weight upload: one device buffer per manifest tensor, exposed
  through `Model::Weights` with lookup.
- Decode loop: embed, block forward, greedy sample. First
  host-orchestrated, then device-resident for vanilla and hybrid.
  Pinned fixture baselines flag kernel or scheduler drift.
- 27B coherence: fixed the linear head-repeat map and the query
  scale. Output matched the reference continuation.
- MXFP4 target: HF config, tokenizer, and chat template parse.
  Loader packs F4E2M1 with E8M0 scales. Greedy matches the GGUF
  reference and stops at declared stop tokens.
- MTP head: RMSNorm, concat, eh_proj, one full-attention block,
  shared head. MTP speculation accepts about half its drafts.
- Verifier: full-vocab logits, token scoring, greedy verification
  loop, then batched speculative scoring with cache rollback.
  Prefill skips the output head. KV caches grow geometrically.
- DFlash2 draft: config parse, grouped conv, MLP half, attention
  half, full layer, stack, selector block, target fusion, context
  projection and attention, draft block, weight loader, candidate
  extraction, mask embeddings, target hidden capture, end-to-end
  speculative generation with output equal to greedy.
- DFlash2 acceptance: fixed the candidate index shrink bug and the
  predecessor/successor logit bug. 4.4 tokens per step on the 27B
  MXFP4 target, at or above the served 2.7 to 2.85.
- DFlash2 numerics: greedy and verify share the fp8-WMMA MXFP4
  kernel, so output stays equal to greedy at 128 tokens and more.
  Wide BF16 head runs on the tensor-core kernel. Verifier argmax
  runs on the device.
- Sampling: penalties, temperature, min_p, top_k, top_p, seeded
  draw. CLI `--sample` and parameter flags.
- KV options: fp16, int8, 4-bit, FP8 kernels plus storage wiring.
  Lossy modes differ from fp32 by design.
- Vision: mmproj config, LayerNorm and GELU, patchify, non-causal
  attention, CLIP block and stack, merger, model load and encode,
  embedding injection, image load and resize, multimodal
  generation and CLI wiring.
- Serving: HTTP layer, SSE, auth, CORS, endpoints listed above.
  Session management, device queueing, history reconciliation,
  message stats, thinking boxes, live thinking streams, token
  budgets, prefill progress, browser cache, model-generated
  titles, tool calling, disconnect handling.
- Performance: tiled GEMMs, decode GEMVs, multi-row verify GEMM,
  tiled attention, scan and norm ILP, Vulkan launch ring, async
  ROCm copies, cached host weights, chunked and bounded prefill.
  Greedy 23 tok/s. DFlash2 60 tok/s. Load 54 s (MXFP4).
- Weight load: the GGUF file memory-maps (`MappedFile`) instead of
  reading into a `std::vector` that value-initializes (zeroes)
  every byte; for the 16.4 GB 27B file that removed a 7 s memset
  and the anonymous 16 GB host buffer. A batched
  `Backend::CopyH2DBatch` uploads every tensor with one
  synchronization per chunk: ROCm issues `hipMemcpyAsync` on the
  null stream and syncs once, Vulkan records a whole chunk into one
  command buffer and submits it once (bounded 256 MiB staging
  window, so host-visible memory stays bounded). 27B Q4_K_M GGUF
  load: ROCm 12.3 s to 2.6 s, Vulkan 17.2 s to 5.4 s. Covered by
  `BackendTest.CopyH2DBatchUploadsEachBuffer` and
  `GgufTest.ParsesFileFromDisk` on both backends; the real MTP
  speculation still equals greedy.
- Attention tile reduction: each 256-key tile now computes its
  max and softmax weights once on one thread and shares them.
  Before, all 256 threads scanned the tile, so a long context
  paid 256 scans per tile. Same order, same numerics. Covered
  by multi-tile device tests on both backends.
- Attention ILP: the score dot, the tile weight sum and the
  value accumulation keep four independent partial sums per
  thread instead of one dependent chain. The chain stalled on
  its own latency. Q8 attention at 32k keys drops from 8.4 to
  4.8 ms per layer and fp32 from 13.1 to 7.7 ms. Order changes,
  so results match within tolerance and DFlash2 still equals
  greedy on the 27B target.
- Split-N (flash-decoding) attention for the single-token path.
  One workgroup per (query row, head) leaves the GPU mostly idle
  at the greedy row count and runs one serial 128-tile online
  softmax, so long context is latency-bound. The new
  `attention_split`/`attention_q8_split`/`attention_q4_split`/
  `attention_fp8_split` kernels split the key range into 16
  chunks (one partial acc/max/sum each) and `attention_combine`
  merges them. The engine splits only the m = 1 path with
  n >= 1024; batches (prefill, DFlash2 verify at m = 7, the
  draft) keep the tiled kernel, which is faster there. Measured
  4.5x faster at m = 1, n = 32768 (3.0 to 0.8 ms per layer Q8),
   and bit-exact: 24 greedy tokens at n = 1170 are byte-identical
   with the split on and off, and DFlash2 still equals greedy.
   Covered by split device tests on both backends.
- GPU-bound decode: generation CPU dropped toward idle. Vulkan
  defers the submission of a whole decode step (one queue submit,
  inter-operation barriers), and the greedy step argmaxes on the
  device (top_k_rows) instead of downloading the vocabulary. ROCm
  runs a blocking-sync stream and stages every host readback/upload
  through reusable pinned memory. The greedy non-speculative loop
  also stops downloading the hidden state, which no drafter reads.
  Measured on the pelican SVG prompt, 27B MXFP4, greedy: Vulkan
  decode CPU 26.5 to 6.8 ms/token (wall 51 to 40 ms/token); ROCm
  main-thread decode CPU 76 to 8.4 ms/token and process CPU 82 to
  48 ms/token. The ROCm residual is a HIP runtime background thread,
  not tessera's thread: a ptrace sample of the spinning thread's RIP
  lands in libc `ioctl` with `request=0xc0184b0c`, which is
  `AMDKFD_IOC_WAIT_EVENTS` (KFD type 'K', nr 12, 24-byte args). The
  runtime busy-polls the KFD event wait while the GPU runs (the
  sync-policy change only stops the main thread from polling too).
  `ROC_ACTIVE_WAIT_TIMEOUT`, `ROC_CPU_WAIT_FOR_SIGNAL`,
  `HSA_ENABLE_INTERRUPT`, `HSA_ENABLE_MWAITX`, and `HSA_ENABLE_SDMA`
  do not change it, and microbenchmarks do not reproduce it.
  Weight lookups are
  indexed by name, so a per-op lookup no longer scans every tensor.
  Output is unchanged: the MXFP4 greedy tokens are identical on
  Vulkan and ROCm, and greedy text on the 27B target still matches.
  `Model::FindDeviceTensor` is covered by
  `EngineTest.LoadGgufModelIndexesEveryWeightByName` and the extended
  `EngineTest.LoadGgufModelUploadsWeights`.
- Multi-row GGUF GEMV, the folding MTP verify's missing half. The
  verify batch is `[anchor, drafts...]`; a 2- or 3-row batch ran the
  per-row GEMV m times, so the verify cost as much as m single forward
  passes. `gemm_q4k_rows{r}`, `gemm_q5k_rows{r}`, `gemm_q6k_rows{r}` and
  `gemm_iq4xs_rows{r}` on both backends put one 32-lane group on one
  weight column and decode each quant block once for all `r` rows
  (`r` = 2, 3, 4, a template parameter so the row loop unrolls; the
  runtime-row version was twice as slow on ROCm and 1.6x slower on
  Vulkan, so the Vulkan shaders ship as one fixed-row shader per count).
  A per-path split-K pass (`ProjectGemvRowsDevice`) raises the workgroup
  count for narrow output counts. Measured with a Q4_K benchmark at
  n=10240, k=5120: a 2-row pass dropped from 0.146 to 0.098 ms on ROCm
  and from 0.129 to 0.082 on Vulkan. The MTP default chain is now two
  drafts, and the 2-row verify costs about 1.1x a single-row forward on
  ROCm (1.3x on Vulkan).
- Vulkan GEMV byte access. The Vulkan decode shaders read the weight
  bytes through 8-bit storage buffers (`GL_EXT_shader_8bit_storage`,
  Vulkan 1.2) instead of a word load plus a shift per byte, and reduce
  with a 32-lane subgroup clustered add instead of a shared-memory tree.
  The Q6_K vocab head shader dropped from 2.38 to 1.88 ms (438 to 553
  GB/s) and the greedy forward from 59 to 54 ms; the Q4_K vec GEMV is
  0.107 to 0.087 ms. Vulkan now needs a Vulkan 1.2 device.
- GGUF MTP now beats greedy. On the 27B Q4_K_M GGUF (7900 XTX), decode
  with `--speculate` (two drafts) reached 37 tok/s on a repetitive loop,
  34 on a 120-token planet list and 31 on a 128-token paragraph on ROCm,
  and 31, 30 and 26 on Vulkan, against 18 tok/s greedy on both. A short
  factual prompt that stops after 25 tokens is 24 on both. Acceptance is
  identical on the two backends (30 of 36, 71 of 96, 51 of 88) and output
  equals greedy token for token.
- Split-K tuning: the target workgroup count dropped from 4096 to 2048
  blocks, measured 6-8% faster end to end (a higher split pays more in
  the reduce pass than it gains).
- Decode phase profiler, opt-in and off by default. `TESSERA_PROFILE=1`
  accumulates wall-clock per phase in the decode loop (anchor forward,
  target forward/head/logits download, draft prep/block/head/argmax,
  verify) and logs a per-phase summary after the decode line;
  `TESSERA_PROFILE=2` adds the per-layer phases (full-attention layer,
  linear-attention layer, FFN) with a synchronization per phase, which
  perturbs the pipeline but attributes the trunk. `src/core/profile.hpp`
  owns the accumulator and the RAII scope; `DecodeCache` carries the
  pointer. It found the GGUF decode culprits on the 7900 XTX: the
  one-workgroup-per-output GEMVs streamed at 40-250 GB/s (the Q6_K
  vocab head alone was 26 ms per pass at 40 GB/s) and the Q4_K/Q5_K/
  IQ4_XS small-n projections were under one wave of workgroups.
- Warp-per-output GEMV kernels: `gemm_q4k_vec`, `gemm_q5k_vec`,
  `gemm_q6k_vec` and `gemm_iq4xs_vec` replace the scalar GEMVs for the
  four hot GGUF formats at decode batch sizes. One 32-lane group
  computes one output element (eight per 256-thread workgroup), every
  lane reads its eight elements of each quant block with aligned loads,
  and the partials reduce with five warp shuffles (Vulkan: the same
  mapping with a shared-memory tree of the same association order).
  The Q6_K head went from 26.0 ms at 40 GB/s to about 1.3 ms at
  780 GB/s; the FFN-shape Q4_K GEMV from 134 to about 390-480 GB/s
  (measured cold, one pass per buffer). A follow-up split-K pass
  (`GemmVecSplit`, `gemm_vec_reduce`) raises the workgroup count for
  narrow projections; deterministic, no atomics. Measured on the 27B
  Q4_K_M GGUF (866 tensors) on the 7900 XTX: greedy decode 7 to
  18 tok/s on ROCm and 8 to 15 tok/s on Vulkan; the MTP strategy
  equals its acceptance but stays at parity with greedy until the
  verify batch has a multi-row GEMV (see Next).
- MTP matches the reference cycle. The strategy now folds the anchor
  into the verify batch (`FoldsAnchor`), pairs the anchor token with
  the trunk hidden one position before it, and writes the MTP KV at
  the anchor's own slot (`pos`, not `pos + 1`), which is the contract
  hipfire/llama.cpp implement (CREDITS.md). Chained drafts feed the
  post-FFN MTP residual back, not the normed hidden. The engine seeds
  the first anchor from the prefill's final hidden
  (`NeedsPrefillAnchor`), re-records the anchor after every verify
  (the architecture verify now fills the hidden for the anchor-only
  case), and the generic single-draft verify returns the anchor's
  hidden so the pairing survives a rejection. Acceptance on the 27B
  Q4_K_M GGUF went from 8 of 17 to 23 of 24 on a repetitive prompt
  (19 of 28 before on the same prompt) and from 8 of 17 to 10 of 15
  on a short factual prompt; both backends agree token for token.
  MTP still runs one target forward per emitted token (the verify is
  not batched), so it matches greedy rather than beating it.
- Row-parallel quantized KV append: `quantize_q8`, `quantize_q4` and
  `quantize_fp8_pack` are one workgroup per row (shared-memory absmax),
  and `QuantizeRowDevice` launches a workgroup. The old one-thread
  launch made a single lane walk each row and dominated fp8 KV decode.
  Measured on the 27B MXFP4 + DFlash2 target (ROCm, GPU0, fp8 KV):
  132.6 to 76.3 ms/step and 29 to 50 tok/s, against the served g1a
  reference at 71.9 ms/step and about 47 tok/s. Greedy fp8 went from
  50 to 43 ms/token, matching fp32. Codes and acceptance are
  unchanged. Covered by the quantize device tests on both backends.

## Next (in order)

- Vulkan GEMV speed. The 8-bit storage buffers and fixed-row shaders
  closed most of the gap (a paragraph is 26 tok/s against 31 on ROCm),
  but the Vulkan batch verify is still 1.3x the ROCm one (73 against 57
  ms) and its GEMV kernels stream at about 85% of the ROCm rate. Next:
  measure the Vulkan non-GEMM batch work (GDN and attention row loops)
  and the per-dispatch barrier overhead in the batched forward.
- Speculative cost. The verifier still runs the full target trunk
  at the draft size. The draft forward still costs more than
  greedy (fp32 weights, many small kernels, per-layer context K/V
  projections). Next levers: fuse the draft context K/V with the
  q/k/v projections as the reference does, and raise small-n GEMM
  occupancy. The MTP chain default stays at one (each draft reads
  the 2.5 GB output head). Split-N flash decoding landed for the
  m = 1 path (see Current status); the verify still runs the tiled
  kernel at m = 7, where a batched or fused verify kernel is the
  remaining speed work.
- GEMM throughput. Next: wide-read treatment on the Q5_K, Q6_K,
  and IQ4_XS decode GEMVs. The tree holds uncommitted f32 and bf16
  rows kernels plus a top-k kernel. They need device tests and a
  measurement before they commit.
- Prefill. Full-attention layers still run per layer at the prompt
  row count. Tiled GEMMs still dequantize each weight block once
  per tile, not once per prompt. Total work stays quadratic in
  prompt length. Measure prompt tok/s on text and image prompts.
- Runtime options. Calibration landed (see Current status): the
  hardware dependent settings (split-K target, prefill chunk, draft
  block) are measured and persisted. Still to wire: batch caps for
  features that do not exist yet. No hard-coded paths or sizes.
- Serving (deferred unless the owner asks). Still deferred: thread
  pool, keep-alive, `/v1/responses`, render/derender/batch,
  `/tokenizer_info`, `/load` and LoRA, 501 embedding/rerank/audio
  surfaces.
- Multimodal. Remaining: image-prefill attention speed (same GEMM
  and attention cost as above). Deepstack injection is not needed.
- R4D FP8 attention. The served target difference not yet matched.
  W4A8 activation quant and the FP8 KV cache were measured and left
  acceptance unchanged, so they stay off by default.
- Multi-GPU (deferred). One `Backend` per engine today. Routes:
  tensor parallelism or pipeline split. Both need sharded loaders
  and cross-device copies. Not started.

## Notes and decisions

- Build type: an unset `CMAKE_BUILD_TYPE` defaults to Release.
  Measure only on a Release build.
- Parallel builds: build with `-j1`. The gcc-15 toolchain
  segfaults under higher parallelism.
- After a compiler crash, touch and rebuild the affected units
  before trusting the result. Stale objects have produced wrong
  numbers. A clean serial build avoids this.
- HTTP serving is deferred. Do not add endpoints, SSE variants,
  auth changes, or concurrency unless the owner asks.
- `std::expected` is mandated by AGENTS.md. The project compiles
  with `-std=c++23`. The code stays C++20 style.
- One backend per binary (`TESSERA_BACKEND`). Verify on the
  backend being changed. State which backend was used.
- Tests: single target (`tessera-tests`), no new top-level test
  files. Device tests skip cleanly with no device.
- Device tests are GPU-only. Vulkan filters CPU physical devices.
  ROCm sees real GPUs only.
- No model names in tests. A qwen shaped test covers the generic
  path with qwen shaped parameters.
- Calibration and the split-K target. The fp8 tensor-core MXFP4
  split-K reassociates the fp32 accumulation, so a different split
  target moves the logits within the fp8 reference-exponent scheme's
  own noise (measured on the 27B MXFP4 target: the plain scalar
  accumulation already differs from the default tensor-core path by
  about 0.5 max absolute, and a split change by up to about 0.9) while
  the greedy token is unchanged. `CalibrateTest.CalibratedSplitMatches
  DefaultWithinTolerance` asserts the produced tokens are identical and
  the first-step logits stay within a per-backend bound (exact on
  Vulkan, where the split is inert). The split target is a speed knob
  only: the produced tokens are the reference.
- Logging: `tessera::log::Diagnostics` with a pluggable sink.
  Every line carries `prefix: message` context.
- Kernel contract: `grid_*` is the workgroup count on both
  backends. Workgroup size is fixed at compile time on Vulkan.
  Scalars bind as 64-bit values.
- Vulkan shaders embed at configure time. Editing a `.comp` file
  needs a fresh `cmake -B` run.
- Reference runtime order: radiance-vllm-mxfp4 first, then
  hipfire, then vLLM. A difference from the reference is a defect
  to root-cause, not a property of the model.
