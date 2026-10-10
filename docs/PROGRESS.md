# Progress

This file tracks tessera development. After each milestone, update
"Current status" and "Next" so both match reality. See AGENTS.md.

Latest suite: 375/375 `ctest` on vulkan. 375/375 `ctest` on ROCm.
Both builds verified on AMD Radeon AI PRO R9700 (vulkan through
RADV GFX1201, rocm through the system ROCm).

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
- Kernel launch plumbing on both backends. `LaunchKernel` binds
  buffers and 64-bit scalars in parameter order. Vulkan pipelines
  through a four-slot launch ring. ROCm copies use async
  device-to-device transfer.
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
  Completion and thinking token budgets are enforced.
  The server binds before the model finishes loading: `/health`
  answers 503 with a JSON `status`/`detail` body until the model is
  loaded and its kernels warmed, then 200. The web UI polls it and
  keeps the composer disabled until ready. Unimplemented surfaces
  return 501.
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
  Greedy 23 tok/s. DFlash2 60 tok/s. Load 54 s.
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

## Next (in order)

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
- Runtime options. Still to wire: batch caps for features that do
  not exist yet. No hard-coded paths or sizes.
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
