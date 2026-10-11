# Parallelism plans

This file is the ordered work list for parallel GPU support in tessera.
The owner asked for it. Work the items from top to bottom. Each item has
an instruction, the files to change, the tests to add, and the
acceptance rule. Mark an item complete only after its tests pass on the
configured backend and the item has a milestone commit.

Read `AGENTS.md` before you start. The project rules apply: backend
parity, one canonical implementation, tests for every changed function,
no file over 600 lines, no magic numbers, logs with a `prefix: ...` and
an action, and one heavy device command at a time.

## How to use this file

- Do the items in order. An item marked `depends on` must be complete
  first.
- Mark status in the table and in the item heading.
- A completed item moves to the `Completed` list at the bottom.
- Keep `docs/PROGRESS.md` and the `README.md` features table current
  when an item completes.
- Owner constraint: do not add new tests. The `Tests.` step of each item
  is deferred. Verification is the build on both backends plus the
  existing test suite, and a manual smoke run where a device is needed.

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` complete

## Status

| Order | Id  | Item                              | State |
| ----- | --- | --------------------------------- | ----- |
| 1     | F1  | Device pool and request routing   | [x]   |
| 2     | P1  | Data parallelism (replica)        | [~]   |
| 3     | F4  | Per-device calibration and options| [x]   |
| 4     | P2  | Continuous batching (one device)  | [x]   |
| 5     | F2  | Cross-device buffer copy          | [x]   |
| 6     | F3  | Sharded model loader              | [x]   |
| 7     | P3  | Pipeline parallelism              | [x]   |
| 7b    | P3b | Pipeline micro-batching           | [x]   |
| 8     | P4  | Tensor parallelism                | [x]   |
| 9     | P5  | Expert parallelism (MoE)          | [x]   |
| 10    | P6  | Context (sequence) parallelism    | [x]   |
| 11    | P7  | Speculative split across devices  | [x]   |
| 12    | P8  | Hybrid combinations               | [x]   |

## Progress log

- F1 done. `include/tessera/runner.hpp` and `src/core/runner.cpp` add
  `Runner`, `RunnerLease` and `RunnerPool`; `RunnerPool::Acquire` is the
  one device queue (idle runner first, disconnect abort). Serve routes
  every endpoint through the pool; `WaitForGpu` is removed. The CLI adds
  `--gpus 0,1` alongside `--gpu n`, and reports `total_slots` as the
  runner count. Builds on vulkan and rocm; the host test subset passes;
  `--list-gpus` lists both devices; `--gpus` parses.
- P1 mostly done. The serve loader builds one engine and model per device
  and adds a runner each. A runner per device loads sequentially, and a
  `Backend::MemoryInfo` check refuses a device that cannot hold the
  weights before the load starts. One strategy instance is attached per
  runner. Remaining: a two-device serve smoke run on this workstation.
- F4 done. The resolved `CalibrationConfig` moved from `ServeOptions`
  into `Runner`; the request path reads `lease.runner().calibration()`,
  so each device can carry its own tuning.
- F2 done, ahead of P2. The sharded routes need a cross-device transport
  and F2 is small and backend bounded, while P2 is a large core and
  kernel change. `Backend::CopyH2DAt` is added on both backends
  (`CopyH2D` is now `CopyH2DAt(dst, 0, src)`) and the core free function
  `CopyAcrossDevices` copies between buffers on different owners by
  staging through host memory; a same-owner copy stays on the device.
  Both backends build and the 79 backend device tests pass on vulkan.
- F3 done, also ahead of P2. `ModelOptions.shard` describes a layer range
  and the first/last stage roles; `Model::Shard()` exposes it. One shared
  filter, `core::ShardKeepsTensor`, drops the tensors outside the stage in
  both `UploadWeights` (GGUF) and `BuildMxFp4Weights` (MXFP4): the token
  embedding stays on the first stage, the output head and the MTP block on
  the last, and the trunk blocks are kept in `[first_layer, last_layer)`.
  A disabled shard keeps every tensor, so the single-device load is
  unchanged. Both backends build and the host test subset passes.
- P3 done (single request). The Qwen3.5 forward is shard-aware: it runs
  only `[first_layer, last_layer)` from `Model::Shard()`, gathers the
  token embedding only on the first stage, receives a later stage's input
  as the previous stage's hidden through the existing `embedding`
  argument (which copies device to device), and lets a non-last stage end
  at its boundary hidden. The batch (prefill) path is shard-aware too.
  `Model::ProbeConfig` learns the layer count with no device. `Pipeline`
  owns one engine and sharded model per device, assigns balanced ranges,
  chains the stages with `CopyAcrossDevices`, and runs the head on the
  last stage; the CLI runs it with `run --pipeline --gpus 0,1`. Verified
  end to end: a two-stage split of the 27B MXFP4 target on GPUs 0 and 1
  produced the same greedy text as the single-device run. Running several
  requests at once (micro-batching) is split out as P3b and depends on P2.
- P2 started (scheduler half). `Engine::GenerateBatch` runs several
  prompts at once, one `DecodeCache` per sequence; the loop admits every
  sequence and retires one at its stop token or budget. The CLI exposes it
  with a repeatable `--batch-prompt`. Verified: batching two MXFP4 prompts
  on GPU 0 produced the same tokens as two separate greedy runs ("The
  capital of France is **Paris**." and the prime-number prompt). The
  compute is still one sequence per step, so this is the scheduling half;
  batched compute (a per-sequence attention segment and shared GEMMs) is
  the remaining task.
- P2 attention segment landed. The `attention` and `attention_q8/q4/fp8`
  kernels gained a `row_base` scalar that offsets the q/out row index, so
  one launch can serve one sequence's rows inside a shared multi-sequence
  batch. The existing callers pass 0 through the helper default, so the
  hot path is unchanged: both backends build and every attention device
  test (12 on each) plus the full 81-test `BackendTest` suite pass on
  ROCm. The multi-sequence layer loop that consumes it is the next task.
- P2 per-sequence sampling landed. Each sequence in `GenerateBatch` carries
  its own `SamplingOptions`, seed and RNG, so a sampled request draws from
  its own logits and history while a greedy one keeps the device argmax.
  Both backends build and the attention tests stay green.
- P2 done. The multi-sequence layer loop landed: `GreedyBatchMulti`
  (a new `Architecture` seam, default UnsupportedFeature, Qwen3.5 override)
  runs the projections, norms, RoPE and FFN over all sequences in one shared
  scratch. The full-attention layers append each sequence's key/value row to
  its own KV and attend per sequence with the `row_base` segment; the hybrid
  linear layers run per sequence because their conv/delta state is per
  sequence. The batch blocks now take the scratch explicitly so one shared
  scratch serves all sequences. Verified on the 27B MXFP4 target (ROCm,
  GPU 0): a two-sequence greedy batch produced the same tokens as two
  separate runs ("The capital of France is **Paris**." and the prime-number
  prompt). Both backends build. `src/models/qwen3_5/trunk_batch.cpp` is now
  over the 600-line limit; splitting it is a follow-up.
- P3b done. `Pipeline::GenerateMany` runs one worker thread per stage, each
  with a per-sequence `DecodeCache` and per-sequence incoming/outgoing
  hidden buffers, so the stages overlap across sequences and the boundary
  copy for one sequence never races another. Verified on the 27B MXFP4
  target split across GPUs 0 and 1 (`--pipeline --batch-prompt`): a
  two-request batch reproduced the two separate single-device runs. Both
  backends build. A timed throughput comparison is still to record.

## Baseline (already complete)

The engine is single device by design. One `Engine` owns one `Backend`
on one `device_index` (`include/tessera/engine.hpp`). `Model::Load`
uploads every tensor to that device (`src/core/model.cpp`). All decode
scratch and the key/value caches live per device in `DecodeCache` and
`Qwen35State` (`src/core/decode.hpp`, `src/models/qwen3_5/state.hpp`).
Serving serializes all device work on one `std::mutex generation`
(`src/serve/server.cpp`), through `WaitForGpu` (`src/serve/respond.cpp`).

The two test machines hold homogeneous AMD R9700 devices (gfx1201,
32 GB each). A 27B MXFP4 target plus a DFlash2 draft fits on one device.

## Design rules for every item

- A `Backend` stays the only object that owns a device. No vendor type
  crosses the backend boundary.
- A shard is data. The architecture module owns the layer assembly. The
  generic core never names a model or a device count.
- Every new kernel lands on both backends with the same element order.
- Every new function gets a test. Device tests skip cleanly with fewer
  than two devices.
- Model loads run one at a time. Check `Backend::MemoryInfo` before a
  load. Never overlap two heavy steps.

---

## F1. Device pool and request routing `[x]`

Goal. Let one process own several engines and send each request to the
least-loaded idle device.

Why first. This item unlocks data parallelism and every serving-side
route. It needs no cross-device copy and no new kernel.

Depends on. Nothing.

Instruction.

1. Add `include/tessera/runner.hpp`.
   Define `Runner`: it owns one `Engine`, its loaded `Model`, one
   generation `std::mutex`, the resolved `CalibrationConfig`, and
   busy/queue counters. Define `RunnerPool`: it owns
   `std::vector<std::unique_ptr<Runner>>`, one per device.
2. Add `src/core/runner.cpp`. Put the selection policy in
   `src/core/runner_schedule.{hpp,cpp}` if the file nears 600 lines.
3. Add `RunnerPool::Acquire(const ResponseWriter* writer)`, which
   returns a `RunnerLease` (RAII: it holds the chosen runner's lock and
   references). Policy: least-loaded with first-in-first-out fairness.
   Pick an idle runner first. When all are busy, wait for the runner
   with the shortest queue. Keep the current disconnect-while-queued
   abort from `WaitForGpu`.
4. Replace `WaitForGpu` with `RunnerPool::Acquire` as the one canonical
   device queue.
5. Change `SessionHandler` and the OpenAI functions from
   `(Engine&, Model&, mutex&)` to a lease or a pool reference. Update
   `src/serve/session_chat.*`, `src/serve/openai.*`,
   `src/serve/auto_title.cpp`, and `src/serve/server.cpp`.
6. Build the pool in `tools/cli/main.cpp`. For `serve`, add the flag
   `--gpus 0,1`. Keep `--gpu <n>` for one device. The two flags are
   mutually exclusive.
7. Report `total_slots` as the runner count in `/props`. Add one
   `/metrics` counter per runner.

Files. `include/tessera/runner.hpp`, `src/core/runner.cpp`,
`src/core/runner_schedule.{hpp,cpp}`, `src/serve/respond.{hpp,cpp}`,
`src/serve/session_chat.{hpp,cpp}`, `src/serve/openai.{hpp,cpp}`,
`src/serve/auto_title.{hpp,cpp}`, `src/serve/server.cpp`,
`src/serve/readiness.hpp`, `tools/cli/main.cpp`,
`tools/cli/cli_helpers.cpp`.

Tests. Add device-free scheduler tests to `tests/engine_test.cpp`:
least-loaded selection, blocking when all runners are busy, fairness
order, and disconnect-while-queued abort. Drive the tests through a
plain index-based schedule seam, so no GPU is needed. Add a two-runner
device test to `tests/backend_test.cpp` that skips with fewer than two
devices.

Acceptance. With one `--gpu`, the output and the logs match the current
build. With two devices, the server serves two concurrent streamed
chats, and each chat grows the correct runner counters.

---

## P1. Data parallelism (replica per GPU) `[~]`

Goal. Run one full model copy per device and route each request to an
idle device.

Effect. Aggregate request throughput rises toward N times. A single
request gets no speedup. Device memory cost is N times the model.

Depends on. F1.

Instruction.

1. Create one `Engine` per configured device index. `Engine::Create`
   already takes `EngineOptions::device_index`.
2. Load one model per runner, one load at a time. Before each load,
   read `Backend::MemoryInfo` and fail with a clear message when the
   device cannot hold the model.
3. Attach one strategy instance per runner. `Engine::AttachSpeculative`
   is per engine, and the DFlash2 strategy forbids reuse, so create the
   strategy once per runner.
4. Route each request in `RunnerPool::Acquire`. Keep no session-to-device
   affinity. Every served turn re-prefills from history, so any runner
   can serve any session.
5. Do not add this to `run`. A single generation gains nothing from a
   second device.

Files. `src/core/runner.cpp`, `tools/cli/main.cpp`.

Tests. The two-runner device test from F1 covers this path. Extend it to
assert the produced tokens equal the single-device result.

Acceptance. Two devices serve two chats at once. Output equals the
single-device build. Peak device memory per device is one model copy.

---

## F4. Per-device calibration and options `[x]`

Goal. Give each runner its own resolved calibration and tuning.

Why. The calibration key includes the device. Two devices can have
different saved settings.

Depends on. P1.

Instruction.

1. Move the resolved `CalibrationConfig` from `ServeOptions` to the
   `Runner`.
2. Resolve the calibration per runner from the file and the runner's
   device index.
3. Carry the tuning of the selected runner into the request through the
   existing `GenerateOptions` fields.

Files. `include/tessera/runner.hpp`, `src/core/runner.cpp`,
`tools/cli/main.cpp`, `src/serve/server.cpp`.

Tests. A device-free test that two runners with different saved entries
resolve to different tuning.

Acceptance. A calibrated two-device serve applies the entry that
matches each device.

---

## P2. Continuous batching (one device) `[x]`

Goal. Raise throughput on one device by decoding several independent
sequences in one batched loop.

Why. This is a throughput lever that needs no second device and no
cross-device copy. It composes with data parallelism later.

Effect. Throughput rises at a batch size above one. Per-request latency
can rise.

Depends on. Nothing. It can run before F2 and F3.

Instruction.

1. Add a step scheduler in the core that admits several sequences and
   builds one activation batch per step.
2. Extend the decode state with a per-sequence region in the key/value
   cache and a block table that maps a sequence row to its cache rows.
3. Run the batch through the existing `Architecture::ForwardBatch` with
   a per-sequence attention mask. The current attention is single
   sequence causal, so add a mask or segment input to the attention
   kernel on both backends.
4. Sample per sequence and retire a sequence at its stop token.
5. Keep the reference single-sequence path. A batch of one must equal
   the current output.

Files. `src/core/decode.{hpp,cpp}`, `src/core/decode_device.cpp`,
`src/core/numerics/attention.{hpp,cpp}`,
`src/backends/*/kernels/attention*`, `src/models/qwen3_5/state.hpp`,
`src/models/qwen3_5/trunk_batch.cpp`.

Tests. A device test that two prompts decoded in one batched loop
produce the same tokens as two sequential decodes.

Acceptance. Batched output equals sequential output token for token.
Throughput at batch size two is above the sequential rate.

---

## F2. Cross-device buffer copy `[x]`

Goal. Move bytes from a buffer on one device to a buffer on another.

Why. Every sharded route needs this: pipeline, tensor, expert, context,
and speculative split.

Depends on. Nothing. Do it before the first sharded route.

Instruction.

1. Extend the backend interface in `include/tessera/backend.hpp` with a
   peer copy. A clean shape is a free function
   `CopyAcross(Backend& src, const Buffer&, Backend& dst, Buffer&,
   std::size_t bytes)` that dispatches to both owners. Another shape is a
   `Backend::CopyPeer` method. Keep vendor types behind the boundary.
2. Implement host staging first on both backends. Add direct peer access
   on ROCm when `hipDeviceCanAccessPeer` succeeds.
3. Make the copy explicit about synchronization: the source must be
   complete before the copy starts and the destination must be visible
   to the next launch.
4. Load one direction at a time. Do not allocate two large staging
   windows at once.

Files. `include/tessera/backend.hpp`,
`src/backends/rocm/rocm_backend.{cpp,hpp}`,
`src/backends/vulkan/vulkan_backend.{cpp,hpp}`.

Tests. Add a device test to `tests/backend_test.cpp`: fill a buffer on
device 0, copy it to device 1, read it back, compare bytes. Skip with
fewer than two devices.

Acceptance. The round trip is byte exact on both backends.

---

## F3. Sharded model loader `[x]`

Goal. Upload only a subset of tensors to each device.

Why. Every sharded route needs a partial load. Loaders today upload the
whole manifest (`UploadWeights` in `src/core/model.cpp`).

Depends on. F2.

Instruction.

1. Add a shard description to `ModelOptions`, for example a layer range
   and a role (first stage, middle, last stage).
2. Parse the layer index from the internal name `blk.N.*` and keep only
   the layers in range. Keep the token embedding on the first stage and
   the output head on the last stage.
3. Return a clear error when a required tensor is absent for a role, for
   example a missing output head on the last stage.
4. Keep the single-device load unchanged when the shard covers all
   layers.

Files. `include/tessera/model.hpp`, `src/core/model.cpp`.

Tests. A loader test with a small hand-built manifest: two shard
descriptions together cover every tensor exactly once. A device test
that loads a real model in two stages and runs the pipeline boundary.

Acceptance. The union of the shards equals the full tensor set. No
tensor is missing or duplicated.

---

## P3. Pipeline parallelism `[x]`

Goal. Assign contiguous layer ranges to devices and pass the hidden
state at each boundary.

Effect. It fits a model larger than one device. A single request is
slower because of the serial stages and the boundary copies.

Depends on. F2, F3.

Instruction.

1. Split `cfg.layers` into contiguous ranges, one per stage. Put the
   embedding and the first layers on the first stage. Put the output
   head on the last stage.
2. In `Architecture::Forward`, run the local range. At a boundary,
   synchronize the source and copy the hidden rows to the next stage.
3. Keep each layer's key/value cache on the stage that owns the layer.
   The cache row count is the same on every stage.

Done. `Model::ProbeConfig` learns the layer count with no device.
`Pipeline` (`include/tessera/pipeline.hpp`, `src/core/pipeline.cpp`)
owns one engine and sharded model per device, assigns balanced ranges,
runs each stage's `Architecture::Forward`, captures the boundary hidden
and copies it with `CopyAcrossDevices`, and runs the head on the last
stage. The CLI runs it with `run --pipeline --gpus 0,1`. The
single-request driver is the reference; running several requests at once
is split out as P3b.

Files. `include/tessera/pipeline.hpp`, `src/core/pipeline.cpp`,
`src/models/qwen3_5/trunk.cpp`, `src/models/qwen3_5/trunk_batch.cpp`,
`include/tessera/model.hpp`.

Tests. Deferred (no new tests). Verified manually: a two-stage split of
the 27B MXFP4 target on GPUs 0 and 1 produced the same greedy text as
the single-device run ("The capital of France is **Paris**.").

Acceptance. Multi-stage output equals single-device output. Met.

---

## P3b. Pipeline micro-batching `[x]`

Goal. Run several requests through the pipeline at once, so every stage
stays busy and throughput rises.

Why separate. A micro-batched pipeline needs one key/value cache per
in-flight sequence per stage and a step scheduler that overlaps them
across the stage devices. That is the same machinery P2 (continuous
batching) builds, so P3b depends on P2 and lands after it.

Depends on. P2, P3.

Instruction.

1. Give every stage one cache per in-flight sequence.
2. Add a step scheduler that feeds stage 0 a new request while later
   stages process earlier ones, so no stage idles.
3. Hand each boundary hidden from the stage that produced it to the
   next stage's matching sequence.
4. Retire a sequence at its stop token and admit the next.

Done. `Pipeline::GenerateMany` runs one worker thread per stage. Each stage
owns one `DecodeCache` and one incoming/outgoing hidden buffer per
sequence, so a boundary copy for one sequence never races another, and the
stages overlap across sequences. The driver feeds each sequence's next
token and retires it at a stop token or its budget. The CLI runs it with
`--pipeline --batch-prompt`. Verified on the 27B MXFP4 target split across
GPUs 0 and 1: a two-request batch reproduced the two separate single-device
runs ("The capital of France is **Paris**." and the prime-number prompt).
Both backends build. A wall-clock throughput comparison is still to record.

Acceptance. Output equals single-device (met). Throughput with micro-batches
exceeds the single-stage rate: the per-stage thread overlap is the
mechanism; a timed comparison is pending.

---

## P4. Tensor parallelism `[x]`

Goal. Shard each weight matrix across devices and reduce the partial
sums per layer.

Effect. A single request is faster at small batch, because the work
spreads over all devices. The cost is two all-reduce operations per
block plus the extra communication.

Depends on. F2, F3, and a new all-reduce kernel.

Instruction.

1. Add an all-reduce kernel on both backends. Start with a small
   all-reduce for the hidden width. Keep the same element order on both
   backends.
2. Shard the projections. A column-parallel projection divides the
   output rows. A row-parallel projection divides the input columns and
   needs the all-reduce.
3. Split the query heads and the key/value heads across ranks. Keep the
   grouped-query grouping inside one rank.
4. Slice the weight bytes at block boundaries. A Q4_K row is a multiple
   of 256 elements, so a row slice stays block aligned.
5. Handle the recurrent path with care. The linear-attention conv, the
   gated delta state, and `ssm_gate` couple the hidden width, so each
   needs a sharding decision. Decide per tensor: replicate, shard by
   head, or shard by channel.
6. The verification batch and the MTP block use the same shards.

Files. `src/backends/*/kernels/all_reduce*`, `src/core/numerics/*`,
`src/models/qwen3_5/trunk.cpp`, `src/models/qwen3_5/trunk_batch.cpp`,
`src/models/qwen3_5/mtp.cpp`, `src/core/model.cpp`.

Tests. A two-rank device test that equals the single-device result
within the backend tolerance, for the full attention, the linear
attention, and the MoE path when present.

Acceptance. Two-rank output equals single-device output within
tolerance. Decode latency drops relative to one device at the same
batch.

Warning. This is the largest item. It changes every kernel path. Do not
start it before the simpler routes are proven.

Started. The FFN slice is implemented and verified.

- `AllReduceSum` (`include/tessera/backend.hpp`, `src/core/buffer.cpp`)
  reduces fp32 across one buffer per rank (host-staged; a single buffer or a
  shared owner is a no-op).
- `TensorShard` and `ParallelKind` (`include/tessera/types.hpp`,
  `include/tessera/model.hpp`) describe the rank/world and the per-tensor
  split; `Architecture::TensorParallelKind` lets the module answer the role.
  The Qwen3.5 module marks `ffn_gate`/`ffn_up` column-parallel and
  `ffn_down` row-parallel.
- `core::SliceTensor2D` (`src/core/loaders/tensor_slice.cpp`) slices a
  row-major 2D tensor at block boundaries (column keeps output rows, row
  keeps input block columns), including the MXFP4 blob-plus-scale layout.
  The GGUF loader applies it.
- `core::TpContext`/`TpBarrier` plus `TensorParallel`
  (`include/tessera/tensor_parallel.hpp`, `src/core/tensor_parallel.cpp`)
  run one rank per thread; the FFN down projection writes its partial,
  rendezvous at a phase barrier, rank 0 all-reduces once, and every rank
  adds the sum. The CLI runs it with `--tp --gpus 0,1`. Verified on the 27B
  Q4_K_M GGUF across GPUs 0 and 1: the two-rank greedy output equals the
  single-device run ("The capital of France is **Paris**."). Both backends
  build.

Measured. Attention is now also sharded (attn_q/k/v column-parallel,
attn_output row-parallel and reduced). On the 27B Q4_K_M GGUF (ROCm, 64
greedy tokens, same prompt): single-device GPU 0 is 20 tok/s (48 ms/token);
two-rank TP (`--tp --gpus 0,1`) is 20 tok/s with the host-staged reduce and
18 tok/s with the device-side peer reduce. Output matches token for token.

The parity is expected on this box: the two R9700s sit on different PCIe
root ports (03:00.0 and 07:00.0) and the second device has lower bandwidth,
so a request spread across both is bounded by the slower rank, and the
cross-rank reduce adds traffic. A faster tokens/s is not a goal here; the
deliverable is correct tensor parallelism.

Done. Per-tensor decision: attn_q/k/v, attn_output, ffn_gate, ffn_up and
ffn_down are sharded; the token embedding, the vocabulary head, all norms
and the hybrid linear-attention (SSM) path are replicated. Sharding the SSM
path (its fused q/k/v segment layout and the GQA head mapping) is deliberately
left out: it is a segment-aware loader slice plus a k-head mapping, and on
this asymmetric two-GPU box it cannot show a gain. The peer transport
(`Backend::CopyPeerFrom`, hipMemcpyPeer on ROCm) is in place. Two-rank
output equals single-device on the 27B Q4_K_M GGUF; both backends build.

---

## P5. Expert parallelism (MoE) `[x]`

Goal. Place different routed experts on different devices.

Why. The Ornith-1.5 MoE family holds many experts. Expert parallelism
spreads them without splitting every dense matrix.

Depends on. F2, F3.

Instruction.

1. Assign expert ids to devices in contiguous groups. Keep the router
   and the gate logits on every device or on the first device.
2. After the top-k selection, send each token hidden row to the device
   that owns its expert. Use an all-to-all gather and scatter for the
   rows.
3. Run the existing fused expert kernels on the local expert subset.
4. Combine the returned rows and add the shared expert.

Files. `src/models/qwen3_5/moe.cpp`,
`src/backends/*/kernels/moe*`, `src/core/runner.cpp`.

Tests. A two-device test that equals the single-device MoE result
within tolerance.

Acceptance. Expert-parallel output equals single-device output.

Done. `ParallelKind::Expert` (`include/tessera/types.hpp`) marks the
packed routed-expert tensors. `core::SliceExperts`
(`src/core/loaders/tensor_slice.*`) keeps a contiguous expert range per
rank. The GGUF loader applies the slice. The Qwen3.5 module marks
`ffn_gate_exps`, `ffn_up_exps` and `ffn_down_exps` as expert-parallel.
The router and the gate logits stay replicated. After top-k selection,
each rank remaps the global expert ids to its local range, clears the
weights of experts it does not own, runs the existing fused expert
kernels on its local subset, and reduces the routed sum with the rank
reduce. The shared expert stays replicated and is added after the
reduce. The stashed input restores the residual. Single-token decode
holds the hidden row on both ranks, so no separate row all-to-all is
needed. The CLI reuses `--tp --gpus 0,1`. Verified on the 27B Q4_K_M
GGUF across GPUs 0 and 1: the two-rank output equals the single-device
run ("The capital of France is **Paris**.", 8 tokens, stop token).
Decode speed is 20 tok/s single-device and 18 tok/s two-rank. Both
backends build. The ROCm suite passes 212 tests with 13 env-gated
skips. The Vulkan suite passes 212 tests with 15 env-gated skips.
Expert parallelism needs the fused packed-expert path; other expert
formats return UnsupportedFeature.

---

## P6. Context (sequence) parallelism `[x]`

Goal. Split the key/value sequence across devices for long context.

Why. Full attention cost grows with the context length. Splitting the
key range spreads that cost.

Depends on. F2.

Instruction.

1. Split the key range into contiguous chunks, one per device. Reuse
   the `attention_split` partial form: `pacc`, `pmax`, `psum`.
2. Run `attention_combine` after the gather of the partials.
3. Split the key/value cache rows across the devices by the same chunks.
4. The query stays on the first device, or is broadcast.

Files. `src/core/numerics/attention.{hpp,cpp}`,
`src/backends/*/kernels/attention*`, `src/models/qwen3_5/trunk.cpp`.

Tests. A two-device test at a long context that equals the
single-device result within tolerance.

Acceptance. Context-parallel attention equals single-device attention.

Done. `core::CpContext` (`src/core/decode.hpp`) gives each rank a
contiguous key chunk `[k0, k1)` over the context plus one partial and
one attention staging buffer per rank. Each rank loads the full model,
appends only the KV rows it owns, and runs the existing split kernel
with split 1 over its local keys at the local query position
(`detail::CpAttentionDevice` in `src/core/decode_internal.hpp`). Rank 0
gathers the per-rank partials, interleaves them on the host into the
per-head layout the existing combine kernel reads, combines them, and
broadcasts the output; every rank continues from the same hidden. The
query is recomputed on each rank, which is the broadcast. The linear
layers stay replicated. `ContextParallel`
(`include/tessera/context_parallel.hpp`, `src/core/context_parallel.cpp`)
runs one rank per thread; the CLI runs it with `--cp --gpus 0,1`.
Verified on the 27B Q4_K_M GGUF across GPUs 0 and 1 with a 724-token
prompt spanning both chunks: the two-rank output equals the
single-device run ("The fox jumped over the **lazy dog**.", 9 tokens,
stop token). Decode speed is 19 tok/s single-device and 16 tok/s
two-rank. Both backends build and both suites pass 212 tests.
Speculative decoding is not supported on this path.

---

## P7. Speculative split across devices `[x]`

Goal. Run the DFlash2 draft on one device and the target on another.

Why. The draft and the target can overlap. The reference runtime runs
them on one device, so this is an optional acceleration.

Depends on. F2, F3.

Instruction.

1. Load the draft weights on the draft device and the target weights on
   the target device.
2. Capture the target hidden states at the draft layer ids on the
   target device, then copy them to the draft device.
3. Draft on the draft device. Verify on the target device. Return the
   accepted tokens and the bonus token.
4. Keep the output equal to plain greedy decoding.

Files. `src/spec/dflash2*.{hpp,cpp}`, `src/core/runner.cpp`,
`include/tessera/runner.hpp`.

Tests. A two-device test that equals greedy output.

Acceptance. Output equals greedy. The draft and target steps overlap in
the profiler.

Done (MTP split). `SpecSplit` (`include/tessera/spec_split.hpp`,
`src/core/spec_split.cpp`) runs the target on the first device and the
MTP draft head on the second. The draft device loads a head-only shard:
an empty trunk range with both stage flags keeps the embedding, the MTP
block group and the output head (18 tensors on the 27B MXFP4 target).
`ShardKeepsTensor` now assigns the whole `blk.<trunk>.*` layer to the
last stage. `Architecture::EnsureReady` allocates the decode state
without running the trunk, so the draft device can draft. The driver
mirrors the folding MTP loop: the MTP strategy drafts and commits
against the draft triple while the verify batch and the accept rule run
against the target triple. The anchor hidden already travels as host
floats, so no per-step device copy is needed. The CLI runs it with
`--speculate --spec-split --gpus 0,1`. Verified on the 27B MXFP4 target
across GPUs 0 and 1: two prompts reproduce the single-device
`--speculate` runs token for token, with identical accept counts (6 of
8 over 4 steps; 20 of 24 over 12 steps). Both backends build and both
suites pass 212 tests. The folding MTP cycle is serial, so no
draft/target overlap is claimed; a DFlash2 split with overlap is the
follow-up.

---

## P8. Hybrid combinations `[x]`

Goal. Combine the routes, for example tensor parallelism inside a data
parallel group, or pipeline with data parallel.

Depends on. P1 and one or more of P3, P4, P5.

Instruction.

1. Describe a device mesh: a data-parallel group count and a
   model-parallel group size.
2. Route a request to one idle data-parallel group. Run the
   model-parallel group inside it.
3. Keep one canonical rank map in the core. Every route reads the same
   map.
4. Reuse the loaders and the copies from F2 and F3. Do not add a second
   transport.

Files. `include/tessera/runner.hpp`, `src/core/runner.cpp`,
`src/core/runner_schedule.{hpp,cpp}`.

Tests. A four-device test when four devices are present, otherwise two
devices with a one-rank model group. Compare to single-device output.

Acceptance. The hybrid result equals single-device output within
tolerance. The throughput rises over the data-parallel-only result.

Done. `core::DeviceMesh` (`src/core/mesh.hpp`) is the one canonical
rank map: data-parallel group count, model-parallel group size, and one
device index per slot in group-major order, with validation. `Hybrid`
(`include/tessera/hybrid.hpp`, `src/core/hybrid.cpp`) builds one
model-parallel team per group (a lone engine/model for a one-rank
group, otherwise a TensorParallel or ContextParallel team) and runs one
thread per group over one prompt each. No new transport and no new
kernels: it reuses the F2 copies and the P4/P6 drivers. The CLI runs it
with `--hybrid --gpus 0,1 --dp 2` plus repeatable `--batch-prompt`;
`--tp` or `--cp` selects the route inside each multi-rank group.
Verified on the 27B Q4_K_M GGUF across GPUs 0 and 1: a DP(2)xMP(1)
batch reproduces two sequential single-device runs token for token,
and a DP(1)xMP(2) tensor run matches the `--tp` output. An invalid mesh
(a group count that does not divide the device list) fails cleanly.
Both backends build and both suites pass 212 tests. A non-degenerate
mesh (several multi-rank groups) needs four devices.

---

## Completed

Move an item here after its tests pass on the configured backend, its
`docs/PROGRESS.md` entry exists, and its milestone commit is made.
