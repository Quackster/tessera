# MoE paging performance: CPU cost and next work

This file tracks CPU cost under `--gpu-layers` paging.
It covers Ornith-1.5 first.
It then applies to dense models.
See `AGENTS.md` for the working rules.
See `docs/MOE.md` for the offload design.
See `docs/ORNITH-1.5.md` for the MoE architecture record.

## Goal

Keep all calculation on the GPU.
Use RAM only as a backing store.
Reduce host CPU work per token.
Measure each change in tokens per second and host CPU load.

## Baseline

Date 2026-10-11: paging works end to end on Vulkan and ROCm.
Compute already runs on the GPU.
No layer runs on the CPU.
Dequant runs inside the GPU kernels.

Ornith Q4_K_M GGUF with `--gpu-layers 20`:
369 tensors resident with 11.0 GB on device.
384 tensors paged with 10.7 GB in RAM.

Rates on the R9700 before this work:
Full VRAM is about 88 tok/s.
Whole layer paging is about 0.4 tok/s.
Active expert paging is about 2.5 tok/s.
Batched expert slice upload is about 4 tok/s.

## Round 1: transfer overhead

Date 2026-10-11: identical 14 MB expert uploads varied from 5 to
19 ms, so fixed per-batch cost dominated, not bytes. Each of the 40
batches per token paid about 15 driver calls plus 2 full device
syncs: fresh `vkAllocateMemory`, a new command buffer and fence, a
pipeline drain before the copy, and a fence wait after it. Prefill
with 23 rows re-uploaded experts per row per layer (about 6.4 GB).

Changes, all output identical, both backends in the same change:
Persistent host staging window, grown on demand.
Reusable copy command buffer and fence on Vulkan.
Flush without draining, so one fence wait covers pending compute
plus the copy (same queue order, no lost barrier).
Per-layer paged index in `Model`, so an ensure touches only its
own layer and empty layers skip the round trip.
Staged expert set, so a prefill batch uploads each shared expert
once per layer instead of once per row.
Device embedding gather and device top-1 for the vanilla dense
decode path, mirroring the MoE trunk.
Single buffer download for F32 readbacks.

Rates after round 1, same prompt, greedy, identical output
`Gravity is the natural attraction between any two`:
N=20: 5.4 tok/s decode, prefill unrecorded this round.
N=30: 9.7 tok/s decode, 35 tok/s prefill.
N=38: 26 tok/s decode, 59 tok/s prefill.
N=40: 58 tok/s decode, 93 tok/s prefill.
Effective staging rate rose from about 3 GB/s to about 7 GB/s.
Host CPU time per run roughly halved.

## Round 2: MoE dense stays resident

Date 2026-10-11: per-token traffic was about 900 MB at N=20
(about 570 MB dense plus about 330 MB expert slices). The design in
`docs/MOE.md` already says dense block tensors stay resident only
the routed experts page. The code paged whole early blocks instead.
`PagedMoeDenseResident` in `src/core/loaders/shard.cpp` implements
the documented rule now: in a model with routed experts, every
non-expert block tensor stays resident. Dense models are unchanged.
The fit check, the help sizing, and the load summary count the
refined resident set through the same predicate, so no second
accounting exists.

Rates after round 2, same prompt, greedy, identical output:
N=20 Vulkan: 14 tok/s decode, 25 tok/s prefill.
N=20 ROCm: 14 tok/s decode, 26 tok/s prefill.
N=30 Vulkan: 22 tok/s decode, 37 tok/s prefill.
N=40 Vulkan: 61 tok/s decode, 92 tok/s prefill.
Paged set at N=20 fell from 384 tensors to 63 expert tensors.
Resident grew from 11.0 GB to 11.5 GB.
Full suite passes on Vulkan and ROCm (230 tests, 0 failures each).

## Remaining hotspots

Per-token cost at N=20 is about 70 ms now.
Expert slice uploads are about 50 ms (330 MB at 6 GB/s).
Expert id readbacks are about 20 small sync round trips.
GPU compute is about 12 ms.
The 40 tok/s target needs 25 ms per token, so about 72 MB at
6 GB/s. Expert bytes alone are 330 MB. Only reuse across tokens
closes that gap on this bus.

A 24-token trace measured expert reuse per layer:
Mean consecutive-token overlap is 3 of 8 experts.
A per-layer LRU of 32 experts hits 59 percent.
A per-layer LRU of 64 experts hits 66 percent.
Early layers reuse less than late layers.
At 59 percent reuse the token would cost about 50 ms, so about
20 tok/s. An expert cache is therefore future work with a known
ceiling near 20 tok/s at N=20, not part of this round.

## Todo

- [x] Batch the expert slice uploads into one `CopyH2DBatch` per layer.
Acceptance: paged output is bit identical to full upload.
- [x] Remove per-batch allocation: persistent staging, reusable
command buffer and fence, flush without draining.
Acceptance: effective staging rate about 7 GB/s, same output.
- [x] Index paged tensors per layer and skip empty layers.
Acceptance: resident layers pay no round trip, same output.
- [x] Upload each shared expert once per prefill layer.
Acceptance: prefill 23 rows at N=20 runs 25 tok/s, same output.
- [x] Move dense decode embedding and argmax to the GPU.
Mirror `GatherEmbeddingDevice` and `DeviceRowArgMax`.
Acceptance: dense paged output is unchanged.
- [x] Keep MoE dense resident, page only routed experts.
Acceptance: N=20 runs 14 tok/s on Vulkan and ROCm, same output.
- [ ] Expert reuse cache across tokens.
Measured LRU-32 hits 59 percent, worth about 20 tok/s at N=20.
Needs cache-aware slice math plus tests on both backends.
- [ ] Keep prefetch deferred while transfers stay bandwidth bound.
Overlap would hide at most layer compute at about 0.3 ms.
This file records the profile date and the reason.

## Measurement

Record each run with the same fields.
Model file and revision.
Backend and device.
Resident layer count and byte counts.
Prompt and token count.
Tokens per second.
Host CPU load during decode.
Correctness check against full upload.

N curve on the R9700 after round 2 (Ornith Q4_K_M GGUF, greedy,
output `Gravity is the natural attraction between any two`):

| N  | decode tok/s | prefill tok/s | backend |
| -- | ------------ | ------------- | ------- |
| 20 | 14           | 25            | Vulkan  |
| 20 | 14           | 26            | ROCm    |
| 30 | 22           | 37            | Vulkan  |
| 40 | 61           | 92            | Vulkan  |

Example commands:

```sh
cmake -B cmake-build-vulkan -DTESSERA_BACKEND=vulkan
cmake --build cmake-build-vulkan -j
ctest --test-dir cmake-build-vulkan --output-on-failure
```

```sh
./cmake-build-vulkan/tessera-cli run \
  --model ~/models/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  --gpu-layers 20 \
  --prompt-text "Explain gravity in one sentence." \
  --max-completion-tokens 8
```

Protect the machine on every run.
Run one heavy command at a time.
Wait for one model load to finish before the next starts.
Cap each command timeout at 5 minutes.
Stream the output and watch the whole stream.
Stop at the first sign of garbage and fix the root cause.

## Update rule

Keep this file current after each milestone.
Record what was completed with the date.
Refresh the Todo list so it always reflects reality.
Add new measurements under Measurement.
Update `docs/MOE.md` status in the same change when design facts change.
Update the `README.md` features table when user visible behavior changes.
Do not pile up finished milestones without a commit.
