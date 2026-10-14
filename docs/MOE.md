# MoE weight offload: RAM backing with VRAM resident layers

This file tracks the work to add a `--gpu-layers` option to tessera.
The option keeps all weights in system RAM.
It keeps a selected layer count in VRAM.
It targets Ornith-1.5 and other MoE models first.
It then applies to dense models.
See `AGENTS.md` for the working rules.
See `docs/ORNITH-1.5.md` for the MoE architecture record.

## Goal

Add a CLI option `--gpu-layers N`.
The engine loads all weights into system RAM.
The engine keeps N layers in VRAM.
The engine pages the remaining layers from RAM per token.
Compute stays on the GPU.
No layer runs on the CPU.
## Background

Tessera loads every tensor to VRAM today.
`Model::Load` parses the file.
`UploadWeights` allocates one device buffer per tensor.
`CopyH2DBatch` copies all bytes to the device.
The file mapping closes after load.
No host copy remains.

Ornith-1.5-35B-A3B is 35B parameters total.
It activates about 3B parameters per token.
The Q4_K_M GGUF file is about 21 GB.
It fits on a 24 GB card today.
It leaves little room for KV cache and context growth.
A larger context or a smaller card needs partial offload.

## Reference research

llama.cpp solves this with `-ngl` (`--n-gpu-layers`).
N is the maximum layer count kept in VRAM.
The rest stays in system RAM.
The loader reports `offloaded N/M layers to GPU`.
The embedding table stays on the CPU.
The output layer goes to the GPU first.
Then blocks go from the last one backward.
The KV cache follows its layer.
`--no-kv-offload` keeps all KV cache in RAM.
`--n-cpu-moe N` keeps expert weights in RAM while attention stays on GPU.
Each overridden layer adds two graph seams.
Decode is memory bound.
Per token speed follows bytes read per token divided by bandwidth.

Tessera differs in one point.
Tessera has no CPU compute backend.
Only Vulkan and ROCm exist.
So RAM is a backing store.
The GPU pages weights from it.
No operation runs on the CPU.

## Decision

Use llama style last-N resident placement.
The output layer plus the last N minus 1 blocks stay in VRAM.
Blocks before them page from RAM per token.
The embedding, router, norms, and shared expert weights stay resident.
They are small and every token uses them.

A sliding window around the current layer was considered.
It moves the same bytes per token.
It adds evict and upload churn on every layer.
It has no reuse benefit because decode visits each layer once per token.
Last-N resident is simpler and faster.
It also matches user knowledge from llama.cpp.

MoE work splits into two phases.
Phase 1 pages whole layers.
It works with the current fused expert kernels as is.
Phase 2 pages only active experts.
It moves about `top_k / num_experts` of the expert bytes.
For Ornith that is 8 of 256.
It needs no new kernels: slices land at their expert offsets in the
existing staging slots, so both MoE kernels keep the unchanged ids.
Phase 1 comes first.

## Agreed scope

- Flag spelling: `--gpu-layers N`.
- Scope: `run` and `serve`.
- Formats: GGUF and MXFP4 from day one.
- Default `0` keeps current full VRAM behavior. No regression.
- Explicit N enables paging. `all` and `999` mean full VRAM.
- Both backends implement the same contract in the same change.

## Design

- Keep the file mapping alive in `Model` for the run lifetime.
- For GGUF this is the mapped GGUF file.
- For MXFP4 the converted weights upload fully and the paged set is
  downloaded back to retained host copies, so the conversion path is
  untouched. This holds the full model on the device transiently.
- Allocate device buffers only for the resident set.
- Allocate one reusable staging window for paged layers.
- Before layer L runs, upload its tensors from RAM to the window.
- Prefetch layer L plus 1 during compute of layer L where possible.
- Resident layers pay zero PCIe bytes per token.
- Paged layers pay their byte size per token.
- Log resident bytes, RAM bytes, and the offloaded layer count at load.
- Refuse to start when the resident set plus KV cache does not fit.

## CLI spec

| Flag | Scope | Default | Meaning |
| --- | --- | --- | --- |
| `--gpu-layers <n\|all>` | run, serve | 0 | Layers kept in VRAM. 0 keeps current full upload. `all` uploads all. |

Examples:

```sh
# Ornith GGUF with 20 resident layers.
./cmake-build-vulkan/tessera-cli run \
  --model ~/models/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  --gpu-layers 20 \
  --prompt-text "Explain gravity in one sentence." \
  --max-completion-tokens 64
```

`--gpu-layers help` sizes the model without loading it. It prints
the offloadable layer count, the KV cache cost at the current
`--context` and KV cache type, and the largest fitting layer count
on the selected GPU:

```sh
./cmake-build-vulkan/tessera-cli run \
  --model ~/models/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  --gpu-layers help --kv-q8 --context 262144
# 40 trunk blocks plus the output head (41 offloadable layers, ...)
# context 262144, kv int8: cache about 2642 MiB (10 full, 30 linear)
# weights: full upload 20697 MiB
# best --gpu-layers 0: ...
```

The weights are exact per tensor. The cache follows the loader's
growth. A 1024 MiB reserve covers the driver, the scratch and the
draft. ROCm sizes against current free memory. Vulkan reports no
free figure, so it sizes against the device total and says so.

## Current status

Date 2026-10-11: paging works end to end on Vulkan and ROCm.
`ModelOptions` carries `gpu_layers`.
`GpuLayerResident` in `src/core/loaders/shard.cpp` selects the resident
set. GGUF keeps the mapped file as RAM backing and uploads only the
resident set. Converted MXFP4 weights upload fully and then evict the
paged set back to RAM, so the conversion path is untouched. Staged
paged layers reach the device per token through
`EnsureLayerResident`, hooked into all four weight loops. The CLI takes
`--gpu-layers N|all` on `run` and `serve`.
`EngineTest.GpuLayerResidentSplitsLastLayers` and
`EngineTest.PagedModelMatchesFullUpload` cover it.
Phase 2 pages only the routed experts: the whole-layer ensure skips
`ffn_*_exps` tensors, each decode token uploads its top-k slices at
their expert offsets, and batched rows above rows times top-k reaches
expert count stage the whole expert set instead. Both MoE kernels keep
the unchanged ids, so paged output is bit-identical to full upload.
Full suite passes on Vulkan and ROCm (221 tests, 0 failures each).

Date 2026-10-11: the transfer path no longer allocates per batch.
Both backends keep a persistent staging window. Vulkan reuses its
copy command buffer and fence and flushes pending compute without
draining it, so one fence wait covers compute plus copy. `Model`
indexes paged tensors per layer and skips empty layers, and selective
expert staging uploads each shared expert once per prefill layer.
The MoE dense refinement from the Decision section above is
implemented now (`PagedMoeDenseResident`): dense block tensors stay
resident and only routed experts page, in dense and MXFP4 loads and
in the fit check alike. Full suite passes on Vulkan and ROCm
(230 tests, 0 failures each).

## What is done

- [x] Record Ornith MoE architecture in `docs/ORNITH-1.5.md`.
- [x] Research llama.cpp offload semantics.
- [x] Agree flag spelling `--gpu-layers N`.
- [x] Agree scope: GGUF plus MXFP4, run plus serve, both backends.
- [x] Agree placement: last-N resident, whole layers in phase 1.
- [x] Write this plan to `docs/MOE.md`.

## Todo

- [x] Add `gpu_layers` to `ModelOptions` with parsing and validation.
- [x] Keep RAM backing alive in `Model` after load.
- [x] Allocate resident set plus staging window in `UploadWeights`.
- [x] Page whole `blk.L.*` tensors per token in the decode path.
- [x] Page converted MXFP4 weights by post-build eviction to RAM.
- [ ] Add prefetch of next layer where the backend allows overlap.
  Assessed 2026-10-11: neither backend allows it today. Vulkan
  flushes and fence-waits every transfer; ROCm synchronizes the
  stream per call. Profiled 2026-10-11 (`TESSERA_PROFILE=2`, Ornith
  `--gpu-layers 20`): each paged layer stages about 16 MB of expert
  slices in about 5 ms and about 27 MB of dense tensors in about
  8 ms, both at the measured bus rate of about 3.1 GiB/s, so the
  transfers are bandwidth-bound, not round-trip-bound. Decode moves
  about 0.9 GB per token at the same rate, which fully accounts for
  the 4 tok/s rate. Overlap could at best hide the smaller side
  (layer compute, about 0.3 ms at full-VRAM rates), so an async
  transfer extension to the `Backend` interface on both backends
  would buy little for a large interface change. Stays deferred.
- [x] Wire `--gpu-layers` through `run` and `serve` plus usage text.
- [x] Extend `LogModelSummary` with resident and RAM byte counts.
- [x] Add VRAM fit check before load starts. `FitGpuLayers` judges
  the requested value (resident plus cache plus the 1024 MiB reserve)
  on `run` and `serve`; an impossible load is refused with the needed
  and fitting sizes instead of dying mid-load. It covers the KV cache
  too, so a small N with a large context is caught up front.
- [x] Add unit tests for layer split and paging order.
- [x] Verify paged runs on Vulkan and ROCm for GGUF and MXFP4.
- [x] Measure decode rate at full VRAM versus paged N on Ornith Q4_K_M.
- [x] Update `README.md` flags table. There is no `docs/PROGRESS.md`
  in the tree, so nothing else tracks this.
- [x] Phase 2: active expert paging for `ffn_*_exps` tensors.
- [x] Batch the expert slice uploads: `HostCopy` carries a
  destination offset and `CopyH2DBatch` honors it on both backends,
  so one token stages a layer's slices in one round trip instead of
  one per expert.

## Qwen3.8-Flash-Next MoE work (and what applies to Ornith)

The `qwen4exp` port (see `docs/QWEN_FLASH_NEXT.md`) added a set of MoE
optimizations. This section records each one and whether it applies to
Ornith (`qwen35moe`).

1. **Warp-per-output IQ expert GEMVs.** The Qwen3.8-Flash-Next experts use
   IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS (gate/up) and IQ4_NL, Q2_0 (down). The
   engine only had one-thread-per-output column GEMVs for those dtypes,
   measured at 10 to 15 GB/s and 66 percent of GPU time. New `_vec` GEMVs
   give one 32-lane group per output and stream the weights at the memory
   rate, on both backends. Ornith does not need this: its Q4_K and Q6_K
   experts already use the `_vec` family.
2. **Fused routed-expert MoE.** Two launches per layer (gate/up with the
   silu product, down with the weighted slot sum), each reading its slice
   from the staging tensor at `ids[slot]*stride`, so no per-expert copy.
   Ornith already had this (`moe_experts_gate_up_q4k`,
   `moe_experts_down_q4k`/`_q6k`); the qwen4exp work only extended it to the
   IQ formats.
3. **Staging-window overlap (ROCm).** The per-batch staging ring used one
   256 MB window, so a layer's batch fit one window and its host memcpy ran
   to completion before the upload. A 4 MB window over an 8-slot ring splits
   a layer's batch into several windows, so the memcpy of the next window
   overlaps the asynchronous upload of the current one. This is in the
   backend and applies to every model, Ornith included.
4. **Shared expert before the routed staging.** The shared expert reads only
   the activation, so running it before the routed staging overlaps its GPU
   work with the host staging copy. Ornith runs the shared expert after the
   fused routed path; the reorder applies there too.
5. **Cache the BF16 shared-expert gate.** The qwen4exp shared-expert gate is
   bf16 and was converted to fp32 with a device-host round trip on every
   layer of every token. Convert once per layer and reuse. Ornith's gate is
   already fp32, so this does not apply.
6. **Pass the QSA position as a scalar.** qwen4exp-specific; the indexer took
   the token position through a per-layer device buffer that the host
   uploaded each layer. Ornith has no QSA.
7. **CPU+GPU hybrid MoE.** The reference for a model larger than VRAM (the
   Strata engine, `docs/HOW_IT_WORKS.md`) does not stream every expert to the
   GPU. It keeps every expert in RAM and has the CPU compute the experts the
   VRAM cache does not hold, concurrently with the GPU computing the cached
   ones. The engine now has `CpuMoeExpertForward` / `CpuMoeExpertsForward`
   (host, multi-threaded) plus the `Model::ExpertCached` and
   `Model::PagedTensorHost` accessors, and the qwen4exp MoE uses them behind
   `TESSERA_CPU_MOE=<n>`. This is the transferable piece; the section below
   applies it to Ornith.

The host CPU limits the hybrid. On this workstation (Intel i7-8700K, 6
cores, AVX2, no AVX-512) the scalar host expert is 8.2 ms against about
0.073 ms for the overlapped GPU staging, so the CPU is roughly a hundred
times slower per expert and any offload slows the token on
Qwen3.8-Flash-Next. Ornith is smaller (about 3B active) and its expert
slices are Q4_K, so the host cost per expert is different; the Ornith
measurement below records what it gives.

## Measurement

### Ornith CPU+GPU hybrid (applied 2026-10-14)

Ornith-1.5-35B-A3B Q4_K_M, ROCm on the R9700, `--context 4096`, greedy,
`--no-chat`, `--gpu-layers 20` (forced paging):

| `TESSERA_CPU_MOE` | decode | output |
| --- | --- | --- |
| off (GPU only) | 36 tok/s (27 ms/token) | `Paris.` |
| 1 | 5 tok/s (169 ms/token) | `Paris.` |
| 2 | 5 tok/s (180 ms/token) | `Paris.` |

Full VRAM (no paging) is unregressed at 87 tok/s, `Paris.`. The hybrid is
correct but slower on this host for the same reason as Qwen3.8-Flash-Next:
the scalar host expert (Q4_K) is far slower than the overlapped GPU
staging. The paged baseline rose from the 14 tok/s recorded below to 36
tok/s because the ROCm staging-window overlap (a 4 MB window over an 8-slot
ring) is in the backend and applies to every model.

Ornith Q4_K_M GGUF, greedy decode, `--gpu-layers 20` (369 tensors
resident, 11.0 GB on device; 384 tensors paged, 10.7 GB in RAM):

- Vulkan (RADV) on the R9700: full VRAM about 88 tok/s (see
  `docs/ORNITH-1.5.md`); paged about 0.4 tok/s. Each paged layer
  uploads about 476 MB per token at about 3.4 GiB/s, so the run is
  PCIe bound as predicted.
- ROCm on the R9700: paged 4 tokens in 14.3 s with the same split and
  the same correct output.

Qwen3.8 27B MXFP4, `--gpu-layers 8` (110 tensors resident, 8.2 GB on
device; 756 tensors paged, 10.9 GB in RAM): one completion token
decodes to `Hello` on Vulkan and on ROCm, identical to the full
upload.

Phase 2 (Ornith Q4_K_M GGUF, `--gpu-layers 20`, Vulkan on the R9700):
8 tokens in 3.2 s (about 2.5 tok/s) with the correct `**Paris**`
output, against about 0.4 tok/s for whole-layer paging. Each paged
layer moves 8 expert slices instead of 256 experts. Batching the
slice uploads into one `CopyH2DBatch` per layer (16 MB in 24 offset
copies, about 5 ms) brings the same run to 2.0 s (about 4 tok/s):
one staging round trip per layer instead of one per expert.

Round 3 (same model, N and prompt, R9700, greedy, byte-identical
output across backends and runs): persistent staging plus flush without
draining raises the effective staging rate to about 7 GB/s, and the
MoE dense refinement cuts the paged set from 384 tensors to 63
expert tensors (resident 11.0 GB to 11.5 GB). N=20 decodes at
14 tok/s on Vulkan and on ROCm with 25 tok/s prefill. N=30
decodes at 22 tok/s. N=40 decodes at 61 tok/s with 92 tok/s
prefill. Cost per token is linear from here: about 8 ms per paged
layer plus about 13 ms fixed, all PCIe bytes at the bus rate.

## Risks

- PCIe traffic sets the paged speed limit. Measure before tuning.
- Prefetch overlap differs by backend. Assert per backend tolerance.
- MXFP4 packed host copy adds RAM use. Account it in the summary line.
- Small N with large context can still fail on KV cache size. The fit check must cover it.
