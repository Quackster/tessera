# `--gpu-layers` progress

This file records the state of the `--gpu-layers` option.
It covers RAM paging with VRAM resident layers.
It covers GGUF and MXFP4 on Vulkan and ROCm.
See `docs/MOE.md` for the offload design.
See `docs/MOE_performance.md` for the measurement history.
See `docs/ORNITH-1.5.md` for the MoE architecture record.

## What it does

`--gpu-layers N` keeps N layers in VRAM.
The rest pages from system RAM per token.
Compute stays on the GPU.
No layer runs on the CPU.
Default `0` uploads everything, the full VRAM behavior.
`all` uploads everything too.

Placement follows llama.cpp order.
The output head stays resident first.
Then trunk blocks go from the last one backward.
`N = 1` keeps the head only.
`N = 2` adds the last block, and so on.

In a model with routed experts, every non-expert block tensor stays
resident too. Only the routed experts page per token. Every token uses
the dense weights whole while touching top_k of num_experts. Dense
models page whole early blocks and are unchanged. The fit check, the
help sizing, and the load summary count the same resident set.

## Status

Paging works end to end on Vulkan and ROCm.
GGUF keeps the mapped file as RAM backing.
MXFP4 converts weights, uploads fully, then evicts the paged set back
to RAM. Staged layers reach the device per token through
`EnsureLayerResident`. Routed experts stage only their top-k slices at
their expert offsets through `EnsureLayerExperts`. Both MoE kernels
keep the unchanged ids. Paged output is bit-identical to full upload.
A per-layer device cache keeps 32 experts per (layer, kind). A repeat
expert gathers from the store into the staging view with a device-side
copy, so only cache misses upload from the host. A kind whose store
fails to allocate uploads straight into staging, so a small device
still decodes.

The transfer path uses one persistent staging window per backend.
Vulkan reuses its copy command buffer and fence. Pending compute
flushes without draining, so one fence wait covers compute plus copy.
`Model` indexes paged tensors per layer. Empty layers skip the round
trip. Prefill uploads each shared expert once per layer instead of
once per row.

## Measurements

Ornith-1.5-35B-A3B Q4_K_M GGUF on the R9700, greedy decode.
Prompt `Explain gravity in one sentence.` unless noted.
Output `Gravity is the natural attraction between any two` and longer
correct continuations, identical on both backends and all runs.

| N | decode tok/s | prefill tok/s | backend | date |
| -- | ------------ | ------------- | ------- | ---- |
| 0 full | 78 to 101 | 114 to 128 | Vulkan | 2026-10-11 |
| 0 full | 89 to 94 | 123 | ROCm | 2026-10-11 |
| 20 | 2.3 | 7.6 | Vulkan | baseline |
| 20 | 5.4 | not recorded | Vulkan | transfer fixes |
| 20 | 14.0 | 25 | Vulkan | dense resident |
| 20 | 13.9 | 26 | ROCm | dense resident |
| 20 | 20.1 | 23 | Vulkan | expert cache |
| 20 | 24.2 | 25 | ROCm | expert cache |
| 30 | 21.6 | 37 | Vulkan | dense resident |
| 38 | 26 | 59 | Vulkan | transfer fixes |
| 40 | 58 to 61 | 92 to 93 | Vulkan | dense resident |

Ornith N=20 detail: 63 paged expert tensors, 10.2 GB in RAM.
Resident grew from 11.0 GB to 11.5 GB with dense residency.
The expert cache holds about 1.3 GB more on the device (32 slices
per layer and kind). Per-token traffic falls from about 330 MB of
expert slices to the miss fraction near 40 percent. Cost per token
falls from about 70 ms to about 49 ms on Vulkan and 41 ms on ROCm.

Qwen3.8 27B MXFP4 with N=8: 110 tensors resident with 8.2 GB on
device, 756 tensors paged with 10.9 GB in RAM. One completion token
decodes to `Hello` on Vulkan and on ROCm, identical to full upload.

Full suite passes on Vulkan and ROCm (240 tests, 0 failures each).
`EngineTest.PagedModelMatchesFullUpload` covers dense paging.
`MoeDecodeTest.PagedMatchesFullUpload` covers expert paging over six
tokens, so cache-hit tokens compare too.
`EngineTest.PagedMoeDenseResidentKeepsMoeDense` covers the
refinement. Backend ref tests cover every staging kernel.

## Remaining work

- Prefetch of layer L plus 1 stays deferred. Transfers are
  bandwidth bound, not round-trip bound. Overlap would hide at most
  layer compute at about 0.3 ms. Re-audited 2026-10-12: the expert
  cache keeps this true. Miss traffic still dominates token cost.
  No async transfer interface exists on either backend.
- Cooperative-matrix tensor cores on Vulkan stay deferred. The
  device exposes the combinations and the shaders compile, but
  cooperative-matrix dispatches wedge this RADV snapshot. One run
  rebooted the machine. Scalar fallbacks deliver the numbers above.
  Re-audited 2026-10-12 on Mesa 26.0.8 RADV GFX1201: the driver is
  unchanged since the crash. Retry risks another reboot. Both WMMA
  kernels stay disabled in source. Retry with a newer RADV before
  re-enabling.

## How to run

Use GPU0. It has the x16 link. GPU1 idles at low clocks and shares
the power budget, so paging numbers from GPU1 run about 5 times
lower and vary run to run.

```sh
./cmake-build-vulkan/tessera-cli run --gpu 0 \
  --model ~/models/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  --gpu-layers 20 \
  --prompt-text "Explain gravity in one sentence." \
  --max-completion-tokens 8
```

```sh
./cmake-build-vulkan/tessera-cli run --gpu 0 \
  --model ~/models/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  --gpu-layers help --kv-q8 --context 262144
```

Protect the machine on every run.
Run one heavy command at a time.
Wait for one model load to finish before the next starts.
Cap each command timeout at 5 minutes.
Never run anything else until a model server exits and frees memory.
Stream the output and watch the whole stream.
Stop at the first sign of garbage and fix the root cause.

## Update rule

Keep this file current after each milestone.
Record what was completed with the date.
Refresh the remaining work list so it always reflects reality.
Add new measurements to the table above.
Update `docs/MOE.md` and `docs/MOE_performance.md` in the same
change when design facts change.
Update the `README.md` features table when user visible behavior
changes. Do not pile up finished milestones without a commit.
