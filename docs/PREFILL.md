# PREFILL.md: prefill speed work

State: the first prefill pass is done and pushed. This file records the
measured result and what to do next, in the order that pays.

## Measured result

Qwen 3.8 27B GGUF, `--kv-q8`, greedy, 9301-token prompt, one run at a time
on GPU 1:

| Build | Prefill | Rate |
| --- | --- | --- |
| Pre-change binary | 120261 ms | 77 tok/s |
| Current | 15381 ms | 604 tok/s |

One 453-token chunk: 5261 ms before, 639 ms now. The 8090-token prompt of
the serve run scales to about 13.4 s, about 13 percent of the 106 s
baseline. Decode is unchanged near 19 tok/s.

## What the time is now

One 453-token chunk, kernel time 628 ms of a 639 ms wall (99 percent):

| Kernel group | ms | Share |
| --- | --- | --- |
| Tensor-core GEMM tiles | 503 | 80% |
| Delta scan (`delta_step_heads`) | 38 | 6% |
| Gated norm, conv, L2 norm, silu, rmsnorm, repeat, add | 79 | 13% |
| Quantized attention | 8 | 1% |

Commits, newest first: `802d3c4` warp-shuffle row norms, `6698670` 128x64
GEMM tile, `325378a` delta scan dispatch counts lanes, `b69da89` four
query rows per attention workgroup, `0d9efe8` register delta scan and wide
key loads, `734942c` fp16 WMMA tiles and batched KV append.

## Next steps, in order

### 1. Tensor-core GEMM tiles (80 percent of the time)

The tiles reach about 55 TFLOPS. The measured fp16 MMA peak on this card is
126 TFLOPS (`/tmp/opencode/bench_mma`), and the int8 and int4 MMA paths
reach 391 and 784 TFLOPS. The tile spends most of its issue slots staging
the activation tile as fp16 and decoding the weight block, not on MMAs.

- Double buffer the k blocks: stage block k+1 while the MMAs of block k run.
  The kernel is `WmmaGemmBody` in `src/backends/rocm/rocm_kernels_wmma.cpp`.
- Check LDS bank conflicts on the fragment loads. The tile stride is
  `kTileK + 8` halfs (72), which maps `(wr + lane) * 72` to banks unevenly.
- Try a 256-column tile for the wide shapes (`n = 17408`), so one weight
  decode serves more rows per column tile.
- Use the int8 or fp8 MMA path. The served reference runs W4A8 with fp8
  activations, so this also moves tessera toward the reference contract.
  The MXFP4 path already has the pieces (`fp8_pack_kernel`,
  `mxfp4_wmma_kernel` in `rocm_kernels_basic.cpp`); the GGUF block formats
  need a per-row activation quant and block-scale handling.

### 2. Per-row kernels pay block launch overhead

`l2norm` costs 121 us per call for 7248 rows of 128 columns and
`rmsnorm_gated` 424 us for 21744 rows. The work per row is a few hundred
cycles, so the cost is one 1024-thread workgroup per row, most of it launch
and retire. The dispatch comes from the core (`grid = rows`, `block = 1024`
in `src/core/decode_internal.hpp`), and the Vulkan built-ins map one
workgroup per row, so the core cannot shrink the grid on its own.

- Give each workgroup R rows on both backends and let the core dispatch
  `ceil(rows / R)`. Start with R = 8.
- Alternative: let the backend report its rows per workgroup, and size the
  dispatch per backend. This keeps the Vulkan built-ins unchanged.

### 3. Conv history round trip

`conv1d_state` costs 349 us per call. Each row shifts the history through
global memory, so row t waits on the store of row t-1.

- Keep the history in registers or in shared memory across the row loop and
  write it once at the end (and per row when `hist_stride != 0`).
- The width is a runtime scalar, so a register array needs a compile-time
  bound. The shipped model uses width 4. Cap the width at 8 or 16, reject
  more in the launcher, and document the cap in `include/tessera/backend.hpp`.

### 4. Attention

The quantized attention is 8 ms per 453-row chunk and about 3.1 s over the
9301-token run, so it is no longer the first target. If it comes back:

- One workgroup per (row block, kv head) serving every query head of that
  kv head. The decode side already does this (`attention_split_gqa`), and
  the prefill side would share each K row and each V element across six
  query heads.
- Stage the K and V tiles in shared memory instead of re-reading them per
  row block.

### 5. Delta scan

`delta_step_heads` costs 816 us per call, about four times its instruction
bound (measured with the loads removed).

- Stage each row's k and q in shared memory per warp. This removes the two
  per-row global round trips. It needs `dv % 8 == 0`, so the warp's eight
  columns stay inside one head, or a fallback for other widths.

### 6. Chunk size and attention budget

The prefill chunk is 512 tokens and the attention pair budget clamps the
late chunks (`kDefaultPrefillAttnPairs`, `ClampPrefillRows`). The attention
is now four times cheaper, so a larger chunk may pay: fewer chunk
boundaries and better GEMM shapes. Measure with `--prefill-chunk 1024` and
`2048` before changing the default.

## Measurement recipe

Build:

    cmake -B cmake-build-rocm -DTESSERA_BACKEND=rocm && cmake --build cmake-build-rocm -j

One chunk (453 tokens) with a kernel trace:

    rocprofv3 --kernel-trace --stats -d /tmp/opencode/trace -- \
      ./cmake-build-rocm/tessera-cli run \
      --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
      --gpus 1 --context 16384 --kv-q8 --no-chat \
      --prompt-file /tmp/opencode/prompt_512.txt --max-completion-tokens 1

Long prompt: `/tmp/opencode/prompt_9k.txt` (9301 tokens). Aggregate the
trace with a script that sums kernel time by name.

Correctness: run greedy with no completion cap and compare the stream byte
for byte against `cmake-build-rocm-base`, the pre-change binary kept at the
repository root. A numerics change that flips a token shows up here first.

Machine rules from `AGENTS.md`: one heavy command at a time, GPU 1 only,
never a model load in a loop or beside a build.
