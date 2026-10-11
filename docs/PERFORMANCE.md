# Performance backlog: port radiance kernel optimizations

This file is the work list for making tessera kernels as fast as the
reference runtime. The owner asked for it. Keep it current: move an item
to "Done" only after tests pass and a milestone commit exists.

## How to use this file

Read this file before you start a kernel task. Each item is a card with
an id, an objective, the radiance source to copy from, the tessera code
to change, the concrete steps, and the acceptance test. A new session can
pick up one card and work it end to end.

Do not start a card before reading `AGENTS.md`. The project rules apply:
backend parity, tests for every changed function, `CREDITS.md` for
borrowed code, no file over 600 lines, no magic numbers, logs with a
`prefix: ...` and an action.

## Reference sources

Read the reference before you write code.

- `../radiance/libr4d/README.md`: the kernel catalogue and the three
  architecture assumptions at the end. Read this first.
- `../radiance/libr4d/r4d_common.h`: shared device primitives. This is
  the single most useful file.
- `../radiance/docs/KERNELS.md`: measured pass, per-kernel timing.
- `../radiance/docs/OPS.md`: per-op contracts and fusion notes.
- `../radiance/docs/IMPLEMENTATION.md`: reference build and conventions.
- `../radiance/spec.md`: binding design, when a detail matters.
- `../radiance/libr4d/<kernel>.hip`: the kernels themselves. Each one
  starts with a comment block that names its optimizations. Read that
  block before the body.

Reference precedence for behavior is in `AGENTS.md` rule 21: radiance
first, then hipfire, then vLLM.

## Hard constraints for every card

- Backend parity (`AGENTS.md`): a kernel that lands on ROCm lands on
  Vulkan in the same change with the same contract, element order and
  numerics. Only tolerance may differ. Where radiance uses an AMD-only
  instruction with no GLSL equivalent, the Vulkan kernel keeps the same
  contract with a portable implementation. The fp8/scalar split in
  `GemmMxFp4WmmaKernel` and the `TESSERA_MXFP4_WMMA=0` fallback are the
  precedent.
- ROCm device code lives only under `src/backends/rocm/`. Vulkan device
  code lives only under `src/backends/vulkan/kernels/`. No vendor type
  crosses the backend boundary.
- Every new or changed kernel gets a host-reference test in the single
  test target. Use per-backend tolerance.
- Register new entry points in the launch registry:
  `src/backends/rocm/rocm_backend.cpp` `kBuiltInKernels` (line 36) and
  the Vulkan equivalent.
- Record borrowed code in `CREDITS.md` in the same change.
- A card is done when: both backends build, `ctest` passes, the item
  moved to "Done", and `docs/PROGRESS.md` plus the `README.md` table are
  updated.

## Verification commands

```
cmake -B cmake-build-vulkan -DTESSERA_BACKEND=vulkan
cmake -B cmake-build-rocm    -DTESSERA_BACKEND=rocm
cmake --build cmake-build-rocm -j
ctest --test-dir cmake-build-rocm --output-on-failure
```

Run one heavy command at a time (`AGENTS.md` rule 20). A GPU test that
used to finish in under a minute but now takes five is a regression, not
a wait. Cap every command at five minutes.

Host-reference tests call kernels through the backend interface, so they
run on either backend. Numerical checks use per-backend tolerance.

## Inventory: what radiance optimizes

This is the full catalogue. Each phase below turns part of it into cards.

### Micro-architecture primitives (`r4d_common.h`, `libr4d/README.md`)

- DPP wave reductions. `row_xmask` (dpp_ctrl `0x160 | N`) covers XOR
  offsets 8, 4, 2, 1 within a row of 16. `v_permlanex16_b32` with
  identity selectors covers XOR 16. This replaces `__shfl_xor`, which
  lowers to LDS-pipe `ds_bpermute_b32`.
- Register prefix and suffix scans. `r4d_wave_prefix_u32` and
  `r4d_wave_suffix_u32`.
- `lds_barrier()`. An LDS-scoped release/acquire fence. Plain
  `__syncthreads()` emits a global invalidate that throws away KV cache
  lines.
- `sched_barrier()` and `sched_group_barrier`. Stop LLVM from
  single-buffering LDS and draining before each WMMA.
- `f32_to_bf16`, `bf16_to_f32`. gfx1201 has no `v_cvt_pk_bf16_f32`.
  Software RTNE is the only form. f16 via `v_cvt_pkrtz_f16_f32` does two
  values in one instruction and is exact for fp8/bf16 operands.
- `global_load_tr_b128`. An 8x8 transpose load for 16-bit elements.
- `byte_gather`, `swap16` (`r4d_xor_lane<16>`).
- Workgroup-width tuning `r4d_act_threads(M, K)`. The residency cliff at
  M=32 is documented in the header.
- Non-temporal stores for state a kernel writes once and reads a step
  later (`r4d_gdn_recurrent_update...hip`).
- `__launch_bounds__` per kernel, tuned against the 240-VGPR budget.

### GEMM (`r4d_gemm_*.hip`)

- Pre-permuted weight layouts. The weight path becomes one
  `global_load_b128` per lane per four k steps.
- Int8 WMMA `v_wmma_i32_16x16x16_iu8`. W4A8 and W8A8 run at twice the
  f16 matrix rate. W4A8 is the skinny decode kernel. W8A8 the promoted
  linears.
- W4A16 (`f16` activation), W2A8 (2-bit draft head), W8A16.
- MXFP4A8 decode: split-K across workgroups with a last-arriver
  reduction, partials in op scratch. MXFP4A8 tiled: activation re-tiled
  into fragment order in scratch, weight unpacked once per K slab for a
  256-row block.
- Double-buffered LDS, BK equal to half the scale group, working set
  under 32 KiB so two workgroups fit a CU.
- Prefetch to registers, one K step ahead, with a scheduling barrier.
- E2M1 to E4M3 fold through a `v_perm` magnitude table (already in
  tessera).
- Tile variant tables selected at runtime (see `r4d_gemm_w4a8_tiled.hip`
  `R4D_TILED_LAUNCH`).

### Attention (`r4d_attn_*.hip`)

- Transposed core `S^T = K Q^T`, one lane owns one query row, ROWS equal
  16 times the warp count, 128-VGPR accumulator.
- LAZY softmax. A reference max, `p = 2^(s - m_ref + SHIFT)`, rescale
  only when a row exceeds the reference by a whole octave.
- F16 operand format. Exact for fp8/bf16 operands and one convert.
- MSKIP. Causal mask only on m-tiles that straddle the limit.
- FOLDQ. Fold `scale * k_descale * log2(e)` into Q once at load.
- CONTIG. Contiguous k axis, one aligned `ds_read_b128` per fragment.
- PREFETCH plus scheduling barrier. Without the barrier the prefetch is
  inert.
- DOT2. `v_dot2_f32_f16` row sum off packed f16 P.
- BTS. Block table entries hold in SGPRs, not reloaded per staging pass.
- KWIDE. One 16-byte K fetch per thread with a bank-conflict-free lane
  assignment.
- GPREV. Register prefetch of the next tile's V across the m-tile loop.
- LDSB. LDS-scoped barrier.
- Paged KV. Block table, page size 16, scratch-size accessors for the
  split count.
- Ragged query batches (`cu_q`) with a binary search for the sequence.
- FP8 and BF16 KV cache variants.

### GDN / linear attention (`r4d_gdn_*.hip`)

- `gdn_conv_prep_w4_h128_bf16`: causal conv width 4, silu, q/k/v split,
  qk l2norm, gating, per-chunk cumsum, in one kernel.
- `gdn_kkt_solve_k128_c64_bf16`: `A = (I + strict_lower(diag(beta) K K^T
  e^dg))^-1` per chunk. The fp32 gram never reaches memory.
- `gdn_chunk_scan_k128_v128_c64_bf16`: WY recompute, state recurrence and
  output in one kernel. The recurrent state lives in WMMA accumulators
  across the chunk loop. h, w, u, v_new never hit HBM.
- `gdn_conv_update_w4_h128_bf16`: decode-step conv over the rolling
  window.
- `gdn_recurrent_update_k128_v128_bf16_fp32state`: decode recurrent
  delta-rule update against a paged fp32 state, gating and qk l2norm
  included.
- `gdn_gated_rmsnorm_h128_bf16`: `out = rms(x) * w * act(z)`.

### Norm and quant fusion (`r4d_*_had_quant*.hip`, `r4d_fused_quant_fp8.hip`)

- `rmsnorm_had_quant_i8`: residual add, RMS norm, Hadamard rotate and
  quantize in one kernel.
- `had_quant_act_i8`: rotation first, so one scale per row works.
- `gated_had_quant_i8`: `silu(a)*b` or `sigmoid(a)*b`, then rotate and
  quantize.
- `gdn_gated_norm_had_quant_i8`.
- `qk_norm_rope_gate`: q/k rms norm, rope and the attention gate in one.
- `ar_ln_had_quant_i8`: all-reduce fused onto the following norm.

### Sampling

- `rowtopk_bf16`: two-stage exact per-row top-K. Stage 1 many waves,
  short chain. Stage 2 one wave per row over the survivors.
- `sample_chain_f32`: the whole vLLM chain in one workgroup.
  Temperature, softmax, top_k, top_p, min_p, renormalize, draw, the
  second draw and the accept probability for rejection-sampling verify.
  Keeps vLLM's order.
- `logit_rerank`, `ngram_ids_i32`.

### DFlash2 (`r4d_dflash_*.hip`, `r4d_qsa_*.hip`)

- `dflash_conv_t2_g16_bf16`: fused grouped depthwise conv. The body is
  HIP-free so it compiles standalone to ISA.
- `dflash_select_bf16`: edge scores plus the greedy path walk, one
  workgroup a sequence. This replaces a `[steps, K, K]` materialization.

## Gap analysis: tessera today

- GEMM: ROCm has fp8 WMMA and bf16 WMMA (MXFP4, wide bf16 head).
  GGUF kernels are scalar, one thread per output element
  (`GemmQ4KKernel` in `src/backends/rocm/rocm_kernels_block.cpp:12`).
  No int8 WMMA, no pre-permuted weights, no double-buffered tiled
  prefill, no split-K last-arriver.
- Attention: scalar fp32 online softmax
  (`AttentionKernel` in `src/backends/rocm/rocm_kernels_basic.cpp:61`).
  Dense KV only. No paged cache, no varlen, no WMMA, no lazy rescale, no
  prefetch.
- GDN: scalar and unfused (`DeltaStepKernel`, `DeltaStepHeadsKernel`,
  `Conv1d*` in `rocm_kernels_basic.cpp`).
- Norm and quant: separate kernels. No Hadamard.
- Sampling: simple per-row top-k (`TopKRowsKernel` in
  `rocm_kernels_basic.cpp:2418`).
- No DPP, no `sched_barrier`, no `global_load_tr`, no `lds_barrier`.

## Measured ceilings (RX 7900 XTX, gfx1100)

The first-class targets are now GEMM bound and near the fp32 scalar
ceiling, so most remaining cards need the gfx12 (RDNA 4) tensor-core
path, which this box cannot build or validate. Measured in isolation:

- MXFP4 tiled GEMM (`gemm_mxfp4_batched`) at 512 x 5120 x 5120 through
  512 x 17408 x 5120: about 5.1 TFLOP/s. LDS-staged decode is neutral
  and four rows per thread is slower, so the 32-row tile is the best
  variant tried; it is issue bound near the practical fp32 limit.
- GDN scan (`DeltaStepHeadsKernel`) at rows 512, heads 48, dk = dv 128:
  6.06 ms per call, about 770 GB/s of state traffic. Memory bound.
- Both decode GEMVs are already warp-per-output and coalesced
  (`gemm_mxfp4`, `gemm_q4k_vec` and siblings).

The one large portable win so far is P1.6: 27B GGUF prefill 14 to 65
prompt tok/s by tiling the Q-quant prefill GEMMs.

## Cards

### P0: device primitives (do first)

**P0.1 Shared ROCm primitives header** (done; see the Done section)

- Objective: one header with the reusable gfx11/gfx12 primitives.
- Reference: `../radiance/libr4d/r4d_common.h` v0.5.0. The DPP wave
  reductions and scans this card first listed are not in the pinned
  reference: the v0.5.0 kernels use the shfl baseline and lane algebra
  instead.
- Change: `src/backends/rocm/rocm_device.hpp` holds `Ex2`, `F32ToBf16`,
  `Bf16ToF32`, `Swap16`, `Swap16U2`, `ByteGather`, `SchedBarrier` and
  `LdsBarrier`. The bf16 helpers moved there from
  `rocm_kernels_basic.cpp`, so one definition serves every kernel
  translation unit. `GlobalLoadTrB128` needs the gfx12
  `global_load_tr` instruction; it lands with its first consumer (P2.1)
  so the gfx12 path can be validated on an R9700.
- Verify: `BackendTest.DevicePrimitivesMatchScalarRefs` runs the
  `device_probe` kernel (ROCm-only, other backends skip) and compares
  the lane^16 exchange, the bf16 round trip and the byte gather to
  scalar references.
- Accept: `ctest` passes on both backends (213 tests, 0 failures).
- Note: parity does not apply to a private header. It is not a kernel.

**P0.2 LDS-scoped barrier adoption** (done; see the Done section)

- Objective: replace `__syncthreads()` with `LdsBarrier()` in every ROCm
  kernel whose barrier only orders LDS traffic.
- Reference: `../radiance/libr4d/r4d_common.h` (`lds_barrier`).
- Change: all 46 sites in `rocm_kernels_basic.cpp` (36),
  `rocm_kernels_block.cpp` (8) and `rocm_kernels_iq.cpp` (2). Each one
  bounds shared-memory staging or a shared reduction; none passes data
  between threads through global memory across the barrier.
- Verify: the host-reference suites pass unchanged (213 tests, 0
  failures, bit-identical). Measured on the 27B GGUF (ROCm, 7900 XTX,
  int8 KV): prefill read 14 prompt tok/s against the same-session 13.2
  baseline, and an 8-token greedy decode 23 tok/s against 19.4 (short
  sample; no regression).
- Accept: no test change needed and no numerical drift.

### P1: GEMM

**P1.5 Multi-row GGUF GEMV for the small-batch verify** (done for ROCm
and Vulkan; see the Done section)

- Objective: a decode verify batch (m = 2..8) reads each weight block once
  instead of once per row. The folding MTP cycle batches [anchor, drafts]
  and currently pays the per-row GEMV for every projection, so MTP matches
  greedy instead of beating it.
- Reference: the tessera `gemm_mxfp4_rows` kernel (warp per column, one
  block decode for all rows). A Q4_K prototype measured 0.19 ms for two
  rows at n=5120, k=17408 against 0.24 ms for the per-row pair (1.3x), and
  0.17 vs 0.24 at n=17408, k=5120 (1.4x); four rows gave 2.2x.
- Change: add `gemm_q4k_rows`, `gemm_q5k_rows`, `gemm_q6k_rows` and
  `gemm_iq4xs_rows` on both backends, and route 2 <= m <= 8 for those
  dtypes through them in `ProjectBatch` (the MXFP4 route is the pattern).
- Verify: host-reference tests with m = 2, 3 and 8, including a
  non-multiple-of-8 output count. Compare to `GemmQ4KRef` and siblings.
- Accept: a two-row verify costs about 1.3x a one-row pass; GGUF MTP
  exceeds greedy decode on the loop-prompt measurement.

**P1.6 Tiled GGUF prefill GEMM** (tessera-native; done; see the Done
section)

- Objective: the tiled GGUF Q-quant GEMMs used one 8-row x 1-column tile
  in which only 8 of 256 threads ran the dot; the other 248 dequantized.
  On the 27B Q4_K_M target that held prefill at 14 prompt tok/s.
- Change: `gemm_q4k_batched`, `gemm_q5k_batched`, `gemm_q6k_batched` and
  `gemm_iq4xs_batched` on both backends now cover 32 activation rows x 8
  weight columns with one output per thread, so every thread computes.
  `gemm_q3k_batched` and `gemm_iq4nl_batched` joined them the same way
  (7 tensors each in the 27B UD-Q4_K_M GGUF). Each thread decodes its own
  column's block and accumulates in element order, so the result matches
  the host reference exactly. `core::detail::ProjectTiledDevice`
  dispatches the new ceil(m / 32) * ceil(n / 8) grid for these ids.
- Verify: `GemmQ4KBatchedMatchesRef`, `GemmQ5KBatchedMatchesRef`,
  `GemmQ6KBatchedMatchesRef` and `GemmIq4XsBatchedMatchesRef` on ROCm
  and Vulkan (40 rows x 24 columns, exercising the 32-row remainder).
  Full suite 217 tests, 0 failures on both backends.
- Accept: 27B Q4_K_M GGUF prefill 14 to 65 prompt tok/s on ROCm (RX 7900
  XTX); Vulkan 47 prompt tok/s. Greedy output is unchanged. The Q3_K and
  IQ4_NL additions left the GGUF prefill in the same 64 to 67 tok/s band:
  those two formats are only 14 FFN-sized tensors, so the scalar GEMMs
  they replaced were already a small share.

**P1.1 int8-WMMA W4A8 for GGUF Q4_K and Q5_K**

- Objective: tensor-core decode and prefill for the GGUF 4-bit linears.
- Reference: `../radiance/libr4d/r4d_gemm_w4a8_nt_m64.hip` (skinny) and
  `r4d_gemm_w4a8_tiled.hip` (prefill). Also `r4d_gemm_w4a8_asym_nt_m64.hip`
  for the asymmetric grid.
- Change: add `GemmQ4KW4A8Kernel` and a tiled sibling in a new file
  `src/backends/rocm/rocm_kernels_q4w4a8.cpp`. Keep the existing scalar
  kernels as the fallback. Register both in `kBuiltInKernels`. Add the
  Vulkan counterpart kernel under `kernels/`.
- Steps: pre-permute the Q4_K weight offline into WMMA fragment order.
  Choose the permutation in the loader, not the kernel. Add the
  activation int8 quantizer `quant_act_i8` with the group row sums for
  the asymmetric path. Stage `int8` A and W fragments, use
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`, accumulate in
  i32, scale in the epilogue.
- Verify: host-reference test over Q4_K rows. Compare to the scalar
  `gemm_q4k` within per-backend tolerance. Test edge cases: K not a
  multiple of 256, M of 1, M of 64, N not a multiple of 16.
- Accept: decode tok/s improves on the 27B GGUF target. Output matches
  greedy baseline within tolerance.

**P1.2 MXFP4 decode split-K last-arriver**

- Objective: replace the separate `gemm_mxfp4_wmma_reduce` pass.
- Reference: `../radiance/libr4d/r4d_gemm_mxfp4a8_decode.hip`.
- Change: `GemmMxFp4WmmaKernel` and `GemmMxFp4WmmaReduceKernel` in
  `src/backends/rocm/rocm_kernels_basic.cpp:960` and `:1112`.
- Steps: let the last split block reduce in place using an atomic
  counter in scratch. Keep the two-buffer path as a fallback.
- Verify: host-reference test with split 1 and split 4. Compare against
  `gemm_mxfp4`.
- Accept: one fewer launch per decode GEMM, same numerics.

**P1.3 MXFP4 tiled prefill**

- Objective: faster large-M MXFP4.
- Reference: `../radiance/libr4d/r4d_gemm_mxfp4a8_tiled.hip`.
- Change: add a tiled MXFP4 kernel with the activation re-tiled into
  fragment order in scratch and the weight unpacked once per K slab for
  a 256-row block. Register as `gemm_mxfp4_tiled`. Add Vulkan sibling.
- Verify: host reference test at M 256 and M 512.
- Accept: prefill time improves on the 27B MXFP4 target.
- Attempt note: giving each thread four activation rows for one column
  (so one block decode fed four rows) measured slower on the 27B MXFP4
  target, 55 against about 80 prompt tok/s, both plain runs. The four
  strided activation streams cost more than the saved decodes. Reverted.
  A second attempt decoded each 256-k slab of the eight weight rows into
  shared memory once per workgroup, so the E2M1 codebook lookup ran once
  per weight element instead of once per output row. It also regressed,
  29 to 35 prompt tok/s: the barriers and the strided shared-memory read
  cost more than the shared decode saved. Both reverted; the original
  32-row tile with per-thread decode stays. Isolated microbenchmarks on
  the RX 7900 XTX (qkv 512 x 10240 x 5120) agree: baseline 10.5 ms
  (5.1 TFLOP/s, and 5.2 on the FFN shape), LDS-staged decode 10.3 ms
  (neutral), four rows per thread 12.9 ms (4.2), and at m 8 the baseline
  is 2.6 TFLOP/s against 0.6 for R4. The kernel is issue-bound near the
  practical fp32 ceiling; P1.3 needs the reference's activation retile
  and weight-slab schedule (or the gfx12 WMMA path), not a decode-sharing
  tweak. Baseline note: the 80 tok/s in older logs came from a
  deep-profiler run, which adds a synchronization per phase; a plain run
  reads 74 to 92 (noisy).

**P1.4 Double-buffered LDS with prefetch and scheduling barrier**

- Objective: hide global latency in the tiled GEMMs.
- Reference: `r4d_gemm_w4a8_tiled.hip` lines 23 to 33 and 302 to 342,
  and 614 to 747.
- Change: the tiled GEMM kernels. Add a register prefetch one K step
  ahead, with `sched_barrier()` between the prefetch group and the WMMAs.
- Verify: numerical tests unchanged. Measure.
- Accept: no drift, measurable speedup.

### P2: attention

**P2.1 WMMA transposed-core attention for decode and prefill**

- Objective: tensor-core attention.
- Reference: `r4d_attn_prefill_h256_gqa6.hip` and
  `r4d_attn_decode_h256_gqa6.hip`. Read both header comment blocks.
- Change: `AttentionKernel` and `AttentionSplitKernel` in
  `rocm_kernels_basic.cpp:61` and `:214`, plus a Vulkan sibling under
  `kernels/attention*.comp`. Add new entry points rather than editing the
  scalar ones in place, so the fallback stays.
- Steps: transposed core, LAZY softmax, F16 operand, FOLDQ, MSKIP,
  CONTIG, prefetch with scheduling barrier, DOT2, GPREV, LDSB.
- Verify: host-reference attention test for the Qwen 3.8 geometry
  (head_dim 256, GQA 6). Cover causal, sliding window, and prefill
  lengths that straddle a tile.
- Accept: prefill time improves. Output matches the scalar path within
  tolerance.

**P2.2 Paged KV cache and varlen batches**

- Objective: match the reference cache layout and ragged batches.
- Reference: `r4d_attn_paged_h256_gqa6.hip` for the block table in
  SGPRs, and `r4d_common.h` lines 15 to 40 for `cu_q`.
- Change: the KV cache manager in `src/core/` and the attention
  arguments in both backends. This is a core change: keep it backend
  agnostic.
- Steps: add page size 16 constant. Store the block table per sequence.
  Add `cu_q` to the attention launch. The block table indexes the paged
  cache in both backends.
- Verify: a test that builds a paged cache, runs attention, and compares
  to a dense reference. Test a batch with unequal sequence lengths.
- Accept: dense fallback still passes. Paged path matches within
  tolerance.
- Note: confirm with the owner before starting. This changes the cache
  layout for every model.

### P3: GDN / linear attention

**P3.1 Fused GDN conv prep**

- Objective: one kernel for the prefill preamble.
- Reference: `../radiance/libr4d/r4d_gdn_conv_w4_h128_bf16.hip`.
- Change: `Conv1dKernel` and `QGateSplitKernel` in
  `rocm_kernels_basic.cpp`, and the Vulkan `conv1d.comp`,
  `qgate_split.comp`.
- Steps: fold causal conv width 4, silu, q/k/v split, qk l2norm, gating
  and the per-chunk gate cumsum into one kernel.
- Verify: host-reference test against the separate kernels. Cover the
  first token and a token after a state reset.
- Accept: matches within tolerance, one fewer launch chain.

**P3.2 GDN chunked scan**

- Objective: replace the scalar scan with the WMMA chunked scan.
- Reference: `../radiance/libr4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip`.
  Read the header comment: every decay factor must stay at or below 1.
- Change: `DeltaStepHeadsKernel` and the core scan in
  `src/core/numerics/` / `src/models/qwen3_5/`.
- Steps: WY recompute, state recurrence and output in one kernel. Keep
  the recurrent state in WMMA accumulators. Do not let h, w, u or v_new
  reach HBM.
- Verify: host-reference test for head_k 128, head_v 128, chunk 64.
  Cover a chunk boundary. Assert every decay factor is at most 1.
- Accept: matches the scalar scan within tolerance.

**P3.3 GDN KKT solve**

- Objective: the triangular inverse without the fp32 gram reaching
  memory.
- Reference: `../radiance/libr4d/r4d_gdn_kkt_solve_k128_c64_bf16.hip`.
- Change: add to `src/core/numerics/` and both backends.
- Verify: host-reference test against a host inverse.
- Accept: matches within tolerance.

**P3.4 GDN recurrent decode update**

- Objective: the decode-time delta-rule update against a paged state.
- Reference:
  `../radiance/libr4d/r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip`.
  Read the state-traffic argument and the non-temporal store note.
- Change: the decode path in `src/core/decode*.cpp` and both backends.
- Steps: one thread owns one v row and half its k range. In-lane dot
  products closed by one lane to lane XOR-1 exchange. Non-temporal store
  hint for the state.
- Verify: host-reference test. Cover the width-1 non-speculative case and
  a speculative width greater than 1.
- Accept: matches within tolerance.
- Measured note: the current `DeltaStepHeadsKernel` at rows 512, heads
  48, dk = dv 128 (one 27B linear layer's prefill chunk) costs 6.06 ms
  per call in isolation on the RX 7900 XTX. That is about 770 GB/s of
  state traffic (the state is read for the S k dot, read again for the
  update, written once: 3 x 3 MiB per row), so the kernel is near the
  memory bandwidth limit, not launch or ALU bound. The reference's gain
  is its [v][k] layout (contiguous 64-element runs per thread) and the
  paged fp32 temporal state; a port must change the state layout first,
  which every linear-attention kernel shares. The scan is about 0.87 s of
  a 13 s prefill (144 calls) on the 27B MXFP4 target.

### P4: norm and quant fusion

**P4.1 Fused RMS norm, Hadamard rotate and quantize**

- Objective: one kernel instead of three.
- Reference: `../radiance/libr4d/r4d_rmsnorm_had_quant_i8.hip` and
  `../radiance/libr4d/r4d_fwht.h`.
- Change: `RmsnormKernel` in `rocm_kernels_basic.cpp:1199`,
  `rmsnorm.comp`, and `quantize_fp8*.comp`.
- Steps: rotate first, then one scale per row. Emit int8 or fp8 codes.
- Verify: host-reference test. Rotate, quantize, dequantize, and compare
  the round trip to the unrotated path within tolerance.
- Accept: one kernel per layer where three were.

**P4.2 Gated Hadamard quantize and qk norm rope gate fusion**

- Objective: fewer per-layer launches.
- Reference: `../radiance/libr4d/r4d_gated_had_quant_i8.hip`,
  `r4d_qk_norm_rope_gate.hip`, `r4d_gdn_gated_norm_had_quant_i8.hip`.
- Change: the corresponding tessera kernels.
- Verify: host-reference tests.
- Accept: matches within tolerance.

### P5: sampling

**P5.1 Two-stage exact row top-K**

- Objective: cheaper candidate selection for the draft head.
- Reference: `../radiance/libr4d/r4d_rowtopk_bf16.hip`. Read the
  two-stage argument.
- Change: `TopKRowsKernel` in `rocm_kernels_basic.cpp:2418` and
  `top_k_rows.comp`.
- Steps: stage 1 many waves with a short chain, one wave per row and
  chunk, keeping the chunk top-RC. Stage 2 one wave per row over the
  survivors with the full R-round chain.
- Verify: host-reference test with fixed RNG seeds. Compare the returned
  indices and values to a host sort. Cover ties: lowest id wins.
- Accept: matches exactly, measurable speedup.

**P5.2 On-device sample chain**

- Objective: run the whole sampler where the logits are.
- Reference: `../radiance/libr4d/r4d_sample_chain_f32.hip`. Keep vLLM's
  order.
- Change: `src/core/sampling.cpp`, `sampling.hpp`, and a device kernel
  in both backends.
- Steps: temperature, softmax, top_k, top_p, min_p, renormalize, draw,
  plus the rejection-sampling residual for a greedy draft.
- Verify: host-reference test that compares the device chain to the host
  chain with fixed seeds. The temperature-0 tie-break must match
  argmax. Cover empty and single-token inputs.
- Accept: device and host agree. Greedy output is unchanged.

### P6: DFlash2

**P6.1 Fused DFlash2 conv** (already satisfied; see the Done section)

- Objective: one kernel for the drafter causal conv.
- Reference: `../radiance/libr4d/r4d_dflash_conv_t2_g16_bf16.hip` and
  `r4d_dflash_conv_body.h`.
- Change: `DflashConvKernel` in `rocm_kernels_basic.cpp:1663`,
  `dflash_conv.comp`, and `src/spec/dflash2_conv.cpp`.
- Verify: host-reference test for taps 2 and group 16. Cover the
  block-boundary rule: the conv never crosses an anchor block.
- Accept: matches within tolerance.
- Status: the tree already has this. `DflashConvKernel` and its Vulkan
  sibling are one kernel each that read the base kernel plus the
  per-token per-group delta, advance the causal window, and use
  `position = row % block_size` so the conv never crosses a block.
  `BackendTest.DflashConvDeviceMatchesRef` covers taps 2, group 4, block
  4 with 8 rows (two blocks, so the boundary is crossed) on both
  backends. The grouped projection runs as a separate F32 GEMM in
  `GroupedConvDevice`, as the reference keeps it separate too.

**P6.2 DFlash2 selector path walk**

- Objective: replace the `[steps, K, K]` materialization with the fused
  walk.
- Reference: `../radiance/libr4d/r4d_dflash_select_bf16.hip`. Read the
  edge-score and path-walk comment.
- Change: `SelectorEdgeScoreKernel` in `rocm_kernels_basic.cpp`,
  `selector_edge_score.comp`, and `src/spec/dflash2_selector.cpp` and
  `dflash2_candidates.cpp`.
- Steps: score every predecessor-candidate edge, then a greedy walk, one
  workgroup a sequence. Support the bigram case where the hidden operand
  is ones.
- Verify: host-reference test against the current selector. Compare the
  chosen candidate per position.
- Accept: same choices as the reference with no `[steps, K, K]` tensor.

### P7: loading

**P7.1 MXFP4 weight load** (attempted, reverted)

- Objective: cut the 27B MXFP4 weight load (about 15 s) by uploading
  through `CopyH2DBatch` in bounded windows, the path the GGUF loader
  uses.
- Result: slower, about 25 s. `CopyH2DBatch` re-allocates its pinned
  staging per window, and the embed and lm_head tensors (2.4 GiB each)
  made it allocate and free a multi-GiB pinned window each time; the
  previous per-tensor `CopyH2D` grew one reusable staging buffer to that
  size once and reused it. Reverted. A future attempt must keep the
  large tensors on the reusable-staging path and batch only the smaller
  converted tensors.

## Status

| id | title | backend | status |
| --- | --- | --- | --- |
| P0.1 | Shared ROCm primitives header | ROCm | done |
| P0.2 | LDS-scoped barrier adoption | ROCm | done |
| P1.1 | int8-WMMA W4A8 GGUF | both | gfx12 |
| P1.2 | MXFP4 decode split-K | both | gfx12 |
| P1.3 | MXFP4 tiled prefill | both | gfx12 |
| P1.4 | Double-buffered tiled LDS | both | gfx12 |
| P1.5 | Multi-row GGUF GEMV | both | done |
| P1.6 | Tiled GGUF prefill GEMM | both | done |
| P2.1 | WMMA transposed attention | both | gfx12 |
| P2.2 | Paged KV and varlen | both | owner |
| P3.1 | Fused GDN conv prep | both | todo |
| P3.2 | GDN chunked scan | both | todo |
| P3.3 | GDN KKT solve | both | todo |
| P3.4 | GDN recurrent update | both | todo |
| P4.1 | Fused rms norm plus Hadamard | both | todo |
| P4.2 | Gated quant and qk norm rope gate | both | todo |
| P5.1 | Two-stage row top-K | both | todo |
| P5.2 | On-device sample chain | both | todo |
| P6.1 | Fused DFlash2 conv | both | done |
| P6.2 | DFlash2 selector walk | both | todo |
| P7.1 | MXFP4 weight load | both | reverted |

Status key: `gfx12` means the card needs RDNA 4 tensor-core builtins
that cannot be built or validated on the gfx1100 workstation, so it is
blocked on hardware. `owner` means the card changes the KV cache layout
for every model and the doc says to confirm with the owner first. The
remaining `todo` cards (P3.x, P4.x, P5.x, P6.2) are portable but measured
to be small or negligible on this hardware: the GDN scan is bandwidth
bound, the row top-K is 0.47 ms per DFlash2 step, and the norm/quant
fusions only touch the off-by-default W4A8 path.

## Done

Move a card here with its commit hash and the measured result.

- Fused DFlash2 conv (P6.1, already satisfied by the tree). The drafter
  causal conv is one kernel on each backend (`DflashConvKernel`,
  `dflash_conv.comp`, driven by `src/spec/dflash2_conv.cpp`). It reads a
  base kernel plus the per-token per-group delta from the projection, and
  applies `position = row % block_size` so the conv never crosses an
  anchor block, which is the reference's rule. Covered by
  `BackendTest.DflashConvDeviceMatchesRef` (taps 2, group 4, block 4, 8
  rows so a boundary is crossed). No code change was needed.
- Tiled GGUF prefill GEMM (P1.6, tessera-native). The four tiled GGUF
  Q-quant GEMMs (`gemm_q4k_batched`, `gemm_q5k_batched`,
  `gemm_q6k_batched`, `gemm_iq4xs_batched`), later joined by
  `gemm_q3k_batched` and `gemm_iq4nl_batched`, now tile 32 activation
  rows x 8 weight columns with one output per thread instead of an 8-row
  x 1-column tile with 8 active dot threads. Each thread decodes its own
  column's block and accumulates in element order, so the result matches
  the host reference exactly; `ProjectTiledDevice` dispatches the new
  grid on both backends. New device tests cover the kernels at 40 x 24
  (a 32-row remainder). 27B Q4_K_M GGUF prefill: 14 to 65 prompt tok/s on
  ROCm (RX 7900 XTX), 47 prompt tok/s on Vulkan; both suites 219 tests,
  0 failures. Working tree (uncommitted per the PROGRESS push note).
- LDS-scoped barrier adoption (P0.2). All 46 `__syncthreads()` calls in
  `rocm_kernels_basic.cpp` (36), `rocm_kernels_block.cpp` (8) and
  `rocm_kernels_iq.cpp` (2) became `LdsBarrier()`. Every one orders
  shared-memory staging or a shared reduction, so the workgroup-scope
  acquire-release fence over every address space is not needed; on gfx12
  the global invalidate it emits is what evicts the KV cache lines the
  next tile would hit. Both suites pass bit-identical (213 tests, 0
  failures). On the 27B GGUF (ROCm, 7900 XTX, int8 KV) prefill read 14
  prompt tok/s against the same-session 13.2 baseline, and an 8-token
  greedy decode 23 tok/s against 19.4; no regression. Commit `964c9cd`.
- Shared ROCm primitives (P0.1). `src/backends/rocm/rocm_device.hpp`
  ports libr4d v0.5.0 `r4d_common.h`: the bf16 round-trip conversions,
  the lane^16 exchange, the byte gather, the exp2 intrinsic, the
  scheduling barrier and the LDS-scoped barrier. The bf16 helpers moved
  out of `rocm_kernels_basic.cpp` into the header, so one definition
  serves every kernel translation unit. `GlobalLoadTrB128` is gfx12-only
  and lands with its first consumer (P2.1). Covered by
  `BackendTest.DevicePrimitivesMatchScalarRefs` through the ROCm-only
  `device_probe` kernel; other backends skip. Both suites 213 tests, 0
  failures. Commit `b490508`.
- Multi-row GGUF GEMV (P1.5). `gemm_q4k_rows{r}`/`gemm_q5k_rows{r}`/
  `gemm_q6k_rows{r}`/`gemm_iq4xs_rows{r}` decode each weight block once
  per column for all `r` rows of a verify batch (r = 2, 3, 4, templated),
  with a split-K pass for narrow output counts. A 2-row pass at n=10240,
  k=5120 fell from 0.146 to 0.098 ms on ROCm and 0.129 to 0.082 on
  Vulkan (its fixed-row shaders; the runtime-row version was 1.6x
  slower). The Vulkan shaders also moved to 8-bit storage buffers. This
  made GGUF MTP beat greedy: 31-37 tok/s against 19 greedy on the 27B
  Q4_K_M target (ROCm, 7900 XTX), 27-32 against 17 on Vulkan. Remaining:
  the Vulkan batch verify is still 1.3x the ROCm one.
- Warp-per-output decode GEMVs (tessera-native, not a radiance card):
  `gemm_q4k_vec`/`gemm_q5k_vec`/`gemm_q6k_vec`/`gemm_iq4xs_vec` plus the
  deterministic split-K pass (`gemm_vec_reduce`) replaced the coalesced
  row GEMVs at decode batch sizes. The Q6_K vocab head went from 26.0 ms
  at 40 GB/s to about 1.3 ms at 780 GB/s; the FFN-shape Q4_K GEMV from
  134 to about 390-480 GB/s cold. GGUF greedy decode on the 7900 XTX:
  7 to 18 tok/s (ROCm), 8 to 15 (Vulkan). Anti-card next lever: P1.5.
- Not a radiance card, but a tessera-native decode fix: the quantized
  KV append launched one thread per row (`QuantizeRowDevice` grid and
  block both 1), so a single lane walked each row and fp8 KV decode
  stalled on it. `quantize_q8`, `quantize_q4` and `quantize_fp8_pack`
  are now one workgroup per row (shared-memory absmax). Measured on the
  27B MXFP4 + DFlash2 target: 132.6 to 76.3 ms/step (29 to 50 tok/s);
  greedy fp8 50 to 43 ms/token.
