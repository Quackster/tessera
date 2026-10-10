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

## Cards

### P0: device primitives (do first)

**P0.1 Shared ROCm primitives header**

- Objective: one header with the reusable gfx12xx primitives.
- Reference: `../radiance/libr4d/r4d_common.h` lines 42 to 210.
- Change: add `src/backends/rocm/rocm_device.hpp` with DPP wave sum, max,
  XOR reduce, prefix and suffix scans, `LdsBarrier`, `SchedBarrier`,
  software `F32ToBf16` and `Bf16ToF32`, `GlobalLoadTrB128`. Keep each
  function inline. Keep the file under 600 lines.
- Steps: port the functions. Add host-compilable fallbacks guarded by
  `__HIP_DEVICE_COMPILE__` where a builtin is device-only. Add a comment
  naming the LDS-pipe reason for DPP.
- Verify: unit test that the DPP wave sum matches a scalar loop on a
  small buffer. Test on ROCm.
- Accept: `ctest` passes. Vulkan is unaffected because the header is
  ROCm-local.
- Note: parity does not apply to a private header. It is not a kernel.

**P0.2 LDS-scoped barrier adoption**

- Objective: replace `__syncthreads()` with `LdsBarrier()` in every ROCm
  kernel whose barrier only orders LDS traffic.
- Reference: `r4d_common.h` lines 217 to 221.
- Change: `src/backends/rocm/rocm_kernels_*.cpp`, attention, GEMM, GDN.
- Verify: existing tests must still pass, bit-identical where the
  reference is deterministic. Measure one hot kernel before and after if
  possible.
- Accept: no test change needed and no numerical drift.

### P1: GEMM

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

**P6.1 Fused DFlash2 conv**

- Objective: one kernel for the drafter causal conv.
- Reference: `../radiance/libr4d/r4d_dflash_conv_t2_g16_bf16.hip` and
  `r4d_dflash_conv_body.h`.
- Change: `DflashConvKernel` in `rocm_kernels_basic.cpp:1663`,
  `dflash_conv.comp`, and `src/spec/dflash2_conv.cpp`.
- Verify: host-reference test for taps 2 and group 16. Cover the
  block-boundary rule: the conv never crosses an anchor block.
- Accept: matches within tolerance.

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

## Status

| id | title | backend | status |
| --- | --- | --- | --- |
| P0.1 | Shared ROCm primitives header | ROCm | todo |
| P0.2 | LDS-scoped barrier adoption | ROCm | todo |
| P1.1 | int8-WMMA W4A8 GGUF | both | todo |
| P1.2 | MXFP4 decode split-K | both | todo |
| P1.3 | MXFP4 tiled prefill | both | todo |
| P1.4 | Double-buffered tiled LDS | both | todo |
| P2.1 | WMMA transposed attention | both | todo |
| P2.2 | Paged KV and varlen | both | todo |
| P3.1 | Fused GDN conv prep | both | todo |
| P3.2 | GDN chunked scan | both | todo |
| P3.3 | GDN KKT solve | both | todo |
| P3.4 | GDN recurrent update | both | todo |
| P4.1 | Fused rms norm plus Hadamard | both | todo |
| P4.2 | Gated quant and qk norm rope gate | both | todo |
| P5.1 | Two-stage row top-K | both | todo |
| P5.2 | On-device sample chain | both | todo |
| P6.1 | Fused DFlash2 conv | both | todo |
| P6.2 | DFlash2 selector walk | both | todo |

## Done

Move a card here with its commit hash and the measured result.

None yet.
