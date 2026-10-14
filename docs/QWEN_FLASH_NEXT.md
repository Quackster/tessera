# Qwen3.8-Flash-Next IQ3_S on tessera

This document tracks the work to run the `Qwen3.8-Flash-Next` model
(`general.architecture = qwen4exp`, GGUF name `Qwen3.8-Flash-Next-GSQ-RCO`,
IQ3_S size) on tessera, and to bring it to at least 48 output tokens per
second. The port source is the Strata engine at `../Strata`. No other source
is used.

This file is a plan and a status report. It does not claim the port is
complete. Below, "done" means the code is in the tree and tested on the
configured backend. "Plan" means it is not written yet.

## Branch

All work for this model goes to the branch `arch/qwen-flash-next`. Commit and
push each finished milestone to that branch. Do not work on `main`.

## 1. The target

The model is a 48 layer sparse mixture-of-experts transformer with three
special mechanisms. The GGUF metadata is:

| Key | Value |
| --- | --- |
| `general.architecture` | `qwen4exp` |
| `block_count` | 48 |
| `embedding_length` | 2560 |
| `attention.head_count` | 24 |
| `attention.head_count_kv` | 2 |
| `attention.key_length` / `value_length` | 256 |
| `expert_count` | 512 |
| `expert_used_count` | 10 |
| `expert_feed_forward_length` | 640 |
| `expert_shared_feed_forward_length` | 640 |
| `ssm.conv_kernel` | 4 |
| `ssm.state_size` | 128 |
| `ssm.group_count` | 16 |
| `ssm.time_step_rank` | 48 |
| `ssm.inner_size` | 6144 |
| `full_attention_interval` | 4 |
| `rope.dimension_count` | 64, sections `[11, 11, 10, 0]`, base `1e7` |
| `hyper_connection.count` | 4 |
| `hyper_connection.low_rank` | 320 |
| `attention.indexer.head_count` | 4 |
| `attention.indexer.key_length` | 128 |
| `attention.indexer.top_k` | 2048 |
| `attention.compress_ratios` | 0 except 4 on every layer where `l % 4 == 3` |
| `ple.layers` | `[1]` |
| `ple.ngram_size` | 3 |
| `ple.heads_per_ngram` | 8 |
| `ple.conv_kernel` | 4 |
| `embedding_length_per_layer_input` | 160 |
| `ple.head_offsets` | 16 values |
| `ple.head_vocab_sizes` | 16 values, about 20,000,003 each |
| `tokenizer.ggml.pre` | `qwen35` |
| `eos_token_id` | 248046 |

The two GGUF files are two shards of one model:

- `...-00001-of-00002.gguf` (54.8 GB): 1223 tensors, all weights except the
  PLE table.
- `...-00002-of-00002.gguf` (28.8 GB): one tensor,
  `per_layer_token_embd.weight`, shape `[160, 320001536]`. This is the
  51 GB PLE n-gram table.

There is no MTP draft file in the downloaded folder. MTP is a separate
download in Strata. It is out of scope for the first end to end milestone.

### 1.1 Per layer tensor inventory

Every layer has, in addition to the MoE group:

- `hc_attn_{norm,down,up,inject}.weight` and
  `hc_ffn_{norm,down,up,inject}.weight`: the hyper-connection (gated
  residual) mixer. `norm` is `[10240]` f32, `down` is `[10240, 320]` bf16,
  `up` is `[320, 10240]` bf16, `inject` is `[10240, 4]` bf16.
- `ffn_gate_inp.weight [2560, 512]`, `ffn_gate_inp_shexp.weight [2560]`,
  `ffn_gate_exps.weight [2560, 640, 512]`,
  `ffn_up_exps.weight [2560, 640, 512]`,
  `ffn_down_exps.weight [640, 2560, 512]`,
  and the shared expert `ffn_{gate,up,down}_shexp.weight`.

A GDN layer (36 of 48) has:
`attn_qkv.weight [2560, 10240]`, `attn_gate.weight [2560, 6144]`,
`ssm_a [48]`, `ssm_dt.bias [48]`, `ssm_alpha.weight [2560, 48]`,
`ssm_beta.weight [2560, 48]`, `ssm_conv1d.weight [4, 10240]`,
`ssm_norm.weight [128]`, `ssm_out.weight [6144, 2560]`.

A QSA layer (12 of 48, `l % 4 == 3`) has:
`attn_q.weight [2560, 12288]`, `attn_k.weight [2560, 512]`,
`attn_v.weight [2560, 512]`, `attn_output.weight [6144, 2560]`,
`attn_q_norm.weight [256]`, `attn_k_norm.weight [256]`,
`indexer.k_proj.weight [2560, 128]`, `indexer.q_proj.weight [2560, 512]`,
`indexer.k_norm.weight [128]`, `indexer.q_norm.weight [128]`.

Layer 1 alone carries the PLE weights:
`ple_key.weight [2560, 10240]`, `ple_value.weight [2560, 2560]`,
`ple_conv1d.weight [4, 10240]`,
`ple_norm_{key,query,conv}.weight [10240]`.

Global tensors: `token_embd.weight [2560, 248320]`,
`output.weight [2560, 248320]`, `output_hc_{norm,down,up}.weight`.

### 1.2 Layer order (from Strata)

The reference order is one block:

1. PLE at layer 1 only: build `key`, `query`, `value` from the gathered
   n-gram rows, compute the gate, the grouped norm, the causal conv over the
   history, and update the residual stack `R` before layer 1 runs.
2. Hyper-connection read (mixer): `hc_attn_read` on the residual stack `R`
   `[4, 2560]` produces the mixed activation and the per-stream injection.
3. Mixer: GDN or QSA on the mixed activation.
4. Hyper-connection write: combine the mixer output back into `R`.
5. Hyper-connection read (FFN): the same mixer with the `hc_ffn_*` weights.
6. MoE: router, 10 routed experts, one shared expert, combine.
7. Hyper-connection write: combine the FFN output back into `R`.

GDN order: `qkv = W @ x`; conv then SiLU on the whole 10240 wide output;
L2 norm on q and k only; `beta = sigmoid`; `gate = softplus(alpha + dt) *
ssm_a`; the gated-delta recurrence; `y = rms_norm(o, ssm_norm) * sigmoid(z)`;
`out = ssm_out @ y`. The q scale `1/sqrt(128)` is applied after the norm.

QSA order: project q (with gate), k, v; norm and rotate q and k; append k and
v to the paged KV cache; the indexer selects at most `top_k` cells; sparse
attention over the selected cells; output projection.

The PLE block is a host side hash (64 bit multiply and xor, `ngram_size = 3`,
`heads_per_ngram = 8`, so 16 rows per token) that indexes
`per_layer_token_embd.weight`, then a device side projection, gate, norm and
causal conv.

## 2. Current tessera state

The following is what exists in the tree today and can be reused.

- Backend abstraction, buffers, kernels, copies: done, both backends.
- GGUF loader: single file, v2 and v3, memory mapped. It does not read the
  `split.count` shards as one model.
- IQ3_S dequant and GEMM: `gemm_iq3s` and `gemm_iq3s_batched` exist on
  Vulkan and on ROCm, and the GGUF type map and the tile slice cover
  `IQ3_S`.
- MoE: routed experts plus one shared expert, with paging and per-expert
  staging. Used by Ornith-1.5.
- Hybrid SSM (GDN): conv1d step, `ssm_gate`, `delta_step_heads`, `l2norm`,
  `repeat_heads`, `rmsnorm_gated`. Used by Qwen3.5.
- Attention and mRoPE: GQA attention with all KV variants, `mrope`.
- MTP and DFlash2 speculation: done for the 27B targets.
- Architecture module pattern and registry: `src/models/qwen3_5/` and
  `src/models/registry.cpp`.

Present in the tree today: the `qwen4exp` module
(`src/models/qwen4exp/`) with the multi-shard load, the QSA and indexer,
the hyper-connection mixer, the PLE with its n-gram table reader, and the
registry entry. `CreateArchitecture("qwen4exp")` returns the module.

Still not present: an expert cache that keeps hot experts resident in VRAM
and streams the misses from system RAM. Without it decode is memory bound
on the expert reads. This is the M6 item below.

## 3. What the port needs

This is a new architecture module plus several new generic kernels. It is
not a change to one function. The work items are:

1. **Multi-shard GGUF load.** Open every `-NNNNN-of-NNNNN` file of one model,
   merge the tensor manifests, and map each tensor from the file that holds
   it. This is a prerequisite for both shards.
2. **qwen4exp config parse.** Read the metadata in section 1 into a config
   that carries the new fields (`hyper_connection`, `ple`, `indexer`,
   `compress_ratios`). `TransformerConfig` is generic today and has no place
   for them. The design choice is between extending `TransformerConfig` and
   giving the module its own side config. That choice needs the owner,
   because core must not name a model.
3. **Weight name map.** `MapWeightName` from `blk.N.<suffix>` to the internal
   names. The GGUF names are already `blk.N.*`, so this is mostly identity
   plus the value conversion for `ssm_a`.
4. **Hyper-connection mixer.** New generic kernels `hc_read` and `hc_write`
   (`hc_read`: grouped RMS norm on the 4 stream stack, low rank down and up,
   per-stream injection; `hc_write`: combine the block output and the
   injection back into the stack). Port `gr_read` and `gr_write` from
   `../Strata/src/kernels/gr.hpp` and the corresponding HIP and Vulkan
   kernels.
5. **QSA sparse attention.** New generic kernels: indexer key append and
   pool, indexer query projection and norm, block scores, top-k selection,
   sparse decode attention over the selected cells, and a paged KV layout
   with compression ratios. Port from `../Strata/src/kernels/qsa*.hpp` and
   `native_qsa*`.
6. **PLE.** New host n-gram hash and a streaming table reader that reads the
   16 rows of one token from `per_layer_token_embd.weight` and never
   uploads the whole 51 GB table. New device kernels for the key and value
   projection, the gate, the grouped norms and the causal conv. Port from
   `../Strata/src/kernels/ple.hpp`, `ngram.hpp` and
   `native_ple_postops`.
7. **MoE adaptation.** The existing MoE path fits the tensor set. The router
   is `ffn_gate_inp` plus `ffn_gate_inp_shexp`. Verify the 512 expert,
   top 10, width 640 shapes and the IQ3_S expert types.
8. **Expert cache and streaming.** See section 5. This is the part that
   decides the 48 tokens per second target.
9. **MTP.** A separate download. Out of scope for the first milestone.

## 4. Milestone plan

Each milestone lands with tests and an end to end check. Milestones run one
at a time, because each loads or runs a 27 GB or larger model.

- **M1: load.** Multi-shard GGUF open and merge; qwen4exp config parse;
  registry entry; `Model::ProbeConfig` succeeds on the real model; the PLE
  table is inspected but not uploaded. Verify with a header only test and
  with `ProbeConfig` on the model path. Memory light.
- **M2: GDN and MoE decode.** Layer assembly for a GDN layer plus the MoE
  block, with the hyper-connection mixer replaced by the identity and the
  PLE skipped. Compare the first-token logits against a Strata run for a
  short prompt. This proves the GDN and MoE port.
- **M3: hyper-connections.** Add `hc_read` and `hc_write`. Re-check the
  logits against Strata.
- **M4: QSA.** Add the indexer, the paged KV cache and the sparse attention.
  Re-check the logits. QSA is 12 of 48 layers but it is the attention path.
- **M5: PLE.** Add the n-gram hash, the streaming table reader and the PLE
  block. Re-check the logits. Stream a prompt and watch the whole output for
  garbage, per rule 19.
- **M6: expert cache and speed.** Add the RAM resident expert set and the
  streamed misses until decode reaches the target. Measure output tokens per
  second and record the number with the card and the context.
- **M7: MTP.** Add the draft head when its file is downloaded.

The non speculative path is the baseline. Speculation is off until M7.

## 5. The 48 tokens per second target

Strata's own published numbers show that the target is at or above the
measured ceiling on this class of card for this size.

- On a Radeon AI PRO R9700 (gfx1201, 32 GB), the Coder pack (experts stored
  like IQ3_S) measured 45.5 output tokens per second at a 4K prompt and 48.3
  at a 16 K prompt, with 29.7 GiB peak VRAM and a 12,288 slot expert cache.
- The community page records 70.9 to 77.3 output tokens per second for the
  full IQ3_S on a card with more headroom, with an expert cache of 8,649
  experts.

The model does not fit one 32 GB card: shard 1 is 54.8 GB and the PLE table
is another 28.8 GB. Strata fits it by keeping the experts in system RAM and
by caching the hot experts in VRAM. To reach 48 output tokens per second,
tessera must have:

1. an expert cache that holds several thousand expert slots in the 32 GB
   card, and
2. a batched or pipelined path that reads the missing experts from RAM over
   PCIe without stalling the token loop.

Without that, decode will be memory bound on the expert reads and will not
reach the target. This is the largest single unknown in the plan.

The target is a measurement, not a property. It must be measured on the
machine that runs the model, and recorded with the card, the context length
and the KV type.

## 6. Memory and machine safety

The workstation has 60 GB of RAM and two Radeon AI PRO R9700 cards (32 GB
each). Rule 20 applies with force:

- One heavy command at a time. Never build and load in one command.
- Only one card must be free. Leave a card that another owner holds alone.
- The IQ3_S model is 83.6 GB on disk. Loading it is a heavy step. Do not run
  a load beside a build or a Python step that copies a checkpoint.
- The PLE table must be read a row at a time. Never materialize it in host
  memory.
- Reference and diagnostic scripts stay memory light. Load one tensor at a
  time. Keep float32.

## 7. Open design questions

These need the project owner before M1 is written, because they touch the
architecture rules.

1. Where do the qwen4exp specific fields live? Extending
   `TransformerConfig` puts model specific fields in generic core. A side
   config keeps core clean but changes the `Architecture` interface.
2. Is the expert cache a new module under `src/core/`, or a capability of the
   model loader with a new backend call? The baby steps here define the
   interface for later architectures.
3. QSA has a paged KV cache with compression ratios. Does the existing KV
   cache abstraction grow a page table, or does QSA own its cache?

## 8. Status

- Reconnaissance: done. The architecture, the tensor inventory and the
  reference layer order are recorded above.
- M1 to M5 (load, GDN, MoE, hyper-connections, QSA, PLE): done. The module
  loads the two shards, decodes, and produces coherent output.
- M6 (expert cache and 48 tokens per second): in progress. The expert
  cache is now sized to the free device memory and the batched staging
  is pipelined (the staging ring), which lifts decode from about 3.7 to
  about 6 output tokens per second at a 4K context on one R9700 (gfx1201,
  32 GB, fp32 KV, `--gpu-layers 18`). The target is not reached. An
  intermittent non-finite logits fault remains (see the M6 investigation
  below).
- M7 (MTP): out of scope. No draft file is present.

Correctness evidence: with no completion cap and `--no-chat`, the prompt
`The capital of France is` produces a reasoning block and the answer
`The capital of France is **Paris**.` in 115 tokens. Every forward block
matches an independent NumPy reference on the real weights (the embedding,
the hyper-connection read and write, the GDN at token 0 and token 1, the
QSA score, attention and output, the PLE, the full MoE router and experts,
the head read and the output projection).

### Progress log

- **Reconnaissance.** Mapped the architecture from Strata: the tensor
  inventory, the layer order, the GDN, QSA, PLE and hyper-connection
  contracts, and the MoE shapes.
- **Quant types, dequant and GEMM (blocker cleared).** The IQ3_S pack is
  mixed precision. It uses three GGML tensor types the engine did not
  know, so the file did not even parse. Added the types end to end:
  - `IQ3_XXS` (ggml 18, 256 elements, 98 bytes)
  - `IQ2_S` (ggml 22, 256 elements, 82 bytes)
  - `Q2_0` (ggml 42, 64 elements, 18 bytes)
  Changes: the `DType` enum, `TensorBytes`, `BlockLayout`, the tensor
  slice geometry, the GGUF type map, the host dequantizers with their
  grids, the host GEMM references, the ROCm GEMV kernels, and the Vulkan
  GEMV shaders. Both backends register `gemm_iq3xxs`, `gemm_iq2s` and
  `gemm_q20`, and `GemmKernelName` maps the three dtypes to them.
- **Config parse.** Added the capability fields and their GGUF parse:
  `HyperConnectionParams`, `PleParams`, `IndexerParams` on
  `TransformerConfig`, parsed from `hyper_connection.*`, `ple.*` and
  `attention.indexer.*`. The generic parser already read the base model
  (attention, MoE, SSM, mRoPE).
- **Hyper-connection mixer (M3).** Ported the gated residual stack. The
  host reference is `HyperConnectionReadRef` and
  `HyperConnectionWriteRef` in `src/core/numerics/hyper.*`. The six
  built-in kernels are `gr_norm`, `gr_down`, `gr_gate`, `gr_mean`,
  `gr_inject` and `gr_write`, registered on both backends. The read path
  rounds the normalized streams and the bottleneck to bfloat16 (carried
  as f32), the reference contract.
- **Verified.** `tests/engine_test.cpp` builds a qwen4exp GGUF and checks
  every parsed field (`EngineTest.ParsesFlashNextConfig`).
  `tests/backend_test.cpp` checks the three dequantizers against
  hand-computed values, the three GEMV kernels against the host
  reference, and the hyper-connection read and write kernels against
  their host reference, on both backends. A header-only run against the
  real pack yields layers 48, hidden 2560, vocab 248320, experts 512/10,
  MoE width 640, hyper-connection 4/320, PLE layer [1], indexer
  4/128/2048. The ROCm build passes the suite (255 tests) and the Vulkan
  build passes (252 tests); 0 failures on either.

- **PLE n-gram hash (part of M5).** Ported the host hash that turns the
  last `ngram_size` token ids into the sixteen PLE table row indices:
  `NgramRows` and `NgramMixed` in `src/core/numerics/ngram.*`. Added the
  `ple.layer_multipliers` field and its GGUF parse. XOR, modulo, the
  forward end-of-text cut and the distinct missing-predecessor sentinel
  follow the reference.
- **PLE block post-ops (part of M5).** Ported the arithmetic after the
  key and value projections: `PlePostopsRef` in
  `src/core/numerics/ple.*`, and the kernels `rmsnorm_rows` (grouped RMS
  norm with a per-row gamma), `ple_gate`, `ple_broadcast`, `ple_conv`
  (dilated causal convolution plus the residual) and `ple_hist_advance`.
  Registered on both backends. The key and value projections reuse
  `gemm_bf16`.

- **QSA indexer scoring (part of M4).** Ported `QsaIndexRef` in
  `src/core/numerics/qsa.*` and the `qsa_index` kernel: per-cell scores
  as the per-head Relu of the pooled-key dot, summed over heads, plus
  the per-block bias and the incomplete-tail bias. Registered on both
  backends. QSA is GQA over a paged FP16 KV cache with an indexer that
  pools one key per `idx_block` cells; the selection, the paged append
  and the attention itself remain.
- **QSA paged append and gate (part of M4).** Ported `KvAppendRef` and
  `QsaGateApplyRef` and the `kv_append` and `qsa_gate_apply` kernels: one
  token's K and V into the paged FP16 cache, and the
  `attn * sigmoid(gate)` multiply with a single fp16 rounding.
- **QSA gather and attention (part of M4).** Ported `KvGatherRef` and
  `QsaAttendRef` and the `kv_gather` and `qsa_attend` kernels: the
  selected cells through the page table into a contiguous scratch, and
  the softmax GQA over them (the q -> kv head map is integer division).
  One block per query head; the softmax stages one score per id in shared
  memory (cap 4096, above the 2051-cell selection bound).
- **QSA top-k selection (part of M4).** Ported `QsaTopKRef` and the
  `qsa_topk` kernel: the largest `width` cells, ascending, ties by
  ascending index. One block; the threshold is found by a 32-bit greedy
  search on a monotonic float key, then thread 0 gathers in index order.
- **QSA indexer key append (completes M4).** Ported `IndexerKeyAppendRef`
  and the `indexer_key_append` kernel: the raw indexer key tail, the
  pooling that fires when a cell completes a block (double-precision
  mean and RMS norm), the NeoX rotation at the block's first cell, and
  the moving spare slot. With this, all QSA kernels exist on both
  backends. (The rotation position is the text case; the mRoPE section
  mapping for images is not wired.)
- **PLE table reader (M5c, part).** Added `PleTable` in
  `src/models/qwen4exp/ple_table.*`: it maps the second shard, locates
  `per_layer_token_embd.weight` (IQ4_NL, 160-value rows), and decodes the
  named rows with the existing IQ4_NL block dequant. The file is mapped,
  so a token's rows fault in only the pages they touch.
- **QSA step buffer.** Added `QsaStepFill`: the per-token
  pos / n_kv / n_bid / width values every QSA kernel reads, derived in one
  place so a caller cannot update one and forget another.
- **GDN output norm (part of M2).** Added the qwen4exp gated-delta output
  norm as `GdnOutNormRef` in `src/core/numerics/norm.{hpp,cpp}` plus the
  `gdn_out_norm` kernel on both backends. It differs from `rmsnorm_gated`
  in the gate activation: qwen4exp multiplies by `sigmoid(z)`, not the
  SiLU the Qwen3.5 path uses, so it is a separate kernel rather than a
  reuse. Covered by `BackendTest.GdnOutNormMatchesRef` (ROCm and Vulkan).
- **Cast kernel (part of M2).** Added `cast_f16_f32`, the inverse of the
  existing `cast_f32_f16`, on both backends. The QSA output gate writes an
  fp16 activation (the served contract); this cast lifts it back to fp32
  for the f32-activation GEMM.
- **Module trunk (M2, in progress).** `src/models/qwen4exp/internal.hpp`
  and `trunk.cpp` now hold `DeriveGeometry`, the projection helper, the
  hyper-connection read/write helpers and the GDN layer (project qkv / gate
  / alpha / beta, `ssm_gate`, the causal conv, the L2 norms, the modulo
  head expansion, the recurrence, the sigmoid output norm and the output
  projection). The QSA layer, the PLE block, the MoE, the state
  initialization and the public methods remain. The module is not yet in
  the build, so the tree stays green.
- **Module scaffolding (M2, in progress).** Added
  `src/models/qwen4exp/architecture.hpp` (the module class),
  `state.hpp` (the decode state) and `weights.cpp` (name, tensor-parallel
  roles). The layer loop (trunk.cpp) and the registry line are next. These
  files are not yet built, so the registered backends are unchanged.
- **Bug fixed: `Fp32ToHalfBits`.** The normal path shifted a 24-bit
  significand (with the implicit leading 1) and always took the carry
  branch, so it dropped the mantissa for every normal value. The function
  was dead code until the QSA reference called it; the host quantizer that
  builds test fixtures also uses it, so the four pinned hybrid baselines
  were regenerated with the corrected conversion. The Vulkan fp16 packer
  in `kv_append`/`qsa_gate_apply` had copied the same bug and was fixed
  with it.
- **Module landed and the model decodes (M1 to M5).** `src/models/qwen4exp/`
  is registered and runs the full 48 layer forward: the hyper-connection
  reads and writes, the GDN and QSA layers, the PLE block before layer 1,
  the routed plus shared MoE, and the output head. Both backends build and
  the suite passes. The forward matches an independent NumPy reference on
  the real weights at every block, and the model answers a factual prompt
  correctly.
- **Debug log level.** Added `log::Level::Debug` and
  `Diagnostics::SetMinLevel`, and moved the per-transfer `batched H2D
  upload` summaries to it so they are off by default. The CLI gains
  `--verbose` to enable them and `--quiet` now also silences info lines.
- **M6 investigation.** The per-token cost is the routed-expert re-upload.
  The dense weights stay resident (`PagedMoeDenseResident`), but the hot
  experts do not. The expert quant type varies per layer, so the expert
  slice size differs by layer and a shared cache must key by both kind and
  slice size. A global LRU cache was prototyped and scales the slot count,
  but its hit rate and throughput do not yet beat the per-(layer, kind)
  cache that is the default, so the prototype was reverted.
- **M6: the expert cache is sized to the free device memory.** The
  per-(layer, kind) store now holds `ExpertCacheSlots(free_bytes,
  total_per)` slices instead of a fixed 32
  (`src/core/loaders/shard.hpp`, `Model::ExpertCacheEntries`). On the
  R9700 at `--gpu-layers 18` the free device memory is about 7.9 GiB and
  the summed per-expert slice bytes about 61 MB, so each pair holds about
  100 slices, about 10,000 slices in total. That is close to the Strata
  12,288-slot reference. Measured at a 4K context, fp32 KV, one R9700:
  decode went from 271 ms/token (3.7 tok/s, 32 slices) to 158 to 166
  ms/token (about 6 tok/s), and the cache hit rate is about 74 percent.
  The expert staging per token fell from about 175 MB to about 106 MB.
  The gain is not yet robust: under external contention the host `memcpy`
  slows and a run can fall back to 271 ms/token, so the number is a best
  case, not a floor.
- **M6 open: the staging path dominates and its cost varies.** A profile
  splits the token into the expert staging and the kernels. The staging
  is a host `memcpy` from the system-RAM paged weights into the pinned
  window, then an H2D copy. In a fast run the staging is about 52
  ms/token and the kernels about 111 ms/token. The host `memcpy` alone
  is 13 to 93 ms per token and varies run to run. The `memcpy` reads a
  warm 31 GB buffer at 17 GB/s in isolation, so the variance is
  contention with the GPU and the ROCm runtime during decode, not the
  source layout or huge pages. The 48 tok/s target needs the staging
  overlapped with the kernels, a prefetch or pipelined path, and a fused
  MoE expert kernel.
- **M6: pipelined the batched staging (the staging ring).** The ROCm
  backend's batched upload now runs through a small ring of pinned
  windows instead of one buffer, so the host `memcpy` of the next window
  overlaps the GPU work of the current one instead of blocking on a
  per-window stream sync. A per-slot event guards reuse: the host waits
  only when it wraps to a slot whose upload may still be in flight. The
  ring bounds the pinned set to a few windows, so the source pages stay
  reclaimable. The decode gain is limited: the per-layer id readback
  (`CopyD2H`) drains the stream before the staging, so the staged `memcpy`
  cannot overlap the previous layer's kernels. Measured at a 4K context,
  fp32 KV, one R9700, `--gpu-layers 18`: the ring decodes at about 345
  ms/token (about 2 tok/s) versus the baseline's 326 ms/token (about 3
  tok/s) in the same contended window, so the ring is the pipelined path
  but the 48 tok/s target still needs the fused MoE expert kernel and/or
  removing the per-layer readback sync. The gain is masked by the
  intermittent non-finite logits fault (below), which stops roughly half
  the loads.
- **M6 open: intermittent non-finite logits.** About half of the model
  loads fail during prefill with `non-finite logits`, and the guard stops
  the run. The committed baseline at `97e7af6` fails the same way, so the
  fault predates the cache sizing. It is not the expert cache: the
  failure is nondeterministic across runs of one binary. The earlier
  finding was that buffered reads of the file returned different bytes on
  each read, so the system-RAM copy made at load can inherit those bytes.
  The trigger is the memory pressure during the load: the 31 GB paged
  copy plus the file's page cache exceeds the 60 GB host, which evicts
  and re-faults the file pages and returns stale bytes, so the RAM copy
  and the device upload inherit them. The fix is a read-path change: read
  the shards with a bounded pread (or direct I/O) instead of the whole-file
  mmap, so the load's peak memory drops from about 85 GB (the file's page
  cache plus the 31 GB paged copy) to about 54 GB (the read buffers plus
  the paged copy), which is under the 60 GB host and avoids the pressure
  that triggers the stale re-fault. It is not blocked; it is a larger
  change that is hard to confirm because the fault is intermittent.
- **Recommended sampling (qwen4exp).** The model card specifies a sampler
  per thinking mode: thinking on uses temperature 1.0 and top_p 0.95;
  thinking off uses temperature 0.7, top_p 0.80 and presence penalty 1.5;
  both use top_k 20, min_p 0.0 and repetition penalty 1.0. Added the
  per-architecture table to the CLI (`RecommendedSampling` in
  `tools/cli/cli_helpers.cpp`); it applies when the user passes no
  sampling flag. With thinking on and the recommended sampler the model
  answers `The capital of France is` with `The capital of France is
  **Paris**.` and stops, where greedy decode loops. The first generated
  token was still wrong at this point; the empty prefill logits entry
  below fixes it.
- **Fixed: empty prefill logits.** `Qwen4ExpArchitecture::ForwardBatch`
  wrote the last row's logits only when `all_logits` was true, so the
  prefill path (`all_logits == false`) returned an empty vector. The
  engine sampled the first token from that empty vector and `ArgMax` of an
  empty span is 0, so the first generated token was always vocab id 0
  (`!`) and the generation degraded from there. The last row now always
  reaches `logits_out`. With thinking on the model now produces a coherent
  reasoning block and the correct answer, and it stops.
- **Root-caused and fixed: the garbage output and the `!` loop.** The model
  garbled on both backends, and the same binary produced different output
  on identical runs, so the forward read different weights each run. The
  paged tensors (about 31 GB) were only mmap-backed. The decode loop
  re-read them from the file's page cache every token, and under memory
  pressure the page cache returned wrong bytes, so a corrupt weight
  reached the GPU. The file on disk is intact: a direct I/O sha256 of
  shard 1 equals the Hugging Face LFS hash, and direct reads are stable,
  while buffered reads returned different bytes on each read. The fix
  copies every paged tensor into a private system RAM buffer at load and
  drops the file mapping (`src/core/model.cpp`), so the decode loop reads
  stable bytes. Covered by `EngineTest.PagedWeightsAreIndependentOfTheFile`,
  which overwrites the source file after load and still generates the
  full-upload tokens.
- **Guard: the engine stops on non-finite logits.** Added
  `core::detail::AllFinite` and `core::detail::RejectNonFinite`, and the
  status code `StatusCode::NonFinite`. `DecodeLogits`, `PrefillTokens` and
  `Architecture::GreedyToken` reject a non-finite row, so a corrupt weight
  read stops the run with a clear diagnostic instead of looping on `!`.
  Covered by `EngineTest.RejectsNonFiniteLogits` and
  `EngineTest.StopsOnNonFiniteLogits`.

- **M6: warp-per-output IQ expert GEMMs (the column GEMV was the cost).**
  A decode profile (`TESSERA_PROFILE`) showed the routed-expert GEMVs at
  66 percent of GPU time, each one-thread-per-output with an uncoalesced
  K loop, measured at 10 to 15 GB/s. The IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL and
  Q2_0 dtypes now dispatch to a warp-per-output (`_vec`) GEMV on both
  backends: one 32-lane group per output, five shuffles, coalesced weight
  reads. On one R9700 (gfx1201), IQ3_S pack, `--gpu-layers 18`, 4K context,
  fp32 KV: decode fell from 141 to 84 ms/token (7 to 11 tok/s) and prefill
  from 6 to 11 prompt tok/s. The output stays coherent. The qwen4exp trunk
  also gained the decode phase profiler qwen3_5 already had.

- **M6: fused routed-expert kernels.** The per-expert loop copied each
  selected expert slice device to device and ran a separate GEMV, about
  90 launches and 30 copies per layer. New fused kernels compute the whole
  routed selection in two launches per layer: gate/up with the silu
  product and down with the weighted slot sum, each reading its slice
  straight from the staging tensor at `ids[slot]*stride`. They cover the
  five expert formats the pack uses (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up;
  IQ4_NL, Q2_0 down) on both backends. The host id readback now happens
  only for paged layers. On one R9700, 4K context, fp32 KV, `--gpu-layers
  18`: decode fell from about 84 to 61 to 76 ms/token (about 13 to 16
  tok/s). The output stays coherent.

- **M6: overlap the expert staging with the upload.** A per-layer host
  profile showed the MoE is 93 percent of the forward host time, and the
  expert staging (`EnsureLayerExperts`) is 1.5 to 2.2 ms per paged layer:
  the cache holds 117 of 512 slots per (layer, kind), so about 20 MB per
  layer is re-read from RAM. The staging ring used one 256 MB window, so the
  host memcpy of a layer's batch ran before its upload. The ROCm window is
  now 4 MB over an 8-slot ring, so a layer's batch splits into several
  windows and the memcpy of the next overlaps the asynchronous upload of
  the current one. On one R9700, 4K context, fp32 KV, `--gpu-layers 18`:
  decode fell from about 70 to 57 to 67 ms/token (about 15 to 17 tok/s).
  The Vulkan path keeps its single-window staging.

### Still required (in order)

 1. M6 follow-ups: add a fused MoE expert kernel and remove (or overlap)
    the per-layer id readback sync, then reach the 48 tok/s target. The
    cache is sized to the free device memory and the batched staging is
    pipelined (the staging ring).
 2. Fix the intermittent non-finite logits: read the shards with direct
    I/O (or a non-mapping read) so the system-RAM copy is stable. The
    fault is triggered by the load's memory pressure.
3. MTP (M7), when the draft file is downloaded.

### Memory note

The paged weights are held in system RAM, not the file mapping. The
Qwen3.8-Flash-Next paged set is about 31 GB and fits the 60 GB host. A
model whose paged set exceeds the host RAM will not load, so size
`--gpu-layers` to keep the paged set within the host budget.
