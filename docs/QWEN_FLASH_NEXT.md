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

Not present in the tree: no `qwen4exp` module, no multi-shard GGUF merge, no
QSA, no indexer, no hyper-connection mixer, no PLE, no n-gram table loader,
and no expert cache that streams experts from system RAM to a 32 GB GPU.

`CreateArchitecture("qwen4exp")` returns `nullptr` today. The README model
table describes the loader as partial. The loader code for the split shards
and the n-gram stream is not in the tree.

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
- 48 tokens per second: not measured, because the model does not decode
  yet.

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

### Still required (in order)

1. The qwen4exp architecture module and registry entry. The module cannot
   land before its decode works, because the `Architecture` interface has
   no partial form; the generic device decode rejects hybrid configs.
2. Layer assembly: GDN and MoE blocks wrapped by the hyper-connection
   mixer, and the QSA block (projections, q/k norm, RoPE, the kernels
   above, the output projection).
3. The PLE key and value projections in the block (the reader and the
   post-ops are done; the projections reuse `gemm_bf16`).
4. The expert cache and streaming path, then the measurement.
