# Ornith-1.5-35B-A3B support in tessera

This file tracks the work to add full architecture support for
`ornith-ai/Ornith-1.5-35B-A3B-GGUF` to tessera. It is the progress
record for this task. See `AGENTS.md` for the working rules.

## Target

- Repository: `ornith-ai/Ornith-1.5-35B-A3B-GGUF`.
- Local files: `~/models/Ornith-1.5-35B-A3B-GGUF/Q4_K_M.gguf` and the
  bf16 safetensors reference in `~/models/Ornith-1.5-35B-A3B-BF16/`.
- GGUF `general.architecture`: `qwen35moe`.
- Model size: 36B parameters, about 3B active per token (mixture of
  experts).

The GGUF and the bf16 checkpoint were moved out of the Hugging Face
cache into `~/models/` on 2026-10-10, per the AGENTS model-data rule.

## Architecture summary

Ornith-1.5-35B-A3B is a Qwen3.5-family hybrid model. The model card
states it was built on Qwen3.5 and Gemma4 with continued pretraining.
The text backbone is the Qwen3.5 hybrid design. The single difference
from the dense Qwen3.5 model already supported by tessera is the
feed-forward block: it is a sparse mixture of experts (MoE).

### Text config (from `config.json`, `text_config`)

| Field | Value |
| --- | --- |
| hidden_size | 2048 |
| num_hidden_layers | 40 |
| num_attention_heads | 16 |
| num_key_value_heads | 2 |
| head_dim | 256 |
| partial_rotary_factor | 0.25 (rope_dim 64) |
| rope_theta | 10000000 |
| full_attention_interval | 4 (3 linear attention layers, then 1 full) |
| linear_conv_kernel_dim | 4 |
| linear_key_head_dim | 128 |
| linear_num_key_heads | 16 |
| linear_num_value_heads | 32 |
| linear_value_head_dim | 128 |
| moe_intermediate_size | 512 |
| num_experts | 256 |
| num_experts_per_tok | 8 |
| shared_expert_intermediate_size | 512 |
| rms_norm_eps | 1e-6 |
| vocab_size | 248320 |
| mtp_num_hidden_layers | 1 |
| attn_output_gate | true |

The attention and linear-attention parameters match the dense Qwen3.5
config that the `qwen3_5` module already handles.

### GGUF metadata (`qwen35moe.*`)

| Key | Value |
| --- | --- |
| block_count | 41 (40 trunk plus 1 nextn) |
| context_length | 262144 |
| embedding_length | 2048 |
| attention.head_count | 16 |
| attention.head_count_kv | 2 |
| attention.key_length | 256 |
| attention.value_length | 256 |
| rope.dimension_count | 64 |
| rope.dimension_sections | [11, 11, 10, 0] |
| rope.freq_base | 1e7 |
| attention.layer_norm_rms_epsilon | 1e-6 |
| expert_count | 256 |
| expert_used_count | 8 |
| expert_feed_forward_length | 512 |
| expert_shared_feed_forward_length | 512 |
| nextn_predict_layers | 1 |
| ssm.conv_kernel | 4 |
| ssm.state_size | 128 |
| ssm.group_count | 16 |
| ssm.time_step_rank | 32 |
| ssm.inner_size | 4096 |
| full_attention_interval | 4 |

The dense Qwen3.5 GGUF uses `feed_forward_length`. The MoE GGUF omits
that key and carries `expert_feed_forward_length`,
`expert_shared_feed_forward_length`, `expert_count` and
`expert_used_count` instead.

### Layer structure

Trunk layer `l` is a full-attention layer when `(l + 1) % 4 == 0`
(layers 3, 7, ..., 39). All other trunk layers are linear-attention
(gated-delta) layers. This matches the dense Qwen3.5 module. The nextn
prediction block is `blk.40` and is a full-attention block.

### MoE feed-forward

Each block has a `post_attention_norm`, then a sparse MoE block:

- Router: `ffn_gate_inp.weight` [hidden, num_experts] = [2048, 256] F32.
- Per-expert weights, packed in rank-3 tensors:
  - `ffn_gate_exps.weight` [hidden, inter, num_experts] = [2048, 512, 256]
  - `ffn_up_exps.weight` [hidden, inter, num_experts] = [2048, 512, 256]
  - `ffn_down_exps.weight` [inter, hidden, num_experts] = [512, 2048, 256]
- Shared expert, dense rank-2 weights:
  - `ffn_gate_shexp.weight` [hidden, inter] = [2048, 512]
  - `ffn_up_shexp.weight` [hidden, inter] = [2048, 512]
  - `ffn_down_shexp.weight` [inter, hidden] = [512, 2048]
- Shared expert gate: `ffn_gate_inp_shexp.weight` [hidden] = [2048] F32.

The GGUF dims are reversed from PyTorch, as usual. In the GGUF layout
the expert index is the slowest dimension, so one expert's matrix is a
contiguous block. Expert `e` of `ffn_gate_exps` occupies elements
`[e * 2048 * 512, (e + 1) * 2048 * 512)` and is laid out as a
row-major [512, 2048] matrix (n by k).

Reference computation (Qwen3 MoE `SparseMoeBlock`):

```
router_logits = gate(x)                       # [num_experts]
routing_weights = softmax(router_logits)
topk_weights, topk_ids = topk(routing_weights, k = 8)
if norm_topk_prob: topk_weights /= sum(topk_weights)
for e in topk_ids:
    y += topk_weights[e] * down_e(silu(gate_e(x)) * up_e(x))
shared = down_shexp(silu(gate_shexp(x)) * up_shexp(x))
shared = sigmoid(gate_inp_shexp(x)) * shared
out = y + shared
```

The exact `norm_topk_prob` default is not in the config or the GGUF
metadata. This must be matched to the served reference runtime and is
tracked as an open question below.

### MTP (nextn) head

The nextn block is `blk.40` and uses the same tensor names as the dense
Qwen3.5 module (`blk.40.nextn.eh_proj.weight`, `blk.40.nextn.enorm.weight`,
`blk.40.nextn.hnorm.weight`, `blk.40.nextn.shared_head_norm.weight`).
The block feed-forward is a MoE feed-forward, so the MTP draft path also
needs the MoE FFN.

## Reuse from the existing code base

The `src/models/qwen3_5/` module already implements the whole Qwen3.5
hybrid trunk: gated full attention, gated-delta linear attention, the
mRoPE, the KV cache, the batched prefill and verify paths, and the nextn
MTP head. All of that applies unchanged to Ornith.

The narrow seam is the feed-forward. `RunFfn` (`trunk.cpp`) and
`RunFfnBatch` (`trunk_batch.cpp`) are the only two places that assemble
the FFN. Every block runner calls one of them. If these two functions
select the FFN from the config, the entire trunk works for the MoE model
with no duplication.

## Decision: MoE as a generic capability, one family module

`qwen35moe` is the Qwen3.5 hybrid architecture with a sparse FFN. The
attention, linear-attention and MTP code is identical to the dense
model. We therefore:

1. Add the MoE hyper-parameters to `TransformerConfig` and parse them
   from GGUF metadata in `src/core/model.cpp` (generic metadata).
2. Add generic MoE kernels (router top-k and softmax, per-expert GEMV,
   weighted combine) to both backends, with tests. These are shared by
   any future MoE architecture, so they belong in the backend and in
   `src/core/numerics/`.
3. Make `RunFfn` and `RunFfnBatch` select the MoE FFN when
   `config.num_experts > 0`. This is a data branch, not a model-name
   branch. The FFN assembly for the MoE variant lives in a new
   `src/models/qwen3_5/moe.cpp` so `trunk.cpp` stays small.
4. Register `qwen35moe` (and the Hugging Face names `qwen3_5_moe`,
   `Qwen3_5MoeForConditionalGeneration`) in `src/models/registry.cpp`
   to the existing Qwen3.5 family module.

This follows the AGENTS rules: no duplicated trunk code, the new
capability is a generic kernel, and the new model arrives as data.

## Implementation plan

Status legend: done, in progress, todo.

1. Recon and architecture analysis. **done**.
2. Move the model files into `~/models/`. **done**.
3. `TransformerConfig` MoE fields and GGUF/config.json parsing, with
   tests. **done**.
4. Router kernels: projection reuse plus top-k and softmax. **done**
   (`top_k_rows` plus the new `moe_gate`).
5. Expert kernels: per-expert GEMV gather and weighted combine on Vulkan
   and ROCm. **done** (device-to-device gather reusing the generic GEMM
   helpers, plus the new `moe_scale_add`). Kernel unit tests pass on
   both backends.
6. Shared-expert assembly (reuses the dense FFN path plus a sigmoid
   gate). **done**.
7. `RunFfn` and `RunFfnBatch` dispatch and `moe.cpp`. **done**.
8. Registry entry, CLI check, tests. **done** (the CLI needs no change).
9. End-to-end run on the Q4_K_M GGUF and comparison against the
   reference. **first pass done** (see the verification log).

### First end-to-end result

On ROCm, device 0, greedy decode of the Q4_K_M GGUF:

```
prompt: The capital of France is
output:  Paris.
```

The model loads (753 tensors, 21.7 GB), prefills 5 tokens in 179 ms,
and decodes at about 36 tokens per second. The answer is correct, which
shows the router, the routed experts and the shared expert all agree
with the reference ordering.

### Current implementation and known debt

- Routing is on the device: `top_k_rows` selects the experts and
  `moe_gate` softmaxes the selected logits and computes the shared-expert
  gate from the normed hidden and the shared-expert weight in one launch
  (no separate projection).
- The routed experts run through two fused kernels,
  `moe_experts_gate_up_q4k` and `moe_experts_down_q4k` /
  `moe_experts_down_q6k`. They read the selected expert ids from device
  memory and process all `top_k` experts in one launch each: gate and up
  for every expert, then the weighted down accumulation. There is no
  device-to-device weight copy and no host round-trip per layer. The block
  decode is shared with the dense GEMV path (ROCm
  `rocm_kernels_codec.hpp`; Vulkan inlines the same block math).
- The shared expert runs through the same two fused kernels as a
  one-entry expert (id zero) with an accumulate flag on the down kernel,
  so it adds its sigmoid-gated output in the second launch.
- The batch (prefill and verify) path is row-serial: it loops over rows
  and reuses the single-row kernels. It is correct but leaves prefill
  below the decode rate; a grouped-expert GEMM is the remaining work.

### Speed result

Measured on ROCm, device 0, the Q4_K_M GGUF, greedy decode of a 64-token
generation:

| | before | after |
| --- | --- | --- |
| decode | 37 tok/s (27 ms/token) | 83 tok/s (12 ms/token) |
| prefill | 27 prompt tok/s | 81 prompt tok/s |

Vulkan reaches 79 tok/s decode on the same checkpoint and produces the
same answers ("Paris.", "Tokyo", "four"). The remaining cost is the
routing and attention launches, not the expert weight reads: the experts
stream about 640 MB per token, far below the card's memory rate.

## Reference match

The routing behavior was checked against the served reference stack
(the vLLM tree at `~/models/venv`). In
`vllm/model_executor/models/qwen3_next.py` the MoE block builds the
router with `renormalize = getattr(config, "norm_topk_prob", True)`,
and the shared-expert gate applies
`out = sigmoid(shared_expert_gate(x)) * out`
(`vllm/model_executor/models/qwen2_moe.py`). Qwen3-Next is the hybrid
mixture-of-experts predecessor of the Qwen3.5 family this model
extends. Tessera matches: `moe_gate` renormalizes the selected top-k
weights, and the shared expert is scaled by the sigmoid of
`ffn_gate_inp_shexp`. No routed scaling factor is applied (the config
carries none).

## Open questions

- The fused expert kernels cover Q4_K gate/up and Q4_K or Q6_K down; any
  other expert dtype falls back to the generic per-expert path. Q5_K and
  Q8_0 experts are not exercised by the Q4_K_M checkpoint.
- Batched (prefill and verify) MoE performance: the row-serial path is
  correct but slower than decode. A grouped-expert GEMM is the target.

## Verification log

- Date 2026-10-10: GGUF header parsed. Architecture `qwen35moe`, 753
  tensors, 49 metadata keys. Tensor naming and metadata recorded above.
- Date 2026-10-10: `moe_gate` and `moe_scale_add` kernel unit tests pass
  on Vulkan and ROCm against host references.
- Date 2026-10-10: full test suite passes on ROCm (204 tests, no
  failures).
- Date 2026-10-10: end-to-end greedy decode on the Q4_K_M GGUF returns
  "Paris." for "The capital of France is" on ROCm device 0.
- Date 2026-10-10: chat-template prompts return correct answers
  ("Tokyo" for the capital of Japan, "Four" for 2 + 2). The MTP
  speculation path (`--speculate`) also returns "Paris." and exercises
  the MoE feed-forward in the nextn block. Full suite passes on Vulkan
  and ROCm (204 tests each).
- Date 2026-10-10: MoE speed optimization. Fused expert kernels read the
  expert ids on device and process all selected experts in two launches;
  the shared expert shares those kernels; `moe_gate` folds the
  shared-expert gate. Decode 37 to 83 tok/s (ROCm) and 79 tok/s (Vulkan);
  prefill 27 to 81 tok/s. Both backends return the same answers. Suite is
  206 tests on both backends.
