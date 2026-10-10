# Credits

External code and ideas used in tessera. Each entry names the
project, the version, the license, and where it is used.

- llama.cpp (MIT License, local checkout at d0b490f25): block
  layouts and dequantization math for Q3_K, Q5_K, Q6_K, IQ4_NL,
  IQ4_XS, IQ3_S (ggml-common.h structures, ggml-quants.c dequantize
  functions), the IQ4 codebook (kvalues_iq4nl), the IQ3_S grid
  (iq3s_grid), and the ggml type ids 20, 21, 23. Ported (not
  compiled) to src/core/numerics/quant.cpp, the vulkan .comp
  kernels, and the rocm kernels in src/backends/rocm. Q8_0 follows
  the public GGUF block layout (fp16 scale plus 32 signed bytes).
- llama.cpp src/models/qwen35.cpp (MIT License, same checkout):
  hybrid layer layout (recurrent gated-delta-net layers interleaved
  with full attention every full_attention_interval blocks, MTP
  blocks excluded from the main pass), fused Q-plus-gate and fused
  qkv tensor shapes, and the ssm.* metadata keys. Referenced for
  the hybrid config in include/tessera/model.hpp and
  src/core/model.cpp; no code copied.
- Gated Delta Networks (arXiv 2412.06464, Yang et al.) and Gated
  Attention for LLMs (NeurIPS 2025, Qiu et al.), plus the Qwen3-Next
  architecture blog (qwen.ai): the gated delta recurrence, the SDPA
  output sigmoid gate, and the 3:1 hybrid layout. Design references
  for src/core/numerics/norm.* and the hybrid roadmap; no code
  copied.
- vLLM gated-delta attention (Apache-2.0,
  vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py and
  vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py):
  the interleaved GQA repeat map (value head h uses key head
  h % num_k_heads) and the 1/sqrt(head_k_dim) query scale applied inside
  the recurrence. Design reference for RepeatHeadsRef and the l2norm
  scale; no code copied.
- Multimodal RoPE (Qwen2-VL, arXiv 2409.12191; vLLM
  MRotaryEmbedding docs): three-section pair layout with a global
  frequency index, text rows equivalent to 1D RoPE. Design reference
  for the mrope built-in; no code copied.
- Hugging Face transformers Qwen3-Next modeling (Apache-2.0,
  src/transformers/models/qwen3_next/modeling_qwen3_next.py): the
  gated-attention q_proj layout (view to [heads, 2*head_dim], chunk
  per head into query then gate) and the sigmoid gate on the SDPA
  output. Design reference for the qgate_split built-in; no code
  copied.
- GPT-2 byte-level BPE tokenizer (OpenAI, MIT) and the Qwen
  pre-tokenization pattern (HF tokenizer config, Apache-2.0): the
  bytes-to-unicode map, merge-by-rank BPE, and the split regex
  `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|
  \s*[\r\n]+|\s+(?!\S)|\s+`. Reimplemented in src/core/tokenizer.cpp;
  no code copied.
- vLLM Qwen3.5 MTP implementation (Apache-2.0,
  vllm/model_executor/models/qwen3_5_mtp.py): the MTP head layout
  (RMSNorm the token embedding and the backbone hidden, concat,
  nextn.eh_proj to one hidden vector, one full-attention block,
  shared head norm and the shared output weight). Design reference
  for src/core/decode_mtp.cpp; no code copied.
- vLLM DFlash2 draft model (Apache-2.0,
  vllm/model_executor/models/qwen3_dflash2.py): the grouped dynamic
  convolution (per-tap base kernel plus a per-token per-group offset
  from kernel_projection, reset every block_size positions). Design
  reference for the dflash_conv built-in; no code copied.
- vLLM DFlash2 candidate selector (_score_edges / CandidateSelector,
  qwen3_dflash2.py): the low-rank predecessor/successor
  transition score that re-ranks the unary top-K. Design
  reference for the selector_edge_score built-in.
- vLLM DFlash sliding attention (_resolve_layer_attention,
  qwen3_dflash.py): the causal sliding-window mask used by
  the DFlash2 draft layers. Design reference for the
  attention window parameter.
- Strata (a hybrid CPU and GPU inference engine, no version
  recorded) tools/calibrate.py, setup.py and tools/test_calibrate.py:
  the calibration approach. Strata measures machine dependent
  settings with a fixed candidate sweep, keeps a value only when it
  beats the default by a noise floor (MIN_GAIN 0.03), confirms the
  winner with an interleaved re-measurement, and stores the result
  per machine and model keyed by a hardware key. Reimplemented in
  src/core/calibrate/calibrate.cpp and tools/cli/calibrate_command.cpp;
  no code copied.
