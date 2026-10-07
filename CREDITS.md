# Credits

External code and ideas used in tessera. Each entry names the
project, the version, the license, and where it is used.

- llama.cpp (MIT License, local checkout at d0b490f25): block
  layouts and dequantization math for Q3_K, Q5_K, Q6_K, IQ4_NL,
  IQ4_XS, IQ3_S (ggml-common.h structures, ggml-quants.c dequantize
  functions), the IQ4 codebook (kvalues_iq4nl), the IQ3_S grid
  (iq3s_grid), and the ggml type ids 20, 21, 23. Ported (not
  compiled) to src/core/numerics/quant.cpp, the vulkan .comp
  kernels, and the rocm kernels in src/backends/rocm.
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
