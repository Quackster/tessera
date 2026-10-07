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
