#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/log.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Buffer;    // forward declaration (defined below)
class Backend;   // forward declaration (defined below)
class Kernel;    // forward declaration (defined below)

// Where a buffer's memory lives.
enum class MemoryKind : int {
  Device = 0,     // accelerator memory
  HostVisible = 1,  // pinned host memory, mappable by the accelerator
};

// Kernel argument binding limits: the layout both backends expose to a
// kernel (descriptor bindings and scalar/push-constant arguments).
constexpr std::size_t kMaxBoundBuffers = 16;
constexpr std::size_t kMaxScalars = 16;

// One kernel launch request. grid_* is the workgroup count per axis on
// both backends. block_* is the workgroup size: honored on rocm
// (capped at the device limit); on vulkan the workgroup size is fixed
// when the kernel is compiled, so block_* is ignored there.
// Arguments bind in kernel parameter order: device buffers first
// (parameters 0..N-1), then 64-bit scalar values.
struct KernelLaunch {
  std::uint32_t grid_x = 1;
  std::uint32_t grid_y = 1;
  std::uint32_t grid_z = 1;
  std::uint32_t block_x = 256;
  std::uint32_t block_y = 1;
  std::uint32_t block_z = 1;
  // Bound buffers, in kernel parameter order (buffers before scalars).
  std::vector<const Buffer*> buffers;
  // 64-bit scalar arguments, in kernel parameter order.
  std::vector<std::uint64_t> scalars;
};

// RAII accelerator buffer allocated through a Backend.
// The handle is opaque to the core; only the allocating backend knows its
// meaning (a VkDeviceMemory record, a hipMalloc pointer, ...).
class Buffer {
 public:
  Buffer() = default;
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  ~Buffer();
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  [[nodiscard]] std::size_t Size() const;
  [[nodiscard]] MemoryKind Kind() const;
  // Opaque accelerator handle; pass to this buffer's backend only.
  [[nodiscard]] void* Handle() const;
  // Valid when Kind() == MemoryKind::HostVisible.
  [[nodiscard]] void* HostMap() const;
  // The backend that allocated this buffer.
  [[nodiscard]] Backend& Owner() const;

   private:
  friend class Backend;
  Buffer(Backend& backend, void* handle, std::size_t size, MemoryKind kind,
         void* host_map);
  void FreeIfOwned() noexcept;
  Backend* backend_ = nullptr;
  void* handle_ = nullptr;
  std::size_t size_ = 0;
  MemoryKind kind_ = MemoryKind::Device;
  void* host_map_ = nullptr;
};

// RAII handle for a compiled kernel loaded through Backend::LoadKernel.
// Destroying the Kernel releases the compiled artifact (a VkPipeline on
// vulkan; nothing for rocm built-ins).
//
// Usage:
//   auto kernel = backend.LoadKernel("fill", {});
//   if (!kernel) return 1;
//   KernelLaunch launch;  // buffers + scalars, in parameter order
//   auto result = backend.LaunchKernel(*kernel, launch);
class Kernel {
 public:
  Kernel() = default;
  Kernel(Kernel&& other) noexcept;
  Kernel& operator=(Kernel&& other) noexcept;
  ~Kernel();
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  // The name used at load time ("fill", ...).
  [[nodiscard]] std::string_view Id() const;
  // The backend that loaded this kernel.
  [[nodiscard]] Backend& Owner() const;
  // Opaque accelerator handle; pass to this kernel's backend only.
  [[nodiscard]] void* Handle() const;

 private:
  friend class Backend;
  Kernel(Backend& backend, void* handle, std::string id);
  void FreeIfOwned() noexcept;
  Backend* backend_ = nullptr;
  void* handle_ = nullptr;
  std::string id_;
};

// Shared launch validation: grid/block axes must be nonzero and the
// argument counts must fit the contract limits.
[[nodiscard]] inline StatusCode ValidateLaunch(const KernelLaunch& launch) {
  if (launch.grid_x == 0 || launch.grid_y == 0 || launch.grid_z == 0 ||
      launch.block_x == 0 || launch.block_y == 0 || launch.block_z == 0) {
    return StatusCode::InvalidArgument;
  }
  if (launch.buffers.size() > kMaxBoundBuffers ||
      launch.scalars.size() > kMaxScalars) {
    return StatusCode::InvalidArgument;
  }
  return StatusCode::Ok;
}

// Built-in kernel argument contract, identical on every backend.
// "fill": buffer 0 is an int32 array; scalar 0 is the value, scalar 1
// the element count.
// "gemm_q4k": buffer 0 is the activation A (fp32, m x k), buffer 1 the
// quantized weights W (Q4_K, n x k rows of k/256 blocks), buffer 2 the
// output C (fp32, m x n); scalars are m, n, k (k a positive multiple of
// kQ4KBlockElements). The dispatch is ceil(m * n / 256) workgroups of
// 256. Every quantized GEMM built-in shares this m, n, k order.
// "gemm_q4k_row": the same buffers, scalars and result as gemm_q4k, but
// one workgroup per output element (thread `t` owns element `b * 256 + t`
// of every block, partials reduced in shared memory). The reads coalesce
// across the workgroup. The dispatch is m * n workgroups of 256.
// "rope": buffer 0 holds rows x heads x head_dim fp32 rotated in place
// (NeoX pairing over rope_dim per head); scalars are rows, heads,
// head_dim, rope_dim, pos_base, and the fp32 theta bits. head_dim and
// rope_dim are even, rope_dim <= head_dim. The dispatch is
// ceil(rows * heads * (rope_dim / 2) / 256) workgroups of 256.
// "attention": buffers are q (m x heads*head_dim fp32), k and v
// (n x kv_heads*head_dim each), out (m x heads*head_dim); scalars are
// m, n, heads, kv_heads, head_dim, q_base, window (0 = full causal;
// a nonzero window keeps only the recent window keys), kv_f16 (0 fp32
// keys/values, 1 fp16) and causal (0 attends all n keys, 1 the causal
// prefix; the CLIP vision encoder is non-causal). Detail:
// m, n, heads, kv_heads, head_dim, q_base. Query row i sits at
// position q_base + i and attends keys 0..pos (clamped to n - 1) with
// scale 1/sqrt(head_dim); head h reads kv head h / (heads/kv_heads).
// head_dim must be at most 256. The dispatch is m * heads workgroups of
// 256 (one workgroup per query row and head).
// "gemm_f32": buffer 0 is A (fp32, m x k), buffer 1 the fp32
// weights W (n x k), buffer 2 the output C (fp32, m x n); scalars are
// m, n, k. The dispatch is ceil(m * n / 256) workgroups of 256.
// "gemm_fp8_block": buffer 0 is A (fp32, m x k), buffer 1 the FP8
// E4M3 weights W (n x k bytes), buffer 2 the (n/128) x (k/128) fp32
// block scales, buffer 3 the output C (fp32, m x n); scalars are m, n, k
// with n and k multiples of 128. The dispatch is ceil(m * n / 256).
// "gemm_bf16": buffer 0 is A (fp32, m x k), buffer 1 the bf16
// weights W (n x k, 2 bytes each), buffer 2 the output C (fp32, m x n);
// scalars are m, n, k. The dispatch is ceil(m * n / 256) workgroups of
// 256.
// "gemm_fp8": buffer 0 is A (fp32, m x k), buffer 1 the FP8 E4M3
// weights W (n x k bytes), buffer 2 one fp32 scale per row of W,
// buffer 3 the output C (fp32, m x n); scalars are m, n, k. The
// dispatch is ceil(m * n / 256) workgroups of 256.
// "gemm_mxfp4": buffer 0 is A (fp32, m x k), buffer 1 the packed MXFP4
// weights W (n x k/2 blob bytes, two E2M1 nibbles per byte, followed by
// the n x k/32 E8M0 scale bytes at offset n*k/2), buffer 2 the output C
// (fp32, m x n); scalars are m, n, k with k a positive multiple of 32.
// One 32-lane warp computes one output element (lane l sums blocks
// l, l+32, ... for coalesced reads, then a shuffle reduces): the dispatch
// is ceil(m * n / 8) workgroups of 256.
// "gemm_q5k", "gemm_q6k", "gemm_q3k", "gemm_iq4nl", "gemm_iq4xs",
// "gemm_iq3s", "gemm_q80": buffer 0 is A (fp32, m x k), buffer 1 the
// quantized weights W, buffer 2 the output C (fp32, m x n); scalars
// are m, n, k with k a positive multiple of 256 (32 for iq4nl and
// q80). The dispatch is ceil(m * n / 256) workgroups of 256.
// "gemm_q4k_batched": tiled C = A x dequant(W)^T for a batch. Buffers
// and scalars match gemm_q4k (m, n, k, k a multiple of 256). One
// workgroup handles 8 activation rows x one weight column; the Q4_K
// block is dequantized once into shared memory and reused across the
// 8 rows. The dispatch is ceil(m / 8) * n workgroups of 256.
// "rmsnorm": buffer 0 is X (fp32, rows x cols), buffer 1 the weight W
// (fp32, cols), buffer 2 the output Y (fp32, rows x cols); scalars
// are rows, cols, and the fp32 epsilon bits. Y = X / sqrt(mean(X^2) +
// eps) * W per row with sequential accumulation. The dispatch is
// ceil(rows * cols / 256) workgroups of 256.
// "sigmoid_gate": buffers are A and G (fp32, n each) and the output O
// (fp32, n); scalar 0 is n. O = A * sigmoid(G) elementwise. The
// dispatch is ceil(n / 256) workgroups of 256.
// "l2norm": buffer 0 is X (fp32, rows x cols), buffer 1 the output Y
// (fp32, rows x cols); scalars are rows, cols, the fp32 epsilon bits
// and the fp32 scale bits. Y = scale * X / sqrt(sum(X^2) + eps) per
// row. The dispatch is ceil(rows / 256) workgroups of 256.
// "rmsnorm_gated": buffer 0 is X (fp32, rows x cols), buffer 1 the
// weight W (fp32, cols), buffer 2 the gate (fp32, rows x cols),
// buffer 3 the output Y (fp32, rows x cols); scalars are rows, cols, the
// fp32 epsilon bits, and the element bases gbase/ybase the gate and output
// are read and written from (zero for the plain rows x cols case, non-zero
// to address one token's slice of a batched buffer without a copy).
// Y = X / sqrt(mean(X^2) + eps) * W * silu(gate) per row. The dispatch is
// ceil(rows / 256) workgroups of 256.
// "ssm_gate": buffers are a_log, dt (fp32, heads each), alpha_raw,
// beta_raw, alpha and beta (fp32, rows x heads); scalars are heads and
// rows. alpha = exp(-exp(a_log) * softplus(alpha_raw + dt)), beta =
// sigmoid(beta_raw). The dispatch is ceil(rows * heads / 256) workgroups
// of 256.
// "delta_step_heads": buffer 0 is S (fp32, heads x dk x dv), buffers 1..4 are
// k (rows x heads x dk), v (rows x heads x dv), q (rows x heads x dk) and o
// (rows x heads x dv), buffers 5 and 6 are alpha and beta (rows x heads);
// scalars are heads, dk, dv, rows, sbase, sstride, abase. The rows gated-delta steps
// run in turn, advancing the shared state: each step reads the state at
// sbase + head offset and writes it at sbase + sstride + head offset, so a
// zero stride updates in place and a non-zero one appends every step to the
// next history slot. The dispatch is ceil(heads * dv / 256) workgroups of 256.
// "repeat_heads": buffer 0 is IN (fp32, rows x num_k_heads x head_k_dim),
// buffer 1 the output OUT (fp32, rows x num_v_heads x head_k_dim);
// scalars are num_v_heads, head_k_dim, factor and rows (num_v_heads is
// divisible by factor; out[h] = in[h % (num_v_heads / factor)], the
// interleaved GQA layout). The dispatch is
// ceil(rows * num_v_heads * head_k_dim / 256) workgroups of 256.
// "add": buffers 0 and 1 are A and B (fp32, n each) and buffer 2 the
// output O (fp32, n); scalar 0 is n. O = A + B elementwise.
// "silu_mul": buffers are G and U (fp32, n each) and the output O
// (fp32, n); scalar 0 is n. O = silu(G) * U elementwise. The dispatch
// for both is ceil(n / 256) workgroups of 256.
// "embedding_f32", "embedding_bf16", "embedding_q4k": buffer 0 is the
// token ids (u32, rows), buffer 1 the embedding table (vocab x cols:
// fp32, bf16 or Q4_K), buffer 2 the fp32 output (rows x cols); scalars
// are rows, cols and vocab. Out[row] = table[ids[row]], with a zero row
// for an out-of-range id. cols must be a multiple of 256 for Q4_K. The
// dispatch is ceil(rows * cols / 256) workgroups of 256.
// "spatial_merge": buffer 0 In (tokens x embed, grid_h x grid_w), buffer 1
// Out ((grid_h/merge)*(grid_w/merge) x merge*merge*embed); scalars embed,
// grid_w, grid_h and merge. The dispatch is ceil(out_elems / 256).
// "image_patchify": buffer 0 In (image [h, w, 3] fp32), buffer 1 Mean
// (3), buffer 2 Std (3), buffer 3 Out (patches x 3*patch*patch); scalars
// h, w and patch. The dispatch is ceil(out_elems / 256).
// "layernorm": buffers X (fp32, rows x cols), weight (fp32, cols), bias
// (fp32, cols) and the output (fp32, rows x cols); scalars rows, cols and
// eps (fp32 bits). One thread per row.
// "bias_add": buffers X (fp32, rows x cols), B (fp32, cols, broadcast
// per row) and the output (fp32, rows x cols); scalars rows and cols. The
// dispatch is ceil(rows*cols / 256).
// "gelu": buffer 0 X (fp32, n), buffer 1 the output (fp32, n); scalar n.
// Tanh approximation. The dispatch is ceil(n / 256).
// "quantize_q8": buffer 0 is In (fp32, rows x cols), buffer 1 the packed
// int8 output (rows x cols bytes), buffer 2 one fp32 scale per row;
// scalars are rows and cols (cols a multiple of 4). The dispatch is one
// thread per row.
// "attention_q8": like attention but keys and values are symmetric int8
// (buffer 1 K, buffer 2 V), with one fp32 scale per key row (buffer 3 for
// K, buffer 4 for V) and the fp32 output (buffer 5); scalars are m, n,
// heads, kv_heads, head_dim, q_base, window and kind. One workgroup per
// (query row, head) runs the tiled online softmax, so the cost is
// O(n * head_dim) per query/head. `kind` selects the K/V decode (0 int8,
// 1 4-bit, 2 OCP FP8 E4M3); it lets one shader serve the three quantized
// attention built-ins.
// "quantize_q4": buffer 0 In (fp32, rows x cols), buffer 1 the packed
// 4-bit output (rows x cols/2 bytes, eight per uint), buffer 2 one fp32
// scale per row; scalars rows and cols (cols a multiple of 8). One thread
// per row.
// "quantize_fp8": buffer 0 is the fp32 rows x cols data quantized and
// dequantized in place to OCP FP8 E4M3, buffer 1 one fp32 scale per row;
// scalars rows and cols. The per-row scale is max(amax/448, 1/(448*512))
// with round-to-nearest-even, the W4A8 activation the served MXFP4 target
// feeds its linear layers. One thread per row.
// "attention_q4": like attention_q8 but keys/values are symmetric 4-bit
// (two per byte), same buffers, scalars and tiled dispatch.
// "quantize_fp8_pack": buffer 0 is In (fp32, rows x cols), buffer 1 the
// packed OCP FP8 E4M3 output (rows x cols bytes, four per uint), buffer 2
// one fp32 scale per row; scalars are rows and cols (cols a multiple of
// 4). The scale is max(amax/448, 1/(448*512)) with round-to-nearest-even,
// the fp8 KV cache quantization. One thread per row.
// "attention_fp8": like attention_q8 but keys and values are OCP FP8 E4M3,
// same buffers, scalars and tiled dispatch.
// "attention_split": split-N (flash-decoding) variant of attention for the
// latency-bound single-token path. The key range splits into `split`
// chunks; workgroup (i, h, s) writes the unnormalized partial acc plus
// the local max/sum for chunk s into pacc/pmax/psum, and
// "attention_combine" merges them. Buffers are q, k, v (as in
// attention), pacc (m*heads*split x head_dim fp32), pmax and psum
// (m*heads*split fp32 each); scalars are m, n, heads, kv_heads,
// head_dim, q_base, window, kv_f16, causal and split. The dispatch is
// m*heads*split workgroups of 256.
// "attention_q8_split", "attention_q4_split", "attention_fp8_split":
// split-N variants of the quantized attentions (same kind rule as
// attention_q8). Buffers are q, k, v, ks, vs (as in attention_q8) plus
// pacc, pmax, psum; scalars are m, n, heads, kv_heads, head_dim,
// q_base, window, kind and split.
// "attention_combine": merges split-N partials into the output.
// Buffers are pacc, pmax, psum and out (m*heads*head_dim fp32);
// scalars are m, heads, head_dim and split. One workgroup per
// (query row, head) of 256 threads.
// "cast_f32_f16": buffer 0 is In (fp32, n elements), buffer 1 the
// output (fp16, n elements, two per uint); scalar n (must be even). The
// dispatch is ceil(n/2 / 256).
// "round_bf16": buffer 0 is the fp32 data rounded in place to bfloat16
// (round to nearest even), keeping fp32 storage; scalar n. This is the
// bf16 activation rounding the served target's fused epilogues apply. The
// dispatch is ceil(n / 256).
// "concat_features": buffer 0 is In (fp32, n x rows x features),
// buffer 1 the output Out (fp32, rows x n*features); scalars are n, rows
// and features. This stacks feature vectors along the feature axis. The
// dispatch is ceil(rows * n * features / 256).
// "selector_edge_score": buffers are PredecessorCodebook (fp32,
// vocab x rank), SuccessorCodebook (fp32, vocab x rank), Hidden (fp32,
// batch x seq x rank), CandidateIds (int32, batch x seq x top_k),
// AnchorIds (int32, batch x seq), Unary (fp32, batch x seq x top_k) and
// the output Out (fp32, batch x seq x top_k x top_k); scalars are
// batch, seq, top_k, rank and vocab. This is the DFlash2 candidate
// selector transition score. Dispatch is ceil(batch*seq*top_k*top_k / 256).
// "dflash_conv": buffers are X (fp32, rows x channels), Delta (fp32,
// rows x taps x num_groups), Base (fp32, taps x channels) and the
// output Y (fp32, rows x channels); scalars are rows, channels, taps,
// group_size, block_size and delta_row_stride (num_groups =
// channels / group_size; delta_row_stride is at least taps x
// num_groups and lets one side of the kernel_projection output be used
// directly). The DFlash2 grouped dynamic convolution resets every
// block_size rows. The
// dispatch is ceil(rows * channels / 256) workgroups of 256.
// "conv1d_step": buffer 0 is X (fp32, channels x width, newest sample
// first), buffer 1 the weights W (fp32, channels x width), buffer 2 the
// output Y (fp32, channels); scalars are channels and width. Y is the
// current-step causal depthwise convolution. The dispatch is
// ceil(channels / 256) workgroups of 256.
// "conv1d": buffer 0 is X (fp32, channels x length), buffer 1 the
// weights W (fp32, channels x width), buffer 2 the output Y (fp32,
// channels x length); scalars are channels, length, width. Y is the
// causal depthwise convolution with sequential accumulation. The
// dispatch is ceil(channels * length / 256) workgroups of 256.
// "delta_step": buffer 0 is the recurrent state S (fp32, dk x dv,
// updated in place), buffers 1..3 are k (dk), v (dv) and q (dk),
// buffer 4 the output o (dv); scalars are dk, dv, the fp32 alpha bits
// and the fp32 beta bits. One step reads r = S^T k, writes
// S' = alpha*(S - beta*k*r^T) + beta*v*k^T, and reports o = S'^T q
// with per-column sequential accumulation. The dispatch is
// ceil(dv / 256) workgroups of 256.
// "mrope": buffer 0 holds rows x heads x head_dim fp32 rotated in
// place (NeoX pairing over rope_dim per head), buffer 1 the per-row
// (t, h, w) position triples (u64); scalars are rows, heads,
// head_dim, rope_dim, the fp32 theta bits, and the temporal, height
// and width section pair counts. Pair j rotates by its section
// position times theta^(-2j/rope_dim); pairs past the sections reuse
// the width id. The dispatch is
// ceil(rows * heads * (rope_dim / 2) / 256) workgroups of 256.
// "qgate_split": buffer 0 is the fused gated-attention projection
// (fp32, rows x heads*2*head_dim, per head query then gate), buffer 1
// the queries Q (fp32, rows x heads*head_dim), buffer 2 the gates G
// (fp32, rows x heads*head_dim); scalars are heads, head_dim and rows.
// The dispatch is ceil(rows * heads * head_dim / 256) workgroups of 256.
[[nodiscard]] inline StatusCode CheckBuiltInArgs(const Kernel& kernel,
                                                 const KernelLaunch& launch) {
  if (kernel.Id() == "fill" &&
      (launch.buffers.size() != 1 || launch.scalars.size() != 2)) {
    return StatusCode::InvalidArgument;
  }
  if (kernel.Id() == "gemm_q4k" || kernel.Id() == "gemm_q4k_row") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % kQ4KBlockElements != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_q4k_batched") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % kQ4KBlockElements != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "rope") {
    if (launch.buffers.size() != 1 || launch.scalars.size() != 6) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t head_dim = launch.scalars[2];
    const std::uint64_t rope_dim = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || head_dim == 0 ||
        rope_dim == 0 || rope_dim > head_dim || (head_dim % 2) != 0 ||
        (rope_dim % 2) != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 9) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[2];
    const std::uint64_t kv_heads = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || heads == 0 ||
        kv_heads == 0 || launch.scalars[4] == 0 ||
        (heads % kv_heads) != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention_split") {
    if (launch.buffers.size() != 6 || launch.scalars.size() != 10) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[2];
    const std::uint64_t kv_heads = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || heads == 0 ||
        kv_heads == 0 || launch.scalars[4] == 0 ||
        (heads % kv_heads) != 0 || launch.scalars[9] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention_combine") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 4) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0 || launch.scalars[3] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_fp8_block") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0 || launch.scalars[1] % 128 != 0 ||
        launch.scalars[2] % 128 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_bf16") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_f32") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_fp8") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_mxfp4" ||
      kernel.Id() == "gemm_mxfp4_batched") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 32 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_q5k" || kernel.Id() == "gemm_q5k_batched" ||
      kernel.Id() == "gemm_q6k" || kernel.Id() == "gemm_q6k_batched" ||
      kernel.Id() == "gemm_q3k" || kernel.Id() == "gemm_iq4xs" ||
      kernel.Id() == "gemm_iq4xs_batched" || kernel.Id() == "gemm_iq3s") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 256 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_iq4nl") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 32 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gemm_q80") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t k = launch.scalars[2];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || k == 0 ||
        k % 32 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "rmsnorm") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "sigmoid_gate") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 1) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "l2norm") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 4) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "rmsnorm_gated") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 5) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "ssm_gate") {
    if (launch.buffers.size() != 6 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "delta_step_heads") {
    if (launch.buffers.size() != 7 || launch.scalars.size() != 7) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0 || launch.scalars[3] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "repeat_heads") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 4) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[0];
    const std::uint64_t head_dim = launch.scalars[1];
    const std::uint64_t factor = launch.scalars[2];
    if (heads == 0 || head_dim == 0 || factor == 0 || heads % factor != 0 ||
        launch.scalars[3] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "add" || kernel.Id() == "silu_mul") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 1) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "spatial_merge") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 4) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t embed = launch.scalars[0];
    const std::uint64_t grid_w = launch.scalars[1];
    const std::uint64_t grid_h = launch.scalars[2];
    const std::uint64_t merge = launch.scalars[3];
    if (embed == 0 || grid_w == 0 || grid_h == 0 || merge == 0 ||
        grid_w % merge != 0 || grid_h % merge != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "image_patchify") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t h = launch.scalars[0];
    const std::uint64_t w = launch.scalars[1];
    const std::uint64_t patch = launch.scalars[2];
    if (h == 0 || w == 0 || patch == 0 || h % patch != 0 || w % patch != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "layernorm") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "bias_add") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 2 ||
        launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "gelu") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 1 ||
        launch.scalars[0] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "quantize_q8") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[1] % 4 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention_q8" || kernel.Id() == "attention_q4" ||
      kernel.Id() == "attention_fp8") {
    if (launch.buffers.size() != 6 || launch.scalars.size() != 8) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[2];
    const std::uint64_t kv_heads = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || heads == 0 ||
        kv_heads == 0 || launch.scalars[4] == 0 || heads % kv_heads != 0 ||
        launch.scalars[7] > 2) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "attention_q8_split" ||
      kernel.Id() == "attention_q4_split" ||
      kernel.Id() == "attention_fp8_split") {
    if (launch.buffers.size() != 8 || launch.scalars.size() != 9) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t heads = launch.scalars[2];
    const std::uint64_t kv_heads = launch.scalars[3];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || heads == 0 ||
        kv_heads == 0 || launch.scalars[4] == 0 || heads % kv_heads != 0 ||
        launch.scalars[7] > 2 || launch.scalars[8] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "quantize_q4") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[1] % 8 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "quantize_fp8") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "quantize_fp8_pack") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[1] % 4 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "cast_f32_f16") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 1) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[0] % 2 != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "round_bf16") {
    if (launch.buffers.size() != 1 || launch.scalars.size() != 1 ||
        launch.scalars[0] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "concat_features") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    for (std::uint64_t value : launch.scalars) {
      if (value == 0) {
        return StatusCode::InvalidArgument;
      }
    }
  }
  if (kernel.Id() == "selector_edge_score") {
    if (launch.buffers.size() != 7 || launch.scalars.size() != 5) {
      return StatusCode::InvalidArgument;
    }
    for (std::uint64_t value : launch.scalars) {
      if (value == 0) {
        return StatusCode::InvalidArgument;
      }
    }
  }
  if (kernel.Id() == "dflash_conv") {
    if (launch.buffers.size() != 4 || launch.scalars.size() != 7) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t channels = launch.scalars[1];
    const std::uint64_t group_size = launch.scalars[3];
    const std::uint64_t taps = launch.scalars[2];
    if (launch.scalars[0] == 0 || channels == 0 || taps == 0 ||
        group_size == 0 || launch.scalars[4] == 0 ||
        channels % group_size != 0 ||
        launch.scalars[5] < taps * (channels / group_size)) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "conv1d_step") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 2) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "conv1d") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 ||
        launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "delta_step") {
    if (launch.buffers.size() != 5 || launch.scalars.size() != 4) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "mrope") {
    if (launch.buffers.size() != 2 || launch.scalars.size() != 8) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t head_dim = launch.scalars[2];
    const std::uint64_t rope_dim = launch.scalars[3];
    const std::uint64_t sections =
        launch.scalars[5] + launch.scalars[6] + launch.scalars[7];
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0 || head_dim == 0 ||
        rope_dim == 0 || rope_dim > head_dim || (head_dim % 2) != 0 ||
        (rope_dim % 2) != 0 || sections > rope_dim / 2) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "qgate_split") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    if (launch.scalars[0] == 0 || launch.scalars[1] == 0) {
      return StatusCode::InvalidArgument;
    }
  }
  if (kernel.Id() == "embedding_f32" || kernel.Id() == "embedding_bf16" ||
      kernel.Id() == "embedding_q4k") {
    if (launch.buffers.size() != 3 || launch.scalars.size() != 3) {
      return StatusCode::InvalidArgument;
    }
    const std::uint64_t cols = launch.scalars[1];
    if (launch.scalars[0] == 0 || cols == 0 || launch.scalars[2] == 0) {
      return StatusCode::InvalidArgument;
    }
    if (kernel.Id() == "embedding_q4k" && cols % kQ4KBlockElements != 0) {
      return StatusCode::InvalidArgument;
    }
  }
  return StatusCode::Ok;
}

// Abstract compute backend. Exactly one instance per Engine; the concrete
// backend (vulkan or rocm) is selected at configure time via
// TESSERA_BACKEND. All device work in the core goes through this
// interface; no vendor type may cross it.
class Backend {
 public:
  virtual ~Backend() = default;

  // "vulkan" or "rocm".
  [[nodiscard]] virtual std::string_view Name() const = 0;
  // Human-readable device name, for diagnostics.
  [[nodiscard]] virtual std::string_view DeviceName() const = 0;

  // Point the backend at the engine's diagnostics channel (called once
  // after construction; before Init).
  virtual void SetDiagnostics(log::Diagnostics* diagnostics);

  // Select the GPU index this backend uses. Default 0 (the first GPU).
  // Call before Init; Init fails with InvalidArgument when the index is
  // out of range. The selected index and device name are logged.
  virtual void SetDeviceIndex(int index);

  // One-time device initialization. Returns Ok when already done.
  virtual std::expected<void, StatusCode> Init() = 0;

  // Allocate `bytes` of `kind` memory (0 bytes is InvalidArgument).
  //
  // Usage:
  //   auto result = backend.AllocateBuffer(4096, MemoryKind::Device);
  //   if (result) result->CopyH2D(*buffer, host_data);
  virtual std::expected<std::unique_ptr<Buffer>, StatusCode> AllocateBuffer(
      std::size_t bytes, MemoryKind kind) = 0;

  // Wrap an allocated handle into a Buffer. Backend implementations call
  // this after a successful allocation; the Buffer frees the handle
  // through this backend when destroyed.
  std::unique_ptr<Buffer> AdoptBuffer(void* handle, std::size_t bytes,
                                     MemoryKind kind, void* host_map);

  // Release a buffer. Called by Buffer's destructor only.
  virtual void FreeBuffer(MemoryKind kind, void* handle) = 0;

  // Wrap a loaded kernel into a Kernel. Backend implementations call this
  // after a successful LoadKernel.
  std::unique_ptr<Kernel> AdoptKernel(void* handle, std::string_view id);

  // Copy host bytes into `dst` (bounds-checked against the buffer size).
  virtual std::expected<void, StatusCode> CopyH2D(
      Buffer& dst, std::span<const std::byte> src) = 0;
  // Copy `bytes` from `src` into host memory (bounds-checked).
  virtual std::expected<void, StatusCode> CopyD2H(
      const Buffer& src, std::byte* dst, std::size_t bytes) = 0;
  // Copy `bytes` from `src` at `offset` into host memory
  // (bounds-checked). Reads a slice without staging the whole buffer
  // (embedding rows, single cache lines).
  virtual std::expected<void, StatusCode> CopyD2HAt(
      const Buffer& src, std::size_t offset, std::byte* dst,
      std::size_t bytes) = 0;
  // Copy `bytes` between two device buffers at the given offsets
  // (bounds-checked). The regions must not overlap. Keeps activations
  // and caches on the device without a host round-trip.
  virtual std::expected<void, StatusCode> CopyD2D(
      const Buffer& src, std::size_t src_offset, Buffer& dst,
      std::size_t dst_offset, std::size_t bytes) = 0;

  // Load a compiled kernel. Vulkan: `code` is a SPIR-V module and `name`
  // its entry point; an empty code resolves a built-in (today "fill" and
  // "gemm_q4k"). ROCm: built-ins only; a non-empty code is
  // UnsupportedFeature. A name the backend does not implement is
  // UnsupportedFeature.
  //
  // Usage:
  //   auto kernel = backend.LoadKernel("fill", {});
  //   if (!kernel) return 1;
  virtual std::expected<std::unique_ptr<Kernel>, StatusCode> LoadKernel(
      std::string_view name, std::span<const std::byte> code) = 0;

  // Release a loaded kernel. Called by the Kernel destructor only.
  virtual void FreeKernel(void* handle) = 0;

  // Launch `kernel`; arguments bind in kernel parameter order (buffers
  // first, then scalars). Shape checks: ValidateLaunch, CheckBuiltInArgs.
  virtual std::expected<void, StatusCode> LaunchKernel(
      const Kernel& kernel, const KernelLaunch& launch) = 0;

  // Block the host until all previously submitted device work is done.
  virtual void Synchronize() = 0;

 protected:
  void LogInfo(std::string_view message) const;
  void LogWarn(std::string_view message) const;
  void LogError(std::string_view message) const;
  log::Diagnostics* diagnostics_ = nullptr;
  // GPU index chosen by SetDeviceIndex (default 0, the first GPU).
  int device_index_ = 0;
};

// Construct the configured backend. Returns nullptr only when the build
// has no backend (a misconfiguration the engine reports as an error).
[[nodiscard]] std::unique_ptr<Backend> CreateBackend();

// Enumerate the GPUs the configured backend can see, in index order (the
// index is what EngineOptions::device_index selects). Creates its own
// temporary device context; no engine is needed. DeviceError when the
// backend is unavailable or sees no GPU.
[[nodiscard]] std::expected<std::vector<std::string>, StatusCode>
ListGpuNames();

}  // namespace tessera
