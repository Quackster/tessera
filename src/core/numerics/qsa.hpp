#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>

#include "tessera/types.hpp"

// QSA (the qwen4exp full-attention layers) numeric contracts, ported from
// ../Strata include/strata/kernels/qsa.hpp. QSA is GQA over a paged FP16
// KV cache with a sparse indexer: the indexer pools one key per
// `idx_block` cells, scores every cached cell, and the selection keeps
// the largest cells. This file starts with the scoring step.

namespace tessera::core {

// The per-token step values every QSA layer kernel reads, computed from
// the position: step = [pos, n_kv = pos+1, n_bid = n_kv/idx_block,
// width = min(n_kv, idx_top_k + idx_block - 1)]. Keeping the derivation
// here means a caller cannot update one and forget another.
constexpr std::size_t kQsaStepCount = 4;

// Fill a host `step` array (kQsaStepCount entries) from `pos`.
void QsaStepFill(std::span<std::int32_t> step, std::int32_t pos,
                 std::size_t idx_block, std::size_t idx_top_k);

// Indexer scoring geometry.
struct QsaIndexShapes {
  std::size_t idx_n_head = 0;  // indexer query heads
  std::size_t idx_dim = 0;     // indexer key width
  std::size_t idx_block = 0;   // cells pooled into one block key
  std::size_t n_bid = 0;       // completed blocks
  std::size_t n_kv = 0;        // cached cells
};

// Host reference. `pooled` is (n_bid + 1, idx_dim), `q_idx` is
// (idx_n_head, idx_dim), `bias` is (n_bid + 1) or empty. `scores` receives
// n_kv values:
//
//   cb = min(j / idx_block, n_bid)
//   score[j] = bias[cb] + sum_h Relu(dot(pooled[cb], q_idx[h]))
//   score[j] += 1e9 for cells of the incomplete tail
//
// The Relu is per head, before the sum over heads. A size mismatch is
// InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> QsaIndexRef(
    std::span<const float> pooled, std::span<const float> q_idx,
    std::span<const float> bias, const QsaIndexShapes& s,
    std::span<float> scores);

// One token's K and V into the paged FP16 cache at logical cell `pos`.
// The pool is [page][kv_head][page_size][head_dim] fp16 with head_dim
// fastest; `page_table[page]` gives the physical page. `kcur`/`vcur` are
// (n_head_kv, head_dim) f32. Writes kcur and vcur into k_pool and v_pool.
// A size mismatch or an out-of-range pos is InvalidArgument.
[[nodiscard]] std::expected<void, StatusCode> KvAppendRef(
    std::span<std::uint16_t> k_pool, std::span<std::uint16_t> v_pool,
    std::span<const std::int32_t> page_table, std::size_t pos,
    std::span<const float> kcur, std::span<const float> vcur,
    std::size_t n_head_kv, std::size_t head_dim, std::size_t page_size);

// The QSA output gate: out[h][d] = fp16(attn[h][d] *
// sigmoid(q_full[h][head_dim + d])). `attn` is (n_head, head_dim) f32,
// `q_full` is (n_head, 2*head_dim) f32 (the raw q projection), `out` is
// (n_head, head_dim) fp16. The split is per head, not interleaved.
[[nodiscard]] std::expected<void, StatusCode> QsaGateApplyRef(
    std::span<const float> attn, std::span<const float> q_full,
    std::size_t n_head, std::size_t head_dim, std::span<std::uint16_t> out);

// The selected cells through the page table into a contiguous scratch.
// `ids` are logical cells (ascending, unique); `k_scratch`/`v_scratch`
// receive (n_ids, n_head_kv, head_dim) fp16 in full. Pool layout is
// [page][kv_head][page_size][head_dim] with head_dim fastest.
[[nodiscard]] std::expected<void, StatusCode> KvGatherRef(
    std::span<const std::uint16_t> k_pool, std::span<const std::uint16_t> v_pool,
    std::span<const std::int32_t> page_table, std::span<const std::int32_t> ids,
    std::size_t n_head_kv, std::size_t head_dim, std::size_t page_size,
    std::span<std::uint16_t> k_scratch, std::span<std::uint16_t> v_scratch);

// Softmax GQA over the gathered cells. `q` is (n_head, head_dim) f32,
// `k_scratch`/`v_scratch` are (n_ids, n_head_kv, head_dim) fp16, `attn`
// receives (n_head, head_dim) f32. The q -> kv head map is integer
// division q / (n_head / n_head_kv). An empty `ids` gives a zero `attn`.
[[nodiscard]] std::expected<void, StatusCode> QsaAttendRef(
    std::span<const float> q, std::span<const std::uint16_t> k_scratch,
    std::span<const std::uint16_t> v_scratch, std::size_t n_ids,
    std::size_t n_head, std::size_t n_head_kv, std::size_t head_dim,
    std::span<float> attn);

// The largest `width` cells of `scores` (n_kv of them), returned as their
// indices in ascending order. Ties are broken by ascending index. `ids`
// must hold exactly `width` entries; width == 0 is a no-op.
[[nodiscard]] std::expected<void, StatusCode> QsaTopKRef(
    std::span<const float> scores, std::size_t width,
    std::span<std::int32_t> ids);

// Append one raw indexer key and, when this cell completes a block, pool
// and rotate that block's row.
//
//   tail (idx_block - 1, idx_dim) f32: the RAW keys of the block being
//     filled, row = pos % idx_block (a cell that completes a block is not
//     stored there)
//   dead (idx_dim) f32: the spare slot's key, rms_norm(raw of cell 0)
//   pooled (rows, idx_dim) f32: block keys, the spare slot at row n_bid
//   block_pos (1) i32: the completed block's first-cell position
//
// On completion, pooled[b] = rope_neox(rms_norm(mean of the block's r raw
// keys), pos_base + b*r) with b = pos / idx_block; the spare slot moves to
// row b+1. Callers maintain the buffers across tokens.
[[nodiscard]] std::expected<void, StatusCode> IndexerKeyAppendRef(
    std::span<const float> raw, std::int32_t pos, std::int32_t pos_base,
    std::span<const float> w_k_norm, double eps, std::span<float> tail,
    std::span<float> dead, std::span<float> pooled,
    std::span<std::int32_t> block_pos, std::span<const float> cos_tab,
    std::span<const float> sin_tab, std::size_t idx_dim,
    std::size_t idx_block, std::size_t n_rot);

}  // namespace tessera::core
