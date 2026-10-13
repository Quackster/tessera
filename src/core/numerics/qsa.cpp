#include "core/numerics/qsa.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "core/numerics/quant.hpp"

namespace tessera::core {

namespace {

void StoreF16(std::span<std::uint16_t> pool, std::size_t index, float value) {
  pool[index] = Fp32ToHalfBits(value);
}

}  // namespace

void QsaStepFill(std::span<std::int32_t> step, std::int32_t pos,
                 std::size_t idx_block, std::size_t idx_top_k) {
  if (step.size() != kQsaStepCount || idx_block == 0) {
    return;
  }
  const std::size_t n_kv = static_cast<std::size_t>(pos) + 1;
  const std::size_t width_budget = idx_top_k + idx_block - 1;
  const std::size_t width = n_kv < width_budget ? n_kv : width_budget;
  step[0] = pos;
  step[1] = static_cast<std::int32_t>(n_kv);
  step[2] = static_cast<std::int32_t>(n_kv / idx_block);
  step[3] = static_cast<std::int32_t>(width);
}

std::expected<void, StatusCode> QsaIndexRef(
    std::span<const float> pooled, std::span<const float> q_idx,
    std::span<const float> bias, const QsaIndexShapes& s,
    std::span<float> scores) {
  if (s.idx_n_head == 0 || s.idx_dim == 0 || s.idx_block == 0 ||
      pooled.size() != (s.n_bid + 1) * s.idx_dim ||
      q_idx.size() != s.idx_n_head * s.idx_dim ||
      (!bias.empty() && bias.size() != s.n_bid + 1) ||
      scores.size() != s.n_kv) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t full = (s.n_kv / s.idx_block) * s.idx_block;
  for (std::size_t j = 0; j < s.n_kv; ++j) {
    std::size_t cb = j / s.idx_block;
    if (cb > s.n_bid) {
      cb = s.n_bid;
    }
    const float* p = pooled.data() + cb * s.idx_dim;
    double score = bias.empty() ? 0.0 : static_cast<double>(bias[cb]);
    for (std::size_t h = 0; h < s.idx_n_head; ++h) {
      const float* q = q_idx.data() + h * s.idx_dim;
      double dot = 0.0;
      for (std::size_t d = 0; d < s.idx_dim; ++d) {
        dot += static_cast<double>(p[d]) * q[d];
      }
      if (dot > 0.0) {
        score += dot;
      }
    }
    if (j >= full) {
      score += 1e9;
    }
    scores[j] = static_cast<float>(score);
  }
  return {};
}

std::expected<void, StatusCode> KvAppendRef(
    std::span<std::uint16_t> k_pool, std::span<std::uint16_t> v_pool,
    std::span<const std::int32_t> page_table, std::size_t pos,
    std::span<const float> kcur, std::span<const float> vcur,
    std::size_t n_head_kv, std::size_t head_dim, std::size_t page_size) {
  if (n_head_kv == 0 || head_dim == 0 || page_size == 0 ||
      page_table.empty() || kcur.size() != n_head_kv * head_dim ||
      vcur.size() != n_head_kv * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t cells = page_size * page_table.size();
  if (pos >= cells) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t logical_page = pos / page_size;
  const std::int32_t physical = page_table[logical_page];
  if (physical < 0) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t cell_in_page = pos % page_size;
  if (k_pool.size() < cells * n_head_kv * head_dim ||
      v_pool.size() < cells * n_head_kv * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < n_head_kv; ++h) {
    const std::size_t base =
        (static_cast<std::size_t>(physical) * n_head_kv + h) * page_size +
        cell_in_page;
    for (std::size_t d = 0; d < head_dim; ++d) {
      StoreF16(k_pool, base * head_dim + d, kcur[h * head_dim + d]);
      StoreF16(v_pool, base * head_dim + d, vcur[h * head_dim + d]);
    }
  }
  return {};
}

std::expected<void, StatusCode> QsaGateApplyRef(
    std::span<const float> attn, std::span<const float> q_full,
    std::size_t n_head, std::size_t head_dim, std::span<std::uint16_t> out) {
  if (n_head == 0 || head_dim == 0 || attn.size() != n_head * head_dim ||
      q_full.size() != n_head * 2 * head_dim ||
      out.size() != n_head * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  for (std::size_t h = 0; h < n_head; ++h) {
    for (std::size_t d = 0; d < head_dim; ++d) {
      const float gate =
          q_full[h * 2 * head_dim + head_dim + d];
      const float value =
          attn[h * head_dim + d] / (1.0f + std::exp(-gate));
      out[h * head_dim + d] = Fp32ToHalfBits(value);
    }
  }
  return {};
}

std::expected<void, StatusCode> KvGatherRef(
    std::span<const std::uint16_t> k_pool, std::span<const std::uint16_t> v_pool,
    std::span<const std::int32_t> page_table, std::span<const std::int32_t> ids,
    std::size_t n_head_kv, std::size_t head_dim, std::size_t page_size,
    std::span<std::uint16_t> k_scratch, std::span<std::uint16_t> v_scratch) {
  if (n_head_kv == 0 || head_dim == 0 || page_size == 0 ||
      k_scratch.size() != ids.size() * n_head_kv * head_dim ||
      v_scratch.size() != ids.size() * n_head_kv * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t cells = page_size * page_table.size();
  for (std::size_t j = 0; j < ids.size(); ++j) {
    const std::int32_t id = ids[j];
    if (id < 0 || static_cast<std::size_t>(id) >= cells) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    const std::size_t physical =
        static_cast<std::size_t>(page_table[id / page_size]);
    for (std::size_t h = 0; h < n_head_kv; ++h) {
      const std::size_t base =
          (physical * n_head_kv + h) * page_size + (id % page_size);
      for (std::size_t d = 0; d < head_dim; ++d) {
        const std::size_t dst = (j * n_head_kv + h) * head_dim + d;
        const std::size_t src = base * head_dim + d;
        if (src >= k_pool.size() || src >= v_pool.size()) {
          return std::unexpected(StatusCode::InvalidArgument);
        }
        k_scratch[dst] = k_pool[src];
        v_scratch[dst] = v_pool[src];
      }
    }
  }
  return {};
}

std::expected<void, StatusCode> QsaAttendRef(
    std::span<const float> q, std::span<const std::uint16_t> k_scratch,
    std::span<const std::uint16_t> v_scratch, std::size_t n_ids,
    std::size_t n_head, std::size_t n_head_kv, std::size_t head_dim,
    std::span<float> attn) {
  if (n_head == 0 || n_head_kv == 0 || head_dim == 0 ||
      n_head % n_head_kv != 0 || q.size() != n_head * head_dim ||
      k_scratch.size() != n_ids * n_head_kv * head_dim ||
      v_scratch.size() != n_ids * n_head_kv * head_dim ||
      attn.size() != n_head * head_dim) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t ratio = n_head / n_head_kv;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
  std::vector<double> scores(n_ids);
  for (std::size_t h = 0; h < n_head; ++h) {
    const std::size_t kv = h / ratio;
    if (n_ids == 0) {
      for (std::size_t d = 0; d < head_dim; ++d) {
        attn[h * head_dim + d] = 0.0f;
      }
      continue;
    }
    double best = -1e30;
    for (std::size_t j = 0; j < n_ids; ++j) {
      const std::uint16_t* kk = k_scratch.data() + (j * n_head_kv + kv) * head_dim;
      double dot = 0.0;
      for (std::size_t d = 0; d < head_dim; ++d) {
        dot += static_cast<double>(q[h * head_dim + d]) *
               Fp16ToFloat(kk[d]);
      }
      scores[j] = dot * scale;
      if (scores[j] > best) {
        best = scores[j];
      }
    }
    double sum = 0.0;
    for (std::size_t j = 0; j < n_ids; ++j) {
      scores[j] = std::exp(scores[j] - best);
      sum += scores[j];
    }
    for (std::size_t d = 0; d < head_dim; ++d) {
      double acc = 0.0;
      for (std::size_t j = 0; j < n_ids; ++j) {
        const std::uint16_t* vv =
            v_scratch.data() + (j * n_head_kv + kv) * head_dim;
        acc += (scores[j] / sum) * static_cast<double>(Fp16ToFloat(vv[d]));
      }
      attn[h * head_dim + d] = static_cast<float>(acc);
    }
  }
  return {};
}

std::expected<void, StatusCode> QsaTopKRef(
    std::span<const float> scores, std::size_t width,
    std::span<std::int32_t> ids) {
  if (width > scores.size() || ids.size() != width) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<std::size_t> order(scores.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  std::stable_sort(order.begin(), order.end(),
                   [&scores](std::size_t a, std::size_t b) {
                     if (scores[a] != scores[b]) {
                       return scores[a] > scores[b];
                     }
                     return a < b;
                   });
  for (std::size_t i = 0; i < width; ++i) {
    ids[i] = static_cast<std::int32_t>(order[i]);
  }
  std::sort(ids.begin(), ids.end());
  return {};
}

std::expected<void, StatusCode> IndexerKeyAppendRef(
    std::span<const float> raw, std::int32_t pos, std::int32_t pos_base,
    std::span<const float> w_k_norm, double eps, std::span<float> tail,
    std::span<float> dead, std::span<float> pooled,
    std::span<std::int32_t> block_pos, std::span<const float> cos_tab,
    std::span<const float> sin_tab, std::size_t idx_dim,
    std::size_t idx_block, std::size_t n_rot) {
  if (idx_dim == 0 || idx_block < 2 || n_rot == 0 || (n_rot % 2) != 0 ||
      n_rot > idx_dim || raw.size() != idx_dim || w_k_norm.size() != idx_dim ||
      tail.size() != (idx_block - 1) * idx_dim || dead.size() != idx_dim ||
      pooled.size() % idx_dim != 0 || block_pos.size() != 1) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t half = n_rot / 2;
  const std::size_t b = static_cast<std::size_t>(pos) / idx_block;
  const std::size_t slot = static_cast<std::size_t>(pos) % idx_block;
  if (pooled.size() / idx_dim < b + 2) {
    return std::unexpected(StatusCode::InvalidArgument);
  }

  auto rotate = [&](float* row, std::int64_t position) {
    if (cos_tab.size() < static_cast<std::size_t>(position) * half + half ||
        sin_tab.size() < static_cast<std::size_t>(position) * half + half) {
      return false;
    }
    for (std::size_t d = 0; d < half; ++d) {
      const std::size_t toff = static_cast<std::size_t>(position) * half + d;
      const float a = row[d];
      const float b2 = row[half + d];
      row[d] = a * cos_tab[toff] - b2 * sin_tab[toff];
      row[half + d] = a * sin_tab[toff] + b2 * cos_tab[toff];
    }
    return true;
  };

  if (slot < idx_block - 1) {
    for (std::size_t d = 0; d < idx_dim; ++d) {
      tail[slot * idx_dim + d] = raw[d];
    }
  }
  if (pos == 0) {
    double ss = 0.0;
    for (std::size_t d = 0; d < idx_dim; ++d) {
      ss += static_cast<double>(raw[d]) * raw[d];
    }
    const double inv =
        1.0 / std::sqrt(ss / static_cast<double>(idx_dim) + eps);
    for (std::size_t d = 0; d < idx_dim; ++d) {
      float y = static_cast<float>(static_cast<double>(raw[d]) * inv *
                                   static_cast<double>(w_k_norm[d]));
      if (d < n_rot) {
        y *= cos_tab[d % half];
      }
      dead[d] = y;
      pooled[d] = y;
    }
  }
  if (slot != idx_block - 1) {
    return {};
  }
  std::vector<double> mean(idx_dim, 0.0);
  for (std::size_t d = 0; d < idx_dim; ++d) {
    double m = 0.0;
    for (std::size_t j = 0; j + 1 < idx_block; ++j) {
      m += static_cast<double>(tail[j * idx_dim + d]);
    }
    m += static_cast<double>(raw[d]);
    mean[d] = m / static_cast<double>(idx_block);
  }
  double ss = 0.0;
  for (std::size_t d = 0; d < idx_dim; ++d) {
    ss += mean[d] * mean[d];
  }
  const double inv =
      1.0 / std::sqrt(ss / static_cast<double>(idx_dim) + eps);
  float* row = pooled.data() + b * idx_dim;
  float* spare = pooled.data() + (b + 1) * idx_dim;
  for (std::size_t d = 0; d < idx_dim; ++d) {
    row[d] = static_cast<float>(mean[d] * inv *
                                static_cast<double>(w_k_norm[d]));
    spare[d] = dead[d];
  }
  block_pos[0] = static_cast<std::int32_t>(
      pos_base + static_cast<std::int32_t>(b * idx_block));
  if (!rotate(row, pos_base + static_cast<std::int64_t>(b * idx_block))) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return {};
}

}  // namespace tessera::core
