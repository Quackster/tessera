#include "models/qwen4exp/architecture.hpp"

#include <string_view>

// Name, weight conversion and tensor-parallel roles for the qwen4exp
// module. The GGUF uses the internal names directly, so MapWeightName is
// the default (identity); ConvertWeight is only consulted by the MXFP4
// loader, which this checkpoint does not use, so it makes no change.

namespace tessera::models::qwen4exp {

namespace {

bool EndsWith(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() &&
         text.substr(text.size() - suffix.size()) == suffix;
}

}  // namespace

std::string_view Qwen4ExpArchitecture::Name() const { return "qwen4exp"; }

std::optional<WeightConversion> Qwen4ExpArchitecture::ConvertWeight(
    std::string_view internal_name, const TransformerConfig& config) const {
  (void)internal_name;
  (void)config;
  return std::nullopt;
}

ParallelKind Qwen4ExpArchitecture::TensorParallelKind(
    std::string_view internal_name) const {
  // Column-parallel: the projections whose output rows split (gate/up, the
  // attention q/k/v, the GDN qkv and gate). Row-parallel: the projections
  // whose input columns split and whose partial output all-reduces (down,
  // attn_output, ssm_out). Expert-parallel: the routed expert tensors.
  for (const std::string_view suffix :
       {".ffn_gate.weight", ".ffn_up.weight", ".ffn_gate_shexp.weight",
        ".ffn_up_shexp.weight", ".attn_q.weight", ".attn_k.weight",
        ".attn_v.weight", ".attn_qkv.weight", ".attn_gate.weight"}) {
    if (EndsWith(internal_name, suffix)) {
      return ParallelKind::Column;
    }
  }
  for (const std::string_view suffix :
       {".ffn_down.weight", ".ffn_down_shexp.weight", ".attn_output.weight",
        ".ssm_out.weight"}) {
    if (EndsWith(internal_name, suffix)) {
      return ParallelKind::Row;
    }
  }
  for (const std::string_view suffix :
       {".ffn_gate_exps.weight", ".ffn_up_exps.weight",
        ".ffn_down_exps.weight"}) {
    if (EndsWith(internal_name, suffix)) {
      return ParallelKind::Expert;
    }
  }
  return ParallelKind::Replicated;
}

std::unique_ptr<Architecture> MakeQwen4ExpArchitecture() {
  return std::make_unique<Qwen4ExpArchitecture>();
}

}  // namespace tessera::models::qwen4exp
