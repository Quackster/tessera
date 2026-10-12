#include "tessera/hybrid.hpp"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "core/mesh.hpp"
#include "tessera/context_parallel.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tensor_parallel.hpp"

namespace tessera {

struct Hybrid::Impl {
  // One data-parallel group: either a lone engine/model (one-rank model
  // group) or a model-parallel team. Groups load one at a time, so only
  // one model is in host memory at once.
  struct Group {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    std::unique_ptr<TensorParallel> tp;
    std::unique_ptr<ContextParallel> cp;
  };
  std::vector<Group> groups;
  HybridMpKind mp_kind = HybridMpKind::Tensor;
  std::size_t context = 0;
  KvCacheType kv_type = KvCacheType::F32;
};

Hybrid::Hybrid(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Hybrid::~Hybrid() = default;
Hybrid::Hybrid(Hybrid&& other) noexcept = default;
Hybrid& Hybrid::operator=(Hybrid&& other) noexcept = default;

std::expected<std::unique_ptr<Hybrid>, StatusCode> Hybrid::Create(
    const HybridOptions& options) {
  core::DeviceMesh mesh;
  mesh.dp_groups = options.dp_groups;
  mesh.mp_ranks = options.devices.size() / (options.dp_groups == 0
                                                ? 1
                                                : options.dp_groups);
  mesh.devices = options.devices;
  if (options.dp_groups == 0 ||
      options.devices.size() != options.dp_groups * mesh.mp_ranks) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (auto valid = core::ValidateMesh(mesh); !valid) {
    return std::unexpected(valid.error());
  }
  auto impl = std::make_unique<Impl>();
  impl->mp_kind = options.mp_kind;
  impl->context = options.context;
  impl->kv_type = options.kv_type;
  impl->groups.reserve(mesh.dp_groups);
  for (std::size_t g = 0; g < mesh.dp_groups; ++g) {
    std::vector<int> ranks;
    ranks.reserve(mesh.mp_ranks);
    for (std::size_t r = 0; r < mesh.mp_ranks; ++r) {
      ranks.push_back(core::MeshDevice(mesh, g, r));
    }
    Impl::Group group;
    if (mesh.mp_ranks == 1) {
      EngineOptions engine_options;
      engine_options.device_index = ranks[0];
      auto engine = Engine::Create(engine_options);
      if (!engine) {
        return std::unexpected(engine.error());
      }
      ModelOptions model_options;
      model_options.path = options.model_path;
      model_options.max_context_length = options.context;
      auto model = (*engine)->LoadModel(model_options);
      if (!model) {
        return std::unexpected(model.error());
      }
      group.engine = std::move(*engine);
      group.model = std::move(*model);
    } else if (options.mp_kind == HybridMpKind::Tensor) {
      TensorParallelOptions tp_options;
      tp_options.model_path = options.model_path;
      tp_options.devices = ranks;
      tp_options.context = options.context;
      tp_options.kv_type = options.kv_type;
      auto tp = TensorParallel::Create(tp_options);
      if (!tp) {
        return std::unexpected(tp.error());
      }
      group.tp = std::move(*tp);
    } else {
      ContextParallelOptions cp_options;
      cp_options.model_path = options.model_path;
      cp_options.devices = ranks;
      cp_options.context = options.context;
      cp_options.kv_type = options.kv_type;
      auto cp = ContextParallel::Create(cp_options);
      if (!cp) {
        return std::unexpected(cp.error());
      }
      group.cp = std::move(*cp);
    }
    impl->groups.push_back(std::move(group));
  }
  return std::unique_ptr<Hybrid>(new Hybrid(std::move(impl)));
}

std::expected<std::vector<std::vector<std::uint32_t>>, StatusCode>
Hybrid::GenerateMany(std::span<const std::vector<std::uint32_t>> prompts,
                     std::size_t max_tokens) {
  Impl& p = *impl_;
  if (prompts.size() != p.groups.size()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  std::vector<std::expected<std::vector<std::uint32_t>, StatusCode>> outputs(
      p.groups.size());
  std::vector<std::thread> threads;
  threads.reserve(p.groups.size());
  for (std::size_t g = 0; g < p.groups.size(); ++g) {
    threads.emplace_back([this, g, &prompts, max_tokens, &outputs] {
      Impl::Group& group = impl_->groups[g];
      if (group.tp != nullptr) {
        outputs[g] = group.tp->Generate(prompts[g], max_tokens);
      } else if (group.cp != nullptr) {
        outputs[g] = group.cp->Generate(prompts[g], max_tokens);
      } else {
        GenerateOptions options;
        options.prompt_tokens = prompts[g];
        options.max_completion_tokens = max_tokens;
        options.kv_type = impl_->kv_type;
        outputs[g] = group.engine->Generate(*group.model, options);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  std::vector<std::vector<std::uint32_t>> result;
  result.reserve(p.groups.size());
  for (auto& output : outputs) {
    if (!output) {
      return std::unexpected(output.error());
    }
    result.push_back(std::move(*output));
  }
  return result;
}

std::size_t Hybrid::GroupCount() const { return impl_->groups.size(); }

const Tokenizer* Hybrid::GetTokenizer() const {
  if (impl_->groups.empty()) {
    return nullptr;
  }
  const Impl::Group& group = impl_->groups.front();
  if (group.tp != nullptr) {
    return group.tp->GetTokenizer();
  }
  if (group.cp != nullptr) {
    return group.cp->GetTokenizer();
  }
  return group.model->GetTokenizer();
}

std::expected<std::string, StatusCode> Hybrid::ChatPrompt(
    std::string_view user_text, bool enable_thinking,
    std::string_view reasoning_effort) const {
  if (impl_->groups.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const Impl::Group& group = impl_->groups.front();
  if (group.tp != nullptr) {
    return group.tp->ChatPrompt(user_text, enable_thinking, reasoning_effort);
  }
  if (group.cp != nullptr) {
    return group.cp->ChatPrompt(user_text, enable_thinking, reasoning_effort);
  }
  return group.model->ChatPrompt(user_text, enable_thinking, reasoning_effort);
}

}  // namespace tessera
