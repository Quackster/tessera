#include "tessera/pipeline.hpp"

#include <memory>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"

namespace tessera {

namespace {

std::uint32_t ArgMax(std::span<const float> values) {
  std::uint32_t best = 0;
  float best_value = values[0];
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (values[i] > best_value) {
      best_value = values[i];
      best = static_cast<std::uint32_t>(i);
    }
  }
  return best;
}

bool IsStop(std::uint32_t token, std::span<const std::uint32_t> stops) {
  for (const std::uint32_t stop : stops) {
    if (stop == token) {
      return true;
    }
  }
  return false;
}

}  // namespace

struct Pipeline::Impl {
  // One pipeline stage: its own engine, sharded model and decode cache, plus
  // the buffers that carry the boundary hidden to and from the neighbours.
  struct Stage {
    std::unique_ptr<Engine> engine;
    std::unique_ptr<Model> model;
    std::unique_ptr<core::DecodeCache> cache;
    std::unique_ptr<Buffer> incoming;  // null on the first stage
    std::unique_ptr<Buffer> outgoing;  // null on the last stage
    // The last trunk layer this stage owns; the boundary capture reads the
    // residual hidden after it.
    std::size_t boundary_layer = 0;
  };

  std::vector<Stage> stages;
  std::size_t hidden_dim = 0;
  std::size_t context = 0;
  std::vector<std::uint32_t> stops;

  // Reset every stage's cache so a new sequence starts at position zero.
  void Reset() {
    for (Stage& stage : stages) {
      stage.cache = std::make_unique<core::DecodeCache>();
    }
  }

  // Run `token` through every stage, returning the last stage's logits
  // when `want_logits`; otherwise only the trunk runs and the logits
  // result is empty. The hidden handoff copies each stage's boundary
  // residual into the next stage's incoming buffer.
  std::expected<std::vector<float>, StatusCode> Step(std::uint32_t token,
                                                     bool want_logits) {
    const std::size_t n = stages.size();
    for (std::size_t i = 0; i < n; ++i) {
      Stage& stage = stages[i];
      const Architecture* arch = stage.model->Arch();
      if (arch == nullptr) {
        return std::unexpected(StatusCode::UnsupportedFeature);
      }
      Backend& backend = stage.engine->Owner();
      const Buffer* incoming = stage.incoming.get();
      if (i + 1 == n) {
        if (!want_logits) {
          auto forwarded = arch->Forward(backend, *stage.model, *stage.cache,
                                         token, nullptr, nullptr, nullptr,
                                         incoming);
          if (!forwarded) {
            return std::unexpected(forwarded.error());
          }
          return std::vector<float>{};
        }
        return arch->Logits(backend, *stage.model, *stage.cache, token,
                            nullptr, nullptr, nullptr, incoming);
      }
      std::vector<std::size_t> capture_layers{stage.boundary_layer};
      std::vector<Buffer*> capture{stage.outgoing.get()};
      auto forwarded =
          arch->Forward(backend, *stage.model, *stage.cache, token, nullptr,
                        &capture_layers, &capture, incoming);
      if (!forwarded) {
        return std::unexpected(forwarded.error());
      }
      Stage& next = stages[i + 1];
      auto copied = CopyAcrossDevices(*stage.outgoing, 0, *next.incoming, 0,
                                      hidden_dim * 4);
      if (!copied) {
        return std::unexpected(copied.error());
      }
    }
    return std::unexpected(StatusCode::DeviceError);
  }
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Pipeline::~Pipeline() = default;
Pipeline::Pipeline(Pipeline&& other) noexcept = default;
Pipeline& Pipeline::operator=(Pipeline&& other) noexcept = default;

std::expected<std::unique_ptr<Pipeline>, StatusCode> Pipeline::Create(
    const PipelineOptions& options) {
  if (options.devices.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto config = Model::ProbeConfig(options.model_path);
  if (!config) {
    return std::unexpected(config.error());
  }
  const std::size_t layers = config->layers;
  const std::size_t stage_count = options.devices.size();
  if (layers < stage_count) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  auto impl = std::make_unique<Impl>();
  impl->hidden_dim = config->hidden_dim;
  impl->context = options.context;
  impl->stages.reserve(stage_count);
  for (std::size_t i = 0; i < stage_count; ++i) {
    // Contiguous, balanced ranges: stage i owns [layers*i/n, layers*(i+1)/n).
    const std::size_t first = layers * i / stage_count;
    const std::size_t last = layers * (i + 1) / stage_count;
    if (last <= first) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
    EngineOptions engine_options;
    engine_options.device_index = options.devices[i];
    auto engine = Engine::Create(engine_options);
    if (!engine) {
      return std::unexpected(engine.error());
    }
    ModelOptions model_options;
    model_options.path = options.model_path;
    model_options.max_context_length = options.context;
    model_options.shard.enabled = true;
    model_options.shard.first_layer = first;
    model_options.shard.last_layer = last;
    model_options.shard.first_stage = (i == 0);
    model_options.shard.last_stage = (i + 1 == stage_count);
    auto model = (*engine)->LoadModel(model_options);
    if (!model) {
      return std::unexpected(model.error());
    }
    Impl::Stage stage;
    stage.engine = std::move(*engine);
    stage.model = std::move(*model);
    stage.boundary_layer = last - 1;
    if (i > 0) {
      auto buffer = stage.engine->Owner().AllocateBuffer(impl->hidden_dim * 4,
                                                         MemoryKind::Device);
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      stage.incoming = std::move(*buffer);
    }
    if (i + 1 < stage_count) {
      auto buffer = stage.engine->Owner().AllocateBuffer(impl->hidden_dim * 4,
                                                         MemoryKind::Device);
      if (!buffer) {
        return std::unexpected(buffer.error());
      }
      stage.outgoing = std::move(*buffer);
    }
    impl->stages.push_back(std::move(stage));
  }
  const std::span<const std::uint32_t> stops =
      impl->stages.back().model->StopTokens();
  impl->stops.assign(stops.begin(), stops.end());
  return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

std::expected<std::vector<std::uint32_t>, StatusCode> Pipeline::Generate(
    std::span<const std::uint32_t> prompt, std::size_t max_tokens) {
  Impl& p = *impl_;
  if (prompt.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (prompt.size() > p.context) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  p.Reset();
  std::expected<std::vector<float>, StatusCode> logits;
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    const bool last = (i + 1 == prompt.size());
    if (last) {
      logits = p.Step(prompt[i], true);
    } else {
      auto step = p.Step(prompt[i], false);
      if (!step) {
        return std::unexpected(step.error());
      }
    }
  }
  if (!logits) {
    return std::unexpected(logits.error());
  }
  const std::size_t budget =
      max_tokens == 0 ? p.context - prompt.size() : max_tokens;
  std::vector<std::uint32_t> produced;
  std::uint32_t next = ArgMax(*logits);
  for (std::size_t count = 0; count < budget; ++count) {
    if (IsStop(next, p.stops)) {
      break;
    }
    produced.push_back(next);
    auto step = p.Step(next, true);
    if (!step) {
      return std::unexpected(step.error());
    }
    next = ArgMax(*step);
  }
  return produced;
}

std::size_t Pipeline::StageCount() const { return impl_->stages.size(); }

const Tokenizer* Pipeline::GetTokenizer() const {
  return impl_->stages.empty() ? nullptr
                               : impl_->stages.front().model->GetTokenizer();
}

std::expected<std::string, StatusCode> Pipeline::ChatPrompt(
    std::string_view user_text, bool enable_thinking) const {
  if (impl_->stages.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  return impl_->stages.front().model->ChatPrompt(user_text, enable_thinking);
}

}  // namespace tessera
