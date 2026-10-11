#include "tessera/pipeline.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
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
    // Owned trunk layer range [first_layer, last_layer) for budgeting.
    std::size_t first_layer = 0;
    std::size_t last_layer = 0;
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
    stage.first_layer = first;
    stage.last_layer = last;
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

namespace {

// A blocking multi-producer/multi-consumer queue for the pipeline jobs.
struct Job {
  std::uint32_t seq = 0;
  std::uint32_t token = 0;
  bool want_head = false;
};

// Job and sequence sentinels (outside the valid ranges).
constexpr std::uint32_t kStopSeq = 0xFFFFFFFFu;
constexpr std::uint32_t kNoToken = 0xFFFFFFFFu;
constexpr std::uint32_t kErrorTok = 0xFFFFFFFEu;

class JobQueue {
 public:
  void Push(Job job) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      jobs_.push_back(job);
    }
    ready_.notify_one();
  }
  Job Pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [this] { return !jobs_.empty(); });
    Job job = jobs_.front();
    jobs_.pop_front();
    return job;
  }

 private:
  std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<Job> jobs_;
};

}  // namespace

std::expected<std::vector<std::vector<std::uint32_t>>, StatusCode>
Pipeline::GenerateMany(std::span<const std::vector<std::uint32_t>> prompts,
                       std::size_t max_tokens) {
  Impl& p = *impl_;
  const std::size_t stage_count = p.stages.size();
  const std::size_t count = prompts.size();
  if (count == 0) {
    return std::vector<std::vector<std::uint32_t>>{};
  }
  for (const std::vector<std::uint32_t>& prompt : prompts) {
    if (prompt.empty() || prompt.size() > p.context) {
      return std::unexpected(StatusCode::InvalidArgument);
    }
  }
  // Per-stage, per-sequence caches and boundary buffers. Each stage owns one
  // cache per sequence so the key/value rows stay with the sequence, and one
  // incoming/outgoing hidden per sequence so a boundary copy for one sequence
  // never races another in flight.
  struct StageRuntime {
    std::vector<std::unique_ptr<core::DecodeCache>> caches;
    std::vector<std::unique_ptr<Buffer>> incoming;  // null on stage 0
    std::vector<std::unique_ptr<Buffer>> outgoing;  // null on the last stage
    JobQueue input;
    std::thread worker;
  };
  std::vector<StageRuntime> runtime(stage_count);
  for (std::size_t i = 0; i < stage_count; ++i) {
    StageRuntime& rt = runtime[i];
    rt.caches.resize(count);
    for (std::size_t s = 0; s < count; ++s) {
      rt.caches[s] = std::make_unique<core::DecodeCache>();
    }
    Backend& backend = p.stages[i].engine->Owner();
    if (i > 0) {
      rt.incoming.resize(count);
      for (std::size_t s = 0; s < count; ++s) {
        auto buffer =
            backend.AllocateBuffer(p.hidden_dim * 4, MemoryKind::Device);
        if (!buffer) {
          return std::unexpected(buffer.error());
        }
        rt.incoming[s] = std::move(*buffer);
      }
    }
    if (i + 1 < stage_count) {
      rt.outgoing.resize(count);
      for (std::size_t s = 0; s < count; ++s) {
        auto buffer =
            backend.AllocateBuffer(p.hidden_dim * 4, MemoryKind::Device);
        if (!buffer) {
          return std::unexpected(buffer.error());
        }
        rt.outgoing[s] = std::move(*buffer);
      }
    }
  }
  JobQueue output;
  std::atomic<bool> failed{false};
  // One worker thread per stage. It pulls jobs for its stage, runs the
  // stage's forward, hands the boundary hidden to the next stage, and the
  // last stage emits the greedy token.
  for (std::size_t i = 0; i < stage_count; ++i) {
    runtime[i].worker = std::thread([&, i] {
      Impl::Stage& stage = p.stages[i];
      Backend& backend = stage.engine->Owner();
      const Architecture* arch = stage.model->Arch();
      for (;;) {
        const Job job = runtime[i].input.Pop();
        if (job.seq == kStopSeq) {
          if (i + 1 < stage_count) {
            runtime[i + 1].input.Push(Job{kStopSeq, 0, false});
          }
          return;
        }
        if (arch == nullptr) {
          failed.store(true);
          output.Push(Job{job.seq, kErrorTok, true});
          continue;
        }
        const Buffer* incoming =
            i > 0 ? runtime[i].incoming[job.seq].get() : nullptr;
        core::DecodeCache& cache = *runtime[i].caches[job.seq];
        if (i + 1 == stage_count) {
          if (job.want_head) {
            auto logits =
                arch->Logits(backend, *stage.model, cache, job.token, nullptr,
                             nullptr, nullptr, incoming);
            if (!logits) {
              failed.store(true);
              output.Push(Job{job.seq, kErrorTok, true});
              continue;
            }
            output.Push(Job{job.seq, ArgMax(*logits), true});
          } else {
            auto forward =
                arch->Forward(backend, *stage.model, cache, job.token, nullptr,
                              nullptr, nullptr, incoming);
            if (!forward) {
              failed.store(true);
              output.Push(Job{job.seq, kErrorTok, true});
              continue;
            }
            output.Push(Job{job.seq, kNoToken, false});
          }
          continue;
        }
        std::vector<std::size_t> capture_layers{stage.boundary_layer};
        std::vector<Buffer*> capture{runtime[i].outgoing[job.seq].get()};
        auto forward =
            arch->Forward(backend, *stage.model, cache, job.token, nullptr,
                          &capture_layers, &capture, incoming);
        if (!forward) {
          failed.store(true);
          output.Push(Job{job.seq, kErrorTok, true});
          continue;
        }
        auto copied = CopyAcrossDevices(*runtime[i].outgoing[job.seq], 0,
                                        *runtime[i + 1].incoming[job.seq], 0,
                                        p.hidden_dim * 4);
        if (!copied) {
          failed.store(true);
          output.Push(Job{job.seq, kErrorTok, true});
          continue;
        }
        runtime[i + 1].input.Push(Job{job.seq, 0, job.want_head});
      }
    });
  }
  // Driver: feed each sequence's next token and retire it at a stop token or
  // its budget. A sequence has at most one token in flight, so the stages
  // overlap across sequences.
  struct Sequence {
    std::vector<std::uint32_t> prompt;
    std::size_t index = 0;
    std::size_t budget = 0;
    std::vector<std::uint32_t> produced;
    std::uint32_t feed = 0;
    bool has_feed = false;
    bool in_flight = false;
    bool done = false;
  };
  std::vector<Sequence> sequences(count);
  for (std::size_t s = 0; s < count; ++s) {
    sequences[s].prompt = prompts[s];
    const std::size_t prompt_size = prompts[s].size();
    sequences[s].budget =
        max_tokens == 0 ? p.context - prompt_size : max_tokens;
    sequences[s].done = sequences[s].budget == 0;
  }
  bool running = true;
  StatusCode error = StatusCode::Ok;
  while (running) {
    running = false;
    for (std::size_t s = 0; s < count; ++s) {
      Sequence& sequence = sequences[s];
      if (sequence.done || sequence.in_flight) {
        running = running || !sequence.done;
        continue;
      }
      running = true;
      Job job{s, 0, false};
      if (sequence.index < sequence.prompt.size()) {
        job.token = sequence.prompt[sequence.index];
        job.want_head = (sequence.index + 1 == sequence.prompt.size());
        ++sequence.index;
      } else if (sequence.has_feed) {
        job.token = sequence.feed;
        job.want_head = true;
        sequence.has_feed = false;
      } else {
        continue;
      }
      sequence.in_flight = true;
      runtime[0].input.Push(job);
    }
    if (!running) {
      break;
    }
    const Job done_job = output.Pop();
    if (done_job.seq < count) {
      sequences[done_job.seq].in_flight = false;
    }
    if (done_job.token == kErrorTok) {
      error = StatusCode::DeviceError;
      break;
    }
    if (done_job.token == kNoToken) {
      continue;
    }
    Sequence& sequence = sequences[done_job.seq];
    if (IsStop(done_job.token, p.stops) ||
        sequence.produced.size() >= sequence.budget) {
      sequence.done = true;
      continue;
    }
    sequence.produced.push_back(done_job.token);
    if (sequence.produced.size() >= sequence.budget) {
      sequence.done = true;
    } else {
      sequence.feed = done_job.token;
      sequence.has_feed = true;
    }
  }
  // Shut the workers down and collect the outputs.
  runtime[0].input.Push(Job{kStopSeq, 0, false});
  for (std::size_t i = 0; i < stage_count; ++i) {
    runtime[i].worker.join();
  }
  if (error != StatusCode::Ok || failed.load()) {
    return std::unexpected(StatusCode::DeviceError);
  }
  std::vector<std::vector<std::uint32_t>> produced;
  produced.reserve(count);
  for (Sequence& sequence : sequences) {
    produced.push_back(std::move(sequence.produced));
  }
  return produced;
}

std::size_t Pipeline::StageCount() const { return impl_->stages.size(); }

Engine& Pipeline::StageEngine(std::size_t i) {
  return *impl_->stages.at(i).engine;
}

Model& Pipeline::StageModel(std::size_t i) {
  return *impl_->stages.at(i).model;
}

std::vector<std::pair<std::size_t, std::size_t>> Pipeline::StageRanges()
    const {
  std::vector<std::pair<std::size_t, std::size_t>> ranges;
  ranges.reserve(impl_->stages.size());
  for (const Impl::Stage& stage : impl_->stages) {
    ranges.emplace_back(stage.first_layer, stage.last_layer);
  }
  return ranges;
}

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
