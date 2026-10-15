#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <expected>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "core/pipeline_impl.hpp"
#include "tessera/architecture.hpp"
#include "tessera/backend.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/types.hpp"

// The overlapped multi-sequence pipeline path: each stage runs on its own
// thread, so while one stage processes one sequence the next stage processes
// another. Each sequence's tokens equal a separate Generate. The
// single-sequence path lives in pipeline.cpp.

namespace tessera {

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
            output.Push(Job{job.seq, pipeline_internal::ArgMax(*logits), true});
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
    if (pipeline_internal::IsStop(done_job.token, p.stops) ||
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

}  // namespace tessera
