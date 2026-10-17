#include "tessera/scheduler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "core/decode.hpp"
#include "core/decode_internal.hpp"
#include "core/generate_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/model.hpp"

namespace tessera {
namespace {

// Queued events bound per job: backpressure while a client pauses.
constexpr std::size_t kMaxQueuedEvents = 64;
// Headroom kept free when sizing the memory-bound context (the
// calibrate reserve, so the scheduler and the report agree).
constexpr std::uint64_t kSchedulerReserveBytes =
    2ull * 1024 * 1024 * 1024;

bool CanScheduleRequest(const Engine& engine,
                        const GenerateOptions& options) {
  // The scheduler drives plain greedy/sampled turns without touching
  // the attached strategy, so a strategy-backed engine still shares:
  // the shared output equals the speculative output, only slower per
  // turn. Thinking budgets stay exclusive (no force-close here).
  (void)engine;
  return options.max_thinking_tokens == 0;
}

// One worker-visible event: a token, a prefill report, or the end.
struct Event {
  enum class Kind { kToken, kProgress, kDone };
  Kind kind = Kind::kToken;
  std::uint32_t token = 0;
  std::size_t done = 0;
  std::size_t total = 0;
  GenerateOutcome outcome;
  StatusCode status = StatusCode::Ok;
};

// One admitted turn: its own cache and decode state (as in
// Engine::GenerateBatch), plus the worker handoff queue.
struct Job {
  GenerateOptions options;
  std::vector<std::uint32_t> prompt;
  std::size_t prefilled = 0;
  std::size_t chunk = 0;
  std::vector<std::uint32_t> history;
  std::vector<std::uint32_t> produced;
  std::vector<std::uint32_t> stops;
  std::size_t budget = 0;
  std::uint32_t next = 0;
  bool has_next = false;
  core::DecodeCache cache;
  std::mt19937_64 rng;
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<Event> events;
  std::atomic<bool> abort{false};
  std::atomic<bool> cancel{false};
  bool admitted = false;
  bool decoding = false;
  std::atomic<bool> finished{false};
  GenerateOutcome outcome;
  StatusCode status = StatusCode::Ok;
  std::chrono::steady_clock::time_point admitted_at;
  std::chrono::steady_clock::time_point decode_at;
  // Worst-case KV rows this turn reserves for admission.
  [[nodiscard]] std::size_t Rows() const {
    return prompt.size() + budget;
  }
};

double MsBetween(std::chrono::steady_clock::time_point from,
                 std::chrono::steady_clock::time_point to) {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::microseconds>(to - from)
                 .count()) /
         1000.0;
}

}  // namespace

bool CanSchedule(const Engine& engine, const GenerateOptions& options) {
  return CanScheduleRequest(engine, options);
}

struct Scheduler::Impl {
  Engine& engine;
  Model& model;
  std::mutex& device;
  SchedulerOptions options;
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::deque<std::shared_ptr<Job>> pending;
  std::vector<std::shared_ptr<Job>> active;
  // Finished jobs awaiting destruction. A worker hands its reference
  // here instead of dropping it: every Buffer dies on the dispatcher
  // thread under the device lock, never on a worker sharing the device.
  std::vector<std::shared_ptr<Job>> graveyard;
  std::thread thread;
  bool started = false;
  bool stopping = false;

  Impl(Engine& engine_in, Model& model_in, std::mutex& device_in,
       SchedulerOptions options_in)
      : engine(engine_in),
        model(model_in),
        device(device_in),
        options(options_in) {}

  ~Impl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      for (const auto& job : pending) {
        job->cancel.store(true);
      }
      for (const auto& job : active) {
        job->abort.store(true);
      }
    }
    cv.notify_all();
    // Wake every worker so no Submit outlives the dispatcher: each
    // in-flight turn ends Aborted below before the thread joins.
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (const auto& job : pending) {
        FinishLocked(job, FinishReason::Aborted, StatusCode::Ok);
      }
      for (const auto& job : active) {
        if (!job->finished.load()) {
          FinishLocked(job, FinishReason::Aborted, StatusCode::Ok);
        }
      }
    }
    if (thread.joinable()) {
      thread.join();
    }
    // The owner joined every Submit caller first, so each finished job
    // sits in the graveyard exactly once. Drop every reference here on
    // this thread under the device lock: no Buffer outlives the device
    // it was allocated on, on any thread.
    {
      std::unique_lock<std::mutex> device_lock(device);
      std::lock_guard<std::mutex> lock(mutex);
      pending.clear();
      active.clear();
      graveyard.clear();
    }
  }

  void EnsureStartedLocked() {
    if (!started) {
      started = true;
      thread = std::thread(&Impl::Run, this);
    }
  }

  // Memory-bound turn count for this KV type; 0 when unsized (no
  // config or no memory report), which admits without a gate.
  std::size_t MemoryBound(KvCacheType type) {
    auto config = model.Config();
    if (!config) {
      return 0;
    }
    std::uint64_t weights = 0;
    for (const auto& weight : model.Weights()) {
      weights += weight.device->Size();
    }
    const DeviceMemoryInfo memory = engine.Owner().MemoryInfo();
    if (memory.free_bytes == 0 && memory.total_bytes == 0) {
      return 0;
    }
    std::uint64_t available = 0;
    if (memory.free_bytes > 0) {
      available = memory.free_bytes;
    } else if (memory.total_bytes > weights) {
      available = memory.total_bytes - weights;
    }
    if (available <= kSchedulerReserveBytes) {
      return 0;
    }
    return MaxContextForKv(available - kSchedulerReserveBytes, *config,
                           type);
  }

  bool HasUnfinishedLocked() const {
    for (const auto& job : active) {
      if (!job->finished.load()) {
        return true;
      }
    }
    return false;
  }

  // Rows the admitted turns may still consume: the unproduced
  // budget each. Prompt and produced rows already sit in the cache
  // (and out of free memory), so charging full rows would double
  // count them and ratchet every later turn out: a second unlimited
  // turn would wait behind the first one's whole generation instead
  // of time-slicing beside it. Produced never passes budget (the
  // decode loop ends the turn there), so the subtraction is exact.
  std::size_t ReservedLocked() const {
    std::size_t rows = 0;
    for (const auto& job : active) {
      if (!job->finished.load()) {
        rows += job->budget - job->produced.size();
      }
    }
    return rows;
  }

  void PushEvent(const std::shared_ptr<Job>& job, Event event) {
    std::lock_guard<std::mutex> lock(job->mutex);
    job->events.push_back(std::move(event));
    job->cv.notify_one();
  }

  void FinishLocked(const std::shared_ptr<Job>& job, FinishReason reason,
                    StatusCode status) {
    const auto now = std::chrono::steady_clock::now();
    job->outcome.produced = job->produced.size();
    job->outcome.reason = reason;
    job->outcome.prompt_tokens = job->prompt.size();
    job->outcome.prefill_ms =
        MsBetween(job->admitted_at, job->decoding ? job->decode_at : now);
    job->outcome.decode_ms =
        job->decoding ? MsBetween(job->decode_at, now) : 0.0;
    job->status = status;
    job->finished.store(true);
    Event done;
    done.kind = Event::Kind::kDone;
    done.outcome = job->outcome;
    done.status = status;
    {
      std::lock_guard<std::mutex> lock(job->mutex);
      job->events.push_back(std::move(done));
    }
    job->cv.notify_one();
  }

  bool QueueHasRoom(const std::shared_ptr<Job>& job) {
    std::lock_guard<std::mutex> lock(job->mutex);
    return job->events.size() < kMaxQueuedEvents;
  }

  void RoundPrefill(const std::vector<std::shared_ptr<Job>>& round) {
    Backend& backend = engine.Owner();
    for (const auto& job : round) {
      if (job->finished.load() || job->abort.load() ||
          job->prefilled >= job->prompt.size() || !QueueHasRoom(job)) {
        continue;
      }
      const std::size_t step =
          job->chunk > 0 ? job->chunk
                         : job->prompt.size() - job->prefilled;
      const std::size_t len =
          std::min(step, job->prompt.size() - job->prefilled);
      auto logits = core::PrefillTokens(
          backend, model, job->cache,
          std::span<const std::uint32_t>(job->prompt)
              .subspan(job->prefilled, len),
          nullptr, nullptr, job->chunk, nullptr);
      if (!logits) {
        engine.Diagnostics().Warn(
            "scheduler",
            std::string("prefill failed: ") + std::string(ToString(logits.error())));
        FinishLocked(job, FinishReason::Aborted, logits.error());
        continue;
      }
      job->prefilled += len;
      Event progress;
      progress.kind = Event::Kind::kProgress;
      progress.done = job->prefilled;
      progress.total = job->prompt.size();
      PushEvent(job, std::move(progress));
      if (job->prefilled >= job->prompt.size()) {
        job->history = job->prompt;
        job->next = PickToken(*logits, job->options.sample,
                              job->options.sampling, job->history, job->rng);
        job->has_next = true;
        job->decoding = true;
        job->decode_at = std::chrono::steady_clock::now();
      }
    }
  }

  void FinishDecode(const std::shared_ptr<Job>& job,
                    std::vector<std::shared_ptr<Job>>& greedy_jobs,
                    std::vector<std::uint32_t>& greedy_tokens) {
    if (job->abort.load()) {
      FinishLocked(job, FinishReason::Aborted, StatusCode::Ok);
      return;
    }
    if (core::detail::IsStopToken(job->next, job->stops)) {
      FinishLocked(job, FinishReason::Stop, StatusCode::Ok);
      return;
    }
    if (job->produced.size() >= job->budget) {
      FinishLocked(job, FinishReason::Length, StatusCode::Ok);
      return;
    }
    if (!QueueHasRoom(job)) {
      return;
    }
    Event token;
    token.kind = Event::Kind::kToken;
    token.token = job->next;
    PushEvent(job, std::move(token));
    job->produced.push_back(job->next);
    job->history.push_back(job->next);
    if (job->produced.size() >= job->budget) {
      FinishLocked(job, FinishReason::Length, StatusCode::Ok);
      return;
    }
    if (job->options.sample) {
      auto logits = core::DecodeLogits(engine.Owner(), model, job->cache,
                                       job->next);
      if (!logits) {
        engine.Diagnostics().Warn(
            "scheduler",
            std::string("decode failed: ") + std::string(ToString(logits.error())));
        FinishLocked(job, FinishReason::Aborted, logits.error());
        return;
      }
      job->next = PickToken(*logits, true, job->options.sampling,
                            job->history, job->rng);
      return;
    }
    greedy_jobs.push_back(job);
    greedy_tokens.push_back(job->next);
  }

  void RoundDecode(const std::vector<std::shared_ptr<Job>>& round) {
    std::vector<std::shared_ptr<Job>> greedy_jobs;
    std::vector<std::uint32_t> greedy_tokens;
    for (const auto& job : round) {
      if (job->finished.load() || !job->has_next) {
        continue;
      }
      FinishDecode(job, greedy_jobs, greedy_tokens);
    }
    if (greedy_jobs.empty()) {
      return;
    }
    std::vector<core::DecodeCache*> caches;
    caches.reserve(greedy_jobs.size());
    for (const auto& job : greedy_jobs) {
      caches.push_back(&job->cache);
    }
    auto advanced = AdvanceGreedyCaches(engine.Owner(), model,
                                       engine.Diagnostics(), caches,
                                       greedy_tokens);
    if (!advanced) {
      for (const auto& job : greedy_jobs) {
        FinishLocked(job, FinishReason::Aborted, advanced.error());
      }
      return;
    }
    for (std::size_t k = 0; k < greedy_jobs.size(); ++k) {
      greedy_jobs[k]->next = (*advanced)[k];
    }
  }

  void Run() {
    for (;;) {
      std::vector<std::shared_ptr<Job>> round;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] {
          return stopping || !pending.empty() || HasUnfinishedLocked();
        });
        if (stopping) {
          return;
        }
        active.erase(
            std::remove_if(active.begin(), active.end(),
                           [](const std::shared_ptr<Job>& job) {
                             return job->finished.load();
                           }),
            active.end());
        const std::size_t maxc = options.max_concurrent == 0
                                     ? kDefaultMaxConcurrent
                                     : options.max_concurrent;
        while (active.size() < maxc && !pending.empty()) {
          auto job = pending.front();
          if (job->cancel.load()) {
            pending.pop_front();
            FinishLocked(job, FinishReason::Aborted, StatusCode::Ok);
            continue;
          }
          const std::size_t bound = MemoryBound(job->cache.kv_type);
          if (bound > 0 && job->Rows() > bound &&
              ReservedLocked() == 0) {
            pending.pop_front();
            FinishLocked(job, FinishReason::Aborted,
                         StatusCode::OutOfMemory);
            continue;
          }
          if (bound > 0 && ReservedLocked() + job->Rows() > bound) {
            break;
          }
          pending.pop_front();
          job->admitted = true;
          job->admitted_at = std::chrono::steady_clock::now();
          active.push_back(std::move(job));
        }
        for (const auto& job : active) {
          if (!job->finished.load()) {
            round.push_back(job);
          }
        }
      }
      if (round.empty()) {
        continue;
      }
      // Retire before driving: finished turns and handed-off turns die
      // here on the device thread, so no Buffer destructor ever runs on
      // a worker while this thread uses the backend. Frees device memory
      // before the round admits more work below.
      {
        std::unique_lock<std::mutex> device_lock(device);
        std::vector<std::shared_ptr<Job>> dead;
        {
          std::lock_guard<std::mutex> lock(mutex);
          for (auto it = active.begin(); it != active.end();) {
            if ((*it)->finished.load()) {
              dead.push_back(std::move(*it));
              it = active.erase(it);
            } else {
              ++it;
            }
          }
          for (auto& job : graveyard) {
            dead.push_back(std::move(job));
          }
          graveyard.clear();
        }
        RoundPrefill(round);
        RoundDecode(round);
      }
      {
        std::lock_guard<std::mutex> lock(mutex);
        cv.notify_all();
      }
    }
  }
};

Scheduler::Scheduler(Engine& engine, Model& model, std::mutex& device_mutex,
                     SchedulerOptions options)
    : impl_(std::make_unique<Impl>(engine, model, device_mutex, options)) {}

Scheduler::~Scheduler() = default;

std::expected<GenerateOutcome, StatusCode> Scheduler::Submit(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  Engine& engine = impl_->engine;
  Model& model = impl_->model;
  if (options.sample &&
      ValidateSamplingOptions(options.sampling, engine.Diagnostics()) !=
          StatusCode::Ok) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  if (!CanScheduleRequest(engine, options)) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  std::vector<std::uint32_t> prompt = options.prompt_tokens;
  if (prompt.empty()) {
    prompt.push_back(options.first_token);
  }
  if (prompt.size() > model.MaxContextLength()) {
    engine.Diagnostics().Warn(
        "scheduler", std::string("prompt of ") +
                         std::to_string(prompt.size()) +
                         " token(s) exceeds the " +
                         std::to_string(model.MaxContextLength()) +
                         " token context; reject without running prefill");
    return std::unexpected(StatusCode::InvalidArgument);
  }
  const std::size_t budget =
      model.EffectiveMaxTokens(prompt.size(), options.max_completion_tokens);
  if (budget == 0) {
    return GenerateOutcome{};
  }
  auto job = std::make_shared<Job>();
  job->options = options;
  job->prompt = std::move(prompt);
  job->budget = budget;
  job->stops = StopSet(model, options);
  job->chunk = ResolvePrefillChunkTokens(engine.PrefillChunkTokens(),
                                        options.prefill_chunk_tokens,
                                        model.MaxContextLength());
  job->cache.kv_type = options.kv_type;
  job->cache.tuning.mxfp4_split_target = options.mxfp4_split_target;
  job->cache.tuning.mxfp4_split_cap = options.mxfp4_split_cap;
  job->cache.tuning.attention_split = options.attention_split;
  job->rng.seed(options.seed);
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->pending.push_back(job);
    impl_->EnsureStartedLocked();
  }
  impl_->cv.notify_one();
  for (;;) {
    std::unique_lock<std::mutex> lock(job->mutex);
    job->cv.wait(lock, [&] { return !job->events.empty(); });
    while (!job->events.empty()) {
      Event event = std::move(job->events.front());
      job->events.pop_front();
      lock.unlock();
      if (event.kind == Event::Kind::kToken) {
        if (!on_token(event.token)) {
          job->abort.store(true);
          impl_->cv.notify_one();
        }
      } else if (event.kind == Event::Kind::kProgress) {
        if (job->options.prefill_progress) {
          job->options.prefill_progress(event.done, event.total);
        }
      } else {
        std::expected<GenerateOutcome, StatusCode> result =
            event.status == StatusCode::Ok
                ? std::expected<GenerateOutcome, StatusCode>(event.outcome)
                : std::unexpected(event.status);
        // Hand the turn to the dispatcher for destruction instead of
        // dropping it here: its Buffers must die on the device thread
        // under the device lock, never on this worker while the device
        // is shared.
        {
          std::lock_guard<std::mutex> ilock(impl_->mutex);
          impl_->graveyard.push_back(std::move(job));
        }
        impl_->cv.notify_one();
        return result;
      }
      lock.lock();
    }
    lock.unlock();
    if (cancelled && cancelled()) {
      job->cancel.store(true);
      job->abort.store(true);
      impl_->cv.notify_one();
    }
  }
}

std::size_t Scheduler::Active() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  std::size_t count = 0;
  for (const auto& job : impl_->active) {
    if (!job->finished.load()) {
      ++count;
    }
  }
  return count;
}

std::size_t Scheduler::Pending() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->pending.size();
}

}  // namespace tessera
