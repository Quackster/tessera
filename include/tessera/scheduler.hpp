#pragma once

#include <cstddef>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>

#include "tessera/engine.hpp"
#include "tessera/types.hpp"

namespace tessera {

class Model;

// Bound on admitted jobs when SchedulerOptions leaves it zero.
constexpr std::size_t kDefaultMaxConcurrent = 4;

// Options for one shared device slot (see Scheduler).
struct SchedulerOptions {
  // Admitted jobs bound; extra Submit calls queue FIFO. A tunable.
  std::size_t max_concurrent = kDefaultMaxConcurrent;
};

// True when Scheduler::Submit can drive this request: plain greedy or
// sampled text with no thinking budget on the request (thinking stays
// on the exclusive RunnerLease path). A speculative strategy on the
// engine does not block sharing: the scheduler never touches it and
// runs the turn as plain greedy, which equals the speculative output.
// Prompt-independent, so the serving layer routes before it tokenizes.
//
// Usage:
//   const bool shared = CanSchedule(pool.At(0).engine(), options);
[[nodiscard]] bool CanSchedule(const Engine& engine,
                               const GenerateOptions& options);

// One device's cooperative decode scheduler. HTTP workers Submit from
// any thread and block until their turn completes; one dispatcher
// thread drives the borrowed engine in Sarathi rounds (chunked
// prefills interleaved with decode steps), so the backend sees a
// single launch stream. Each job's tokens equal a separate greedy or
// sampled Generate with the same seed. Safe for concurrent use;
// device work is confined to the dispatcher thread. The engine, the
// model and the device lock outlive this (the Runner owns all three
// and destroys this first).
//
// Usage:
//   Scheduler shared(engine, model, device_mutex);
//   auto outcome = shared.Submit(options, on_token);
class Scheduler {
 public:
  Scheduler(Engine& engine, Model& model, std::mutex& device_mutex,
            SchedulerOptions options = {});
  ~Scheduler();
  Scheduler(const Scheduler&) = delete;
  Scheduler& operator=(const Scheduler&) = delete;

  // Enqueue and run to completion, invoking on_token and the options
  // prefill hook on the caller thread. Stops early when on_token
  // returns false or cancelled reports true (Aborted outcome with the
  // tokens produced so far). InvalidArgument for a bad prompt or
  // sampling setup, UnsupportedFeature for a thinking budget,
  // OutOfMemory when one turn alone exceeds the memory-bound context.
  // A turn on a strategy-backed engine runs as plain greedy (equal
  // output, no draft speedup) so contended speculative serves share.
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const;

  // Admitted (prefilling or decoding) and queued jobs, for
  // least-loaded routing and /metrics. Informational only.
  [[nodiscard]] std::size_t Active() const;
  [[nodiscard]] std::size_t Pending() const;

 private:
  struct Impl;
  mutable std::unique_ptr<Impl> impl_;
};

}  // namespace tessera
