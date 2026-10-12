#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/scheduler.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

namespace tessera {

// One engine and its loaded model on one device, plus the lock that
// serializes that device and the scheduler that shares it. The pool
// owns runners; Submit time-slices turns on the scheduler while
// Acquire still hands out the whole device for paths the scheduler
// does not drive (speculation, thinking budgets, tools). The runner
// also carries the KV cache storage type every turn on it decodes
// with; the serve layer reads it onto each request because the pool
// outlives any single turn's options.
//
// Usage:
//   auto runner = std::make_unique<Runner>(engine, model, 0);
//   pool.Add(std::move(runner));
class Runner {
 public:
  Runner(std::unique_ptr<Engine> engine, std::unique_ptr<Model> model,
         int device_index, CalibrationConfig calibration = {},
         KvCacheType kv_type = KvCacheType::F32);
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  [[nodiscard]] Engine& engine() const;
  [[nodiscard]] Model& model() const;
  [[nodiscard]] const Tokenizer* tokenizer() const;
  [[nodiscard]] int device_index() const;
  [[nodiscard]] const CalibrationConfig& calibration() const;
  // Storage type of the full-attention KV cache for turns on this
  // runner (F32 unless the launcher passed --kv-f16|--kv-q8|--kv-q4).
  //
  // Usage:
  //   GenerateOptions options;
  //   options.kv_type = pool.At(0).kv_type();
  [[nodiscard]] KvCacheType kv_type() const;

  // Enqueue a turn on this device's scheduler and run it to
  // completion, streaming on the caller thread. Applies this
  // runner's calibration first. Safe for concurrent use; at most
  // SchedulerOptions::max_concurrent turns decode at once.
  //
  // Usage:
  //   auto outcome = runner.Submit(options, on_token);
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const;

  // Scheduler jobs admitted and queued (leases excluded). For
  // least-loaded routing and /metrics. Informational only.
  [[nodiscard]] std::size_t SchedulerActive() const;
  [[nodiscard]] std::size_t SchedulerPending() const;

 private:
  friend class RunnerPool;
  friend class RunnerLease;
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<Model> model_;
  int device_index_ = 0;
  CalibrationConfig calibration_;
  KvCacheType kv_type_ = KvCacheType::F32;
  // Held while a lease is active and around every scheduler round;
  // the only lock that serializes a device. Declared before the
  // scheduler so it outlives the dispatcher thread that borrows it.
  std::mutex mutex_;
  // Destroyed before the engine, the model and the device lock, so
  // its dispatcher joins while everything it borrows is still alive.
  std::unique_ptr<Scheduler> scheduler_;
  // Leases currently held, for /metrics. Informational only.
  std::atomic<std::size_t> active_{0};
};

class RunnerPool;

// RAII hold on one runner: the runner's device lock stays held until the
// lease is released or destroyed. Use the accessors to reach the engine
// and model without touching the pool.
//
// Usage:
//   if (auto lease = pool.Acquire([&] { return writer.IsPeerGone(); })) {
//     lease->engine().GenerateStreaming(lease->model(), options, on_token);
//   }
class RunnerLease {
 public:
  RunnerLease() = default;
  RunnerLease(RunnerLease&& other) noexcept;
  RunnerLease& operator=(RunnerLease&& other) noexcept;
  ~RunnerLease();
  RunnerLease(const RunnerLease&) = delete;
  RunnerLease& operator=(const RunnerLease&) = delete;

  [[nodiscard]] bool valid() const { return runner_ != nullptr; }
  [[nodiscard]] Runner& runner() const;
  [[nodiscard]] Engine& engine() const;
  [[nodiscard]] Model& model() const;
  [[nodiscard]] const Tokenizer* tokenizer() const;
  // End the hold early (idempotent). The destructor calls this.
  void Release();
  // Briefly release the device so scheduler turns progress while a
  // long exclusive turn runs, then hold it again. Call only between
  // backend operations (the serve on_token hook); the strategy and
  // cache state sit idle while released. No-op when not held.
  //
  // Usage:
  //   lease.YieldFor(std::chrono::milliseconds(1));
  void YieldFor(std::chrono::milliseconds quantum);

 private:
  friend class RunnerPool;
  RunnerLease(Runner& runner, std::unique_lock<std::mutex> lock);
  Runner* runner_ = nullptr;
  std::unique_lock<std::mutex> lock_;
};

// A pool of runners, one per device. Data-parallel serving routes each
// request to an idle runner (least-loaded: any free device wins).
//
// Usage:
//   RunnerPool pool;
//   pool.Add(std::make_unique<Runner>(std::move(engine), std::move(model), 0));
//   if (auto lease = pool.Acquire()) { /* use lease->model() */ }
class RunnerPool {
 public:
  RunnerPool() = default;
  RunnerPool(const RunnerPool&) = delete;
  RunnerPool& operator=(const RunnerPool&) = delete;

  // Add a runner. Call before serving; the pool takes ownership.
  void Add(std::unique_ptr<Runner> runner);

  [[nodiscard]] std::size_t size() const;
  [[nodiscard]] bool empty() const;
  // The runner at `index` (valid for index < size). Used for metadata
  // and diagnostics, not for generation.
  [[nodiscard]] Runner& At(std::size_t index) const;

  // Block until a runner is free, then return a lease on it. Returns
  // nullopt when `cancelled` says to give up and when the pool is empty.
  // `cancelled` is polled between attempts; it must be safe to call from
  // this thread.
  [[nodiscard]] std::optional<RunnerLease> Acquire(
      const std::function<bool()>& cancelled = {});

  // One non-blocking pass over the runners: a lease on the first free
  // device, or nullopt when every device is busy (or the pool is
  // empty). Used by Acquire's queue loop; callers that must not wait
  // take the lease when free and fall back otherwise.
  //
  // Usage:
  //   if (auto fast = pool.TryAcquire()) { /* exclusive */ }
  [[nodiscard]] std::optional<RunnerLease> TryAcquire();

  // Enqueue a turn on the least-loaded runner (fewest scheduler jobs;
  // ties go to the first) and run it to completion. Extra callers wait
  // in arrival order instead of failing, so parallel windows and
  // parallel API clients each get their turn. Safe for concurrent use.
  //
  // Usage:
  //   auto outcome = pool.SubmitLeastLoaded(options, on_token);
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> SubmitLeastLoaded(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const;

  // Scheduler jobs plus leases currently held, one count per runner
  // in index order.
  [[nodiscard]] std::vector<std::size_t> Active() const;

 private:
  // Guards the runner list (Add versus Acquire/At). The list is immutable
  // once serving starts.
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Runner>> runners_;
};

}  // namespace tessera
