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
#include "tessera/pipeline.hpp"
#include "tessera/scheduler.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

namespace tessera {

// One generation backend for the serve pool. A runner owns its devices;
// Submit runs one turn to completion, streaming on the caller thread.
// Two backends implement it: SingleDeviceRunner (one engine, one model,
// one device, shared by the scheduler) and PipelineRunner (one model
// layer-split across several devices). The serve layer drives only this
// interface, so a split model needs no separate HTTP path.
//
// A runner carries the KV cache storage type every turn on it decodes
// with; the serve layer reads it onto each request because the pool
// outlives any single turn's options.
//
// Usage:
//   auto runner = std::make_unique<SingleDeviceRunner>(engine, model, 0);
//   pool.Add(std::move(runner));
class Runner {
 public:
  virtual ~Runner() = default;
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  // The engine and model the serve layer reads for metadata: diagnostics,
  // the tokenizer, chat rendering and the context length. A split runner
  // returns its front stage, which carries the full model config.
  [[nodiscard]] virtual Engine& engine() const = 0;
  [[nodiscard]] virtual Model& model() const = 0;
  [[nodiscard]] virtual const Tokenizer* tokenizer() const = 0;
  [[nodiscard]] virtual int device_index() const = 0;
  [[nodiscard]] virtual const CalibrationConfig& calibration() const = 0;
  // Storage type of the full-attention KV cache for turns on this
  // runner (F32 unless the launcher passed --kv-f16|--kv-q8|--kv-q4).
  //
  // Usage:
  //   GenerateOptions options;
  //   options.kv_type = pool.At(0).kv_type();
  [[nodiscard]] virtual KvCacheType kv_type() const = 0;
  // True when one model spans several devices (the pipeline runner). The
  // serve layer refuses tools, images and auto-title on such a runner,
  // because those paths run a single device's engine directly.
  [[nodiscard]] virtual bool is_split() const = 0;
  // True when Submit alone can drive this request (the shared path). The
  // single-device runner defers to CanSchedule(engine, ...); the split
  // runner drives everything through Submit, so it always returns true.
  [[nodiscard]] virtual bool CanSchedule(
      const GenerateOptions& options) const = 0;

  // Run one turn to completion, streaming on the caller thread. Applies
  // this runner's calibration first. Safe for concurrent use: the
  // single-device runner time-slices on its scheduler, the split runner
  // serializes on its device lock.
  //
  // Usage:
  //   auto outcome = runner.Submit(options, on_token);
  [[nodiscard]] virtual std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const = 0;

  // Jobs admitted and queued (leases excluded). For least-loaded routing
  // and /metrics. Informational only; zero for a split runner.
  [[nodiscard]] virtual std::size_t SchedulerActive() const = 0;
  [[nodiscard]] virtual std::size_t SchedulerPending() const = 0;

 protected:
  Runner() = default;
  friend class RunnerPool;
  friend class RunnerLease;
  // Held while a lease is active and around every exclusive turn; the
  // only lock that serializes a runner's devices.
  mutable std::mutex mutex_;
  // Leases currently held or split turns running, for /metrics.
  // Informational only; mutable so a const Submit can update it.
  mutable std::atomic<std::size_t> active_{0};
};

// One engine and its loaded model on one device, shared by the scheduler
// and handed out whole for paths the scheduler does not drive (thinking
// budgets, tools). The pool owns runners.
//
// Usage:
//   auto runner = std::make_unique<SingleDeviceRunner>(engine, model, 0);
//   pool.Add(std::move(runner));
class SingleDeviceRunner : public Runner {
 public:
  SingleDeviceRunner(std::unique_ptr<Engine> engine,
                     std::unique_ptr<Model> model, int device_index,
                     CalibrationConfig calibration = {},
                     KvCacheType kv_type = KvCacheType::F32);

  [[nodiscard]] Engine& engine() const override;
  [[nodiscard]] Model& model() const override;
  [[nodiscard]] const Tokenizer* tokenizer() const override;
  [[nodiscard]] int device_index() const override;
  [[nodiscard]] const CalibrationConfig& calibration() const override;
  [[nodiscard]] KvCacheType kv_type() const override;
  [[nodiscard]] bool is_split() const override;
  [[nodiscard]] bool CanSchedule(
      const GenerateOptions& options) const override;
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const override;
  [[nodiscard]] std::size_t SchedulerActive() const override;
  [[nodiscard]] std::size_t SchedulerPending() const override;

 private:
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<Model> model_;
  int device_index_ = 0;
  CalibrationConfig calibration_;
  KvCacheType kv_type_ = KvCacheType::F32;
  // Destroyed before the engine, the model and the device lock, so its
  // dispatcher joins while everything it borrows is still alive.
  std::unique_ptr<Scheduler> scheduler_;
};

// One model layer-split across several devices, driven by the pipeline
// decode target. Submit serializes turns on the runner's device lock and
// runs the whole split generation, so the shared serve path works
// unchanged. The front stage backs engine()/model() for metadata.
//
// Usage:
//   auto runner = std::make_unique<PipelineRunner>(pipeline, 0);
//   pool.Add(std::move(runner));
class PipelineRunner : public Runner {
 public:
  PipelineRunner(std::unique_ptr<Pipeline> pipeline, int device_index,
                 CalibrationConfig calibration = {},
                 KvCacheType kv_type = KvCacheType::F32);

  [[nodiscard]] Engine& engine() const override;
  [[nodiscard]] Model& model() const override;
  [[nodiscard]] const Tokenizer* tokenizer() const override;
  [[nodiscard]] int device_index() const override;
  [[nodiscard]] const CalibrationConfig& calibration() const override;
  [[nodiscard]] KvCacheType kv_type() const override;
  [[nodiscard]] bool is_split() const override;
  [[nodiscard]] bool CanSchedule(
      const GenerateOptions& options) const override;
  [[nodiscard]] std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled = {}) const override;
  [[nodiscard]] std::size_t SchedulerActive() const override;
  [[nodiscard]] std::size_t SchedulerPending() const override;

 private:
  std::unique_ptr<Pipeline> pipeline_;
  Engine* facade_engine_ = nullptr;
  Model* facade_model_ = nullptr;
  int device_index_ = 0;
  CalibrationConfig calibration_;
  KvCacheType kv_type_ = KvCacheType::F32;
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
