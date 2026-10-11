#pragma once

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/model.hpp"
#include "tessera/tokenizer.hpp"

namespace tessera {

// One engine and its loaded model on one device, plus the lock that
// serializes that device. The pool owns runners and hands one out at a
// time; the lock is the device slot.
//
// Usage:
//   auto runner = std::make_unique<Runner>(engine, model, 0);
//   pool.Add(std::move(runner));
class Runner {
 public:
  Runner(std::unique_ptr<Engine> engine, std::unique_ptr<Model> model,
         int device_index, CalibrationConfig calibration = {});
  Runner(const Runner&) = delete;
  Runner& operator=(const Runner&) = delete;

  [[nodiscard]] Engine& engine() const;
  [[nodiscard]] Model& model() const;
  [[nodiscard]] const Tokenizer* tokenizer() const;
  [[nodiscard]] int device_index() const;
  [[nodiscard]] const CalibrationConfig& calibration() const;

 private:
  friend class RunnerPool;
  friend class RunnerLease;
  std::unique_ptr<Engine> engine_;
  std::unique_ptr<Model> model_;
  int device_index_ = 0;
  CalibrationConfig calibration_;
  // Held while a lease is active; the only lock that serializes a device.
  std::mutex mutex_;
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

  // Leases currently held, one count per runner in index order.
  [[nodiscard]] std::vector<std::size_t> Active() const;

 private:
  // Guards the runner list (Add versus Acquire/At). The list is immutable
  // once serving starts.
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<Runner>> runners_;
};

}  // namespace tessera
