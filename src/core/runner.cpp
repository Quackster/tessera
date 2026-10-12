#include "tessera/runner.hpp"

#include <chrono>
#include <thread>
#include <utility>

namespace tessera {

namespace {

// Poll interval while queued for a device: short enough to start
// promptly, long enough to stay out of the way. Matches the single-device
// queue's former cadence.
constexpr int kAcquirePollMs = 10;

}  // namespace

Runner::Runner(std::unique_ptr<Engine> engine, std::unique_ptr<Model> model,
               int device_index, CalibrationConfig calibration)
    : engine_(std::move(engine)),
      model_(std::move(model)),
      device_index_(device_index),
      calibration_(std::move(calibration)) {
  scheduler_ =
      std::make_unique<Scheduler>(*engine_, *model_, mutex_, SchedulerOptions{});
}

Engine& Runner::engine() const { return *engine_; }

Model& Runner::model() const { return *model_; }

const Tokenizer* Runner::tokenizer() const { return model_->GetTokenizer(); }

int Runner::device_index() const { return device_index_; }

const CalibrationConfig& Runner::calibration() const { return calibration_; }

std::expected<GenerateOutcome, StatusCode> Runner::Submit(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  GenerateOptions tuned = options;
  ApplyToGenerateOptions(calibration_, tuned);
  return scheduler_->Submit(tuned, on_token, cancelled);
}

std::size_t Runner::SchedulerActive() const { return scheduler_->Active(); }

std::size_t Runner::SchedulerPending() const {
  return scheduler_->Pending();
}

RunnerLease::RunnerLease(Runner& runner, std::unique_lock<std::mutex> lock)
    : runner_(&runner), lock_(std::move(lock)) {
  runner_->active_.fetch_add(1, std::memory_order_relaxed);
}

RunnerLease::RunnerLease(RunnerLease&& other) noexcept
    : runner_(other.runner_), lock_(std::move(other.lock_)) {
  other.runner_ = nullptr;
}

RunnerLease& RunnerLease::operator=(RunnerLease&& other) noexcept {
  if (this != &other) {
    Release();
    runner_ = other.runner_;
    lock_ = std::move(other.lock_);
    other.runner_ = nullptr;
  }
  return *this;
}

RunnerLease::~RunnerLease() { Release(); }

void RunnerLease::Release() {
  if (runner_ != nullptr) {
    runner_->active_.fetch_sub(1, std::memory_order_relaxed);
    runner_ = nullptr;
  }
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
}

Runner& RunnerLease::runner() const { return *runner_; }

Engine& RunnerLease::engine() const { return runner_->engine(); }

Model& RunnerLease::model() const { return runner_->model(); }

const Tokenizer* RunnerLease::tokenizer() const { return runner_->tokenizer(); }

void RunnerPool::Add(std::unique_ptr<Runner> runner) {
  std::lock_guard<std::mutex> lock(mutex_);
  runners_.push_back(std::move(runner));
}

std::size_t RunnerPool::size() const { return runners_.size(); }

bool RunnerPool::empty() const { return runners_.empty(); }

Runner& RunnerPool::At(std::size_t index) const { return *runners_[index]; }

std::optional<RunnerLease> RunnerPool::Acquire(
    const std::function<bool()>& cancelled) {
  if (runners_.empty()) {
    return std::nullopt;
  }
  for (;;) {
    // Any idle runner wins; idle runners all have zero load, so the first
    // free one is the least loaded.
    for (const auto& runner : runners_) {
      std::unique_lock<std::mutex> lock(runner->mutex_, std::try_to_lock);
      if (lock.owns_lock()) {
        return RunnerLease(*runner, std::move(lock));
      }
    }
    // A peer that disconnects while queued stops waiting instead of
    // holding its place behind a generation it never reads.
    if (cancelled && cancelled()) {
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kAcquirePollMs));
  }
}

std::expected<GenerateOutcome, StatusCode> RunnerPool::SubmitLeastLoaded(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  std::unique_lock<std::mutex> pool_lock(mutex_);
  if (runners_.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  Runner* chosen = runners_.front().get();
  std::size_t head_load =
      chosen->SchedulerActive() + chosen->SchedulerPending();
  for (const auto& runner : runners_) {
    const std::size_t load =
        runner->SchedulerActive() + runner->SchedulerPending();
    if (load < head_load) {
      head_load = load;
      chosen = runner.get();
    }
  }
  Runner& target = *chosen;
  pool_lock.unlock();
  return target.Submit(options, on_token, cancelled);
}

std::vector<std::size_t> RunnerPool::Active() const {
  std::vector<std::size_t> active;
  active.reserve(runners_.size());
  for (const auto& runner : runners_) {
    active.push_back(runner->active_.load(std::memory_order_relaxed) +
                     runner->SchedulerActive());
  }
  return active;
}

}  // namespace tessera
