#include "tessera/runner.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <span>
#include <thread>
#include <utility>

#include "tessera/context_parallel.hpp"
#include "tessera/spec_split.hpp"
#include "tessera/tensor_parallel.hpp"

namespace tessera {

namespace {

// Poll interval while queued for a device: short enough to start
// promptly, long enough to stay out of the way. Matches the single-device
// queue's former cadence.
constexpr int kAcquirePollMs = 10;

}  // namespace

SingleDeviceRunner::SingleDeviceRunner(std::unique_ptr<Engine> engine,
                                       std::unique_ptr<Model> model,
                                       int device_index,
                                       CalibrationConfig calibration,
                                       KvCacheType kv_type)
    : engine_(std::move(engine)),
      model_(std::move(model)),
      device_index_(device_index),
      calibration_(std::move(calibration)),
      kv_type_(kv_type) {
  scheduler_ = std::make_unique<Scheduler>(*engine_, *model_, mutex_,
                                           SchedulerOptions{});
}

Engine& SingleDeviceRunner::engine() const {
  return *engine_;
}

Model& SingleDeviceRunner::model() const {
  return *model_;
}

const Tokenizer* SingleDeviceRunner::tokenizer() const {
  return model_->GetTokenizer();
}

int SingleDeviceRunner::device_index() const {
  return device_index_;
}

const CalibrationConfig& SingleDeviceRunner::calibration() const {
  return calibration_;
}

KvCacheType SingleDeviceRunner::kv_type() const {
  return kv_type_;
}

bool SingleDeviceRunner::is_split() const {
  return false;
}

bool SingleDeviceRunner::CanSchedule(const GenerateOptions& options) const {
  return tessera::CanSchedule(*engine_, options);
}

std::expected<GenerateOutcome, StatusCode> SingleDeviceRunner::Submit(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  GenerateOptions tuned = options;
  ApplyToGenerateOptions(calibration_, tuned);
  return scheduler_->Submit(tuned, on_token, cancelled);
}

std::size_t SingleDeviceRunner::SchedulerActive() const {
  return scheduler_->Active();
}

std::size_t SingleDeviceRunner::SchedulerPending() const {
  return scheduler_->Pending();
}

PipelineRunner::PipelineRunner(std::unique_ptr<Pipeline> pipeline,
                               int device_index, CalibrationConfig calibration,
                               KvCacheType kv_type)
    : pipeline_(std::move(pipeline)),
      device_index_(device_index),
      calibration_(std::move(calibration)),
      kv_type_(kv_type) {
  if (pipeline_ != nullptr && pipeline_->StageCount() > 0) {
    facade_engine_ = &pipeline_->StageEngine(0);
    facade_model_ = &pipeline_->StageModel(0);
  }
}

Engine& PipelineRunner::engine() const {
  return *facade_engine_;
}

Model& PipelineRunner::model() const {
  return *facade_model_;
}

const Tokenizer* PipelineRunner::tokenizer() const {
  return facade_model_ != nullptr ? facade_model_->GetTokenizer() : nullptr;
}

int PipelineRunner::device_index() const {
  return device_index_;
}

const CalibrationConfig& PipelineRunner::calibration() const {
  return calibration_;
}

KvCacheType PipelineRunner::kv_type() const {
  return kv_type_;
}

bool PipelineRunner::is_split() const {
  return true;
}

bool PipelineRunner::CanSchedule(const GenerateOptions& options) const {
  (void)options;
  return true;
}

std::expected<GenerateOutcome, StatusCode> PipelineRunner::Submit(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  GenerateOptions tuned = options;
  ApplyToGenerateOptions(calibration_, tuned);
  // The serve on_token already watches for a dropped peer; also honor the
  // explicit cancel between tokens. The pipeline holds its own device lock
  // and interleaves concurrent turns between tokens, so Submit does not
  // serialize the whole turn; the pool bounds the in-flight turns.
  const auto gated = [&](std::uint32_t token) {
    if (cancelled && cancelled()) {
      return false;
    }
    return on_token(token);
  };
  active_.fetch_add(1, std::memory_order_relaxed);
  auto outcome =
      pipeline_->GenerateStreamingConcurrent(tuned, gated, cancelled);
  active_.fetch_sub(1, std::memory_order_relaxed);
  return outcome;
}

std::size_t PipelineRunner::SchedulerActive() const {
  return 0;
}

std::size_t PipelineRunner::SchedulerPending() const {
  return 0;
}

namespace {

// The generation face of one split layout, so one ParallelRunner drives
// the tensor-parallel, context-parallel and split-speculative layouts
// (AGENTS.md rule 2: no duplicate logic).
class SplitDriver {
 public:
  virtual ~SplitDriver() = default;
  virtual Engine& engine() = 0;
  virtual Model& model() = 0;
  virtual const Tokenizer* tokenizer() const = 0;
  virtual std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled) = 0;
};

class TensorParallelDriver final : public SplitDriver {
 public:
  explicit TensorParallelDriver(std::unique_ptr<TensorParallel> layout)
      : layout_(std::move(layout)) {}

  Engine& engine() override { return layout_->RankEngine(0); }
  Model& model() override { return layout_->RankModel(0); }
  const Tokenizer* tokenizer() const override {
    return layout_->GetTokenizer();
  }
  std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled) override {
    return layout_->GenerateStreaming(prompt, options, on_token, cancelled);
  }

 private:
  std::unique_ptr<TensorParallel> layout_;
};

class ContextParallelDriver final : public SplitDriver {
 public:
  explicit ContextParallelDriver(std::unique_ptr<ContextParallel> layout)
      : layout_(std::move(layout)) {}

  Engine& engine() override { return layout_->RankEngine(0); }
  Model& model() override { return layout_->RankModel(0); }
  const Tokenizer* tokenizer() const override {
    return layout_->GetTokenizer();
  }
  std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled) override {
    return layout_->GenerateStreaming(prompt, options, on_token, cancelled);
  }

 private:
  std::unique_ptr<ContextParallel> layout_;
};

class SpecSplitDriver final : public SplitDriver {
 public:
  explicit SpecSplitDriver(std::unique_ptr<SpecSplit> layout)
      : layout_(std::move(layout)) {}

  Engine& engine() override { return layout_->TargetEngine(); }
  Model& model() override { return layout_->TargetModel(); }
  const Tokenizer* tokenizer() const override {
    return layout_->GetTokenizer();
  }
  std::expected<GenerateOutcome, StatusCode> GenerateStreaming(
      std::span<const std::uint32_t> prompt, const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled) override {
    return layout_->GenerateStreaming(prompt, options, on_token, cancelled);
  }

 private:
  std::unique_ptr<SpecSplit> layout_;
};

// One Runner over a split layout. Submit holds the runner's device lock for
// the whole turn, because every layout carries one generation's state in
// its ranks and caches; concurrent callers queue on the lock in arrival
// order. The front device backs engine()/model() for serve metadata.
class ParallelRunner final : public Runner {
 public:
  ParallelRunner(std::unique_ptr<SplitDriver> driver, int device_index,
                 CalibrationConfig calibration, KvCacheType kv_type)
      : driver_(std::move(driver)),
        device_index_(device_index),
        calibration_(std::move(calibration)),
        kv_type_(kv_type) {}

  Engine& engine() const override { return driver_->engine(); }
  Model& model() const override { return driver_->model(); }
  const Tokenizer* tokenizer() const override { return driver_->tokenizer(); }
  int device_index() const override { return device_index_; }
  const CalibrationConfig& calibration() const override { return calibration_; }
  KvCacheType kv_type() const override { return kv_type_; }
  bool is_split() const override { return true; }
  bool CanSchedule(const GenerateOptions& options) const override {
    (void)options;
    return true;
  }
  std::expected<GenerateOutcome, StatusCode> Submit(
      const GenerateOptions& options,
      const std::function<bool(std::uint32_t)>& on_token,
      const std::function<bool()>& cancelled) const override {
    GenerateOptions tuned = options;
    ApplyToGenerateOptions(calibration_, tuned);
    std::span<const std::uint32_t> prompt = tuned.prompt_tokens;
    if (prompt.empty()) {
      prompt = std::span<const std::uint32_t>(&tuned.first_token, 1);
    }
    // The serve on_token already watches for a dropped peer; also honor
    // the explicit cancel between tokens.
    const auto gated = [&](std::uint32_t token) {
      if (cancelled && cancelled()) {
        return false;
      }
      return on_token(token);
    };
    std::lock_guard<std::mutex> lock(mutex_);
    active_.fetch_add(1, std::memory_order_relaxed);
    auto outcome = driver_->GenerateStreaming(prompt, tuned, gated, cancelled);
    active_.fetch_sub(1, std::memory_order_relaxed);
    return outcome;
  }
  std::size_t SchedulerActive() const override { return 0; }
  std::size_t SchedulerPending() const override { return 0; }

 private:
  std::unique_ptr<SplitDriver> driver_;
  int device_index_ = 0;
  CalibrationConfig calibration_;
  KvCacheType kv_type_ = KvCacheType::F32;
};

}  // namespace

std::unique_ptr<Runner> MakeTensorParallelRunner(
    std::unique_ptr<TensorParallel> layout, int device_index,
    CalibrationConfig calibration, KvCacheType kv_type) {
  return std::make_unique<ParallelRunner>(
      std::make_unique<TensorParallelDriver>(std::move(layout)), device_index,
      std::move(calibration), kv_type);
}

std::unique_ptr<Runner> MakeContextParallelRunner(
    std::unique_ptr<ContextParallel> layout, int device_index,
    CalibrationConfig calibration, KvCacheType kv_type) {
  return std::make_unique<ParallelRunner>(
      std::make_unique<ContextParallelDriver>(std::move(layout)), device_index,
      std::move(calibration), kv_type);
}

std::unique_ptr<Runner> MakeSpecSplitRunner(std::unique_ptr<SpecSplit> layout,
                                            int device_index,
                                            CalibrationConfig calibration,
                                            KvCacheType kv_type) {
  return std::make_unique<ParallelRunner>(
      std::make_unique<SpecSplitDriver>(std::move(layout)), device_index,
      std::move(calibration), kv_type);
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

RunnerLease::~RunnerLease() {
  Release();
}

void RunnerLease::Release() {
  if (runner_ != nullptr) {
    runner_->active_.fetch_sub(1, std::memory_order_relaxed);
    runner_ = nullptr;
  }
  if (lock_.owns_lock()) {
    lock_.unlock();
  }
}

void RunnerLease::YieldFor(std::chrono::milliseconds quantum) {
  if (runner_ == nullptr || !lock_.owns_lock()) {
    return;
  }
  lock_.unlock();
  std::this_thread::sleep_for(quantum);
  lock_.lock();
}

Runner& RunnerLease::runner() const {
  return *runner_;
}

Engine& RunnerLease::engine() const {
  return runner_->engine();
}

Model& RunnerLease::model() const {
  return runner_->model();
}

const Tokenizer* RunnerLease::tokenizer() const {
  return runner_->tokenizer();
}

void RunnerPool::Add(std::unique_ptr<Runner> runner) {
  std::lock_guard<std::mutex> lock(mutex_);
  runners_.push_back(std::move(runner));
}

std::size_t RunnerPool::size() const {
  return runners_.size();
}

bool RunnerPool::empty() const {
  return runners_.empty();
}

Runner& RunnerPool::At(std::size_t index) const {
  return *runners_[index];
}

std::optional<RunnerLease> RunnerPool::Acquire(
    const std::function<bool()>& cancelled) {
  if (runners_.empty()) {
    return std::nullopt;
  }
  for (;;) {
    if (auto slot = TryAcquire()) {
      return slot;
    }
    // A peer that disconnects while queued stops waiting instead of
    // holding its place behind a generation it never reads.
    if (cancelled && cancelled()) {
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(kAcquirePollMs));
  }
}

std::optional<RunnerLease> RunnerPool::TryAcquire() {
  // Any idle runner wins; idle runners all have zero load, so the first
  // free one is the least loaded.
  for (const auto& runner : runners_) {
    std::unique_lock<std::mutex> lock(runner->mutex_, std::try_to_lock);
    if (lock.owns_lock()) {
      return RunnerLease(*runner, std::move(lock));
    }
  }
  return std::nullopt;
}

std::expected<GenerateOutcome, StatusCode> RunnerPool::SubmitLeastLoaded(
    const GenerateOptions& options,
    const std::function<bool(std::uint32_t)>& on_token,
    const std::function<bool()>& cancelled) const {
  std::unique_lock<std::mutex> pool_lock(mutex_);
  if (runners_.empty()) {
    return std::unexpected(StatusCode::InvalidArgument);
  }
  // Load counts queued scheduler jobs and in-flight turns, so a pool of
  // split runners (zero scheduler jobs each) still spreads turns instead
  // of always choosing the first.
  Runner* chosen = runners_.front().get();
  std::size_t head_load = chosen->SchedulerActive() +
                          chosen->SchedulerPending() +
                          chosen->active_.load(std::memory_order_relaxed);
  for (const auto& runner : runners_) {
    const std::size_t load = runner->SchedulerActive() +
                             runner->SchedulerPending() +
                             runner->active_.load(std::memory_order_relaxed);
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
