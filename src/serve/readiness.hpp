#pragma once

#include <mutex>
#include <string>
#include <utility>

namespace tessera {

class Model;
class Tokenizer;

}  // namespace tessera

namespace tessera::serve {

// Serving readiness, shared between the HTTP handler and a background
// model loader. The server answers requests before the model is ready,
// so /health and every model endpoint report this state until it flips
// to Ready. The loader runs on its own thread while connection threads
// read snapshots, so every method is safe to call concurrently.
//
// Usage:
//   serve::Readiness readiness;
//   readiness.Loading("loading weights");
//   readiness.Ready(&model, model.GetTokenizer());
class Readiness {
 public:
  enum class Phase { kLoading, kReady, kFailed };

  // A point-in-time copy safe to use outside the lock. `model` and
  // `tokenizer` are valid only while the phase is kReady, and stay valid
  // for the whole serve call (the loader owns them until Serve returns).
  struct Status {
    Phase phase = Phase::kLoading;
    std::string detail;
    Model* model = nullptr;
    const Tokenizer* tokenizer = nullptr;
  };

  // Record that the loader is working, with a step for /health.
  void Loading(std::string detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = Phase::kLoading;
    detail_ = std::move(detail);
    model_ = nullptr;
    tokenizer_ = nullptr;
  }

  // Record that the model is loaded and warm; serving is enabled.
  void Ready(Model* model, const Tokenizer* tokenizer) {
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = Phase::kReady;
    detail_.clear();
    model_ = model;
    tokenizer_ = tokenizer;
  }

  // Record that the load failed. `detail` is the reason. The server
  // stays up so a client can read the failure from /health.
  void Failed(std::string detail) {
    std::lock_guard<std::mutex> lock(mutex_);
    phase_ = Phase::kFailed;
    detail_ = std::move(detail);
    model_ = nullptr;
    tokenizer_ = nullptr;
  }

  // The current state. Call once per request and use the copy.
  [[nodiscard]] Status Get() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return Status{phase_, detail_, model_, tokenizer_};
  }

 private:
  mutable std::mutex mutex_;
  Phase phase_ = Phase::kLoading;
  std::string detail_;
  Model* model_ = nullptr;
  const Tokenizer* tokenizer_ = nullptr;
};

}  // namespace tessera::serve
