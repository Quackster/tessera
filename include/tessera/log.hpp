#pragma once

#include <cstdio>
#include <functional>
#include <memory>
#include <mutex>
#include <string_view>

namespace tessera::log {

// Severity of a diagnostic line. Debug is below the default threshold, so
// it is dropped unless SetMinLevel(Debug) raises the verbosity.
enum class Level : int {
  Debug = -1,
  Info = 0,
  Warn = 1,
  Error = 2,
};

// Diagnostics channel shared by the engine, models, and backends.
//
// Every line is "prefix: message" (rule 14: actionable logs), written
// through a sink. The default sink writes to stderr; replace it for tests
// or GUI front-ends. Safe to call from multiple threads. Copies share
// the lock, so a copied channel stays one channel.
//
// Usage:
//   log::Diagnostics log;
//   log.Info("engine 1", "backend rocm initialized");
class Diagnostics {
 public:
  using Sink = std::function<void(Level level, std::string_view prefix,
                                 std::string_view message)>;

  Diagnostics() = default;

  // Replace the sink; takes effect for subsequent lines only.
  void SetSink(Sink sink) {
    std::lock_guard<std::mutex> lock(*mutex_);
    sink_ = std::move(sink);
  }

  // Drop every line below `level` before it reaches the sink. Defaults to
  // Info, so Debug diagnostics (per-transfer summaries, cache probes) stay
  // off unless a front-end asks for them.
  void SetMinLevel(Level level) {
    std::lock_guard<std::mutex> lock(*mutex_);
    min_level_ = level;
  }

  void Debug(std::string_view prefix, std::string_view message) const {
    Emit(Level::Debug, prefix, message);
  }
  void Info(std::string_view prefix, std::string_view message) const {
    Emit(Level::Info, prefix, message);
  }
  void Warn(std::string_view prefix, std::string_view message) const {
    Emit(Level::Warn, prefix, message);
  }
  void Error(std::string_view prefix, std::string_view message) const {
    Emit(Level::Error, prefix, message);
  }

 private:
  void Emit(Level level, std::string_view prefix,
            std::string_view message) const {
    std::lock_guard<std::mutex> lock(*mutex_);
    if (static_cast<int>(level) < static_cast<int>(min_level_)) {
      return;
    }
    sink_(level, prefix, message);
  }

  // Shared across copies: a copied channel stays one channel.
  mutable std::shared_ptr<std::mutex> mutex_ =
      std::make_shared<std::mutex>();
  Level min_level_ = Level::Info;
  Sink sink_ = [](Level level, std::string_view prefix,
                  std::string_view message) {
    const char* level_name = "info";
    switch (level) {
      case Level::Debug: level_name = "debug"; break;
      case Level::Warn: level_name = "warn"; break;
      case Level::Error: level_name = "error"; break;
      default: break;
    }
    std::fprintf(stderr, "tessera [%s] %s: %s\n", level_name, prefix.data(),
                 message.data());
  };
};

}  // namespace tessera::log
