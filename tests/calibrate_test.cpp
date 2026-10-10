#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <expected>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/decode.hpp"
#include "test_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

using tessera::CalibrationConfig;
using tessera::CalibrationEntry;
using tessera::CalibrationFile;
using tessera::CalibrationReport;
using tessera::CandidateValues;
using tessera::FormatCalibrationReport;
using tessera::HardwareKey;
using tessera::kMinGain;
using tessera::KvCacheType;
using tessera::MakeHardwareKey;
using tessera::MaxContextForKv;
using tessera::MeasureFn;
using tessera::Measurement;
using tessera::MedianOf;
using tessera::PickBest;
using tessera::Setting;
using tessera::StatusCode;
using tessera::SweepOptions;
using tessera::TransformerConfig;
using tessera::testing::FreshTempDir;
using tessera::testing::WriteString;

namespace {

// A stand-in engine: the measured rate is a function of the setting value,
// so the sweep can be driven without a device. `rate` receives the call
// index for the current value, which lets a test inject a one-off noisy
// win. Values are keyed by the field the test cares about.
struct StandInEngine {
  std::function<std::uint64_t(const CalibrationConfig&)> key;
  std::function<double(std::size_t call_index)> rate;
  mutable std::map<std::uint64_t, std::size_t> calls;

  Measurement measure(const CalibrationConfig& config) const {
    const std::uint64_t id = key(config);
    const std::size_t call = calls[id]++;
    Measurement m;
    m.decode_tps = rate(call);
    m.prefill_tps = rate(call);
    return m;
  }
};

MeasureFn AsMeasure(const StandInEngine& engine) {
  return [&engine](Setting, const CalibrationConfig& config, std::size_t, bool)
             -> std::expected<Measurement, StatusCode> {
    return engine.measure(config);
  };
}

// Disable every sweep except `setting`, so a test isolates one setting.
void OnlySweep(SweepOptions& options, Setting setting) {
  options.sweep_split_target = setting == Setting::MxFp4SplitTarget;
  options.sweep_split_cap = setting == Setting::MxFp4SplitCap;
  options.sweep_prefill_chunk = setting == Setting::PrefillChunkTokens;
  options.sweep_attention_split = setting == Setting::AttentionSplit;
  options.sweep_draft_tokens = setting == Setting::DraftTokens;
  options.sweep_draft_context = setting == Setting::DraftContext;
}

CalibrationEntry MakeEntry(const HardwareKey& key, CalibrationConfig config,
                           double decode_tps, double prefill_tps) {
  CalibrationEntry entry;
  entry.key = key;
  entry.config = config;
  entry.decode_tps = decode_tps;
  entry.prefill_tps = prefill_tps;
  entry.date = "2026-10-10";
  return entry;
}

}  // namespace






TEST(CalibrateTest, CalibrationFileRoundTrips) {
  const auto dir = FreshTempDir("tessera_tests_calibration");
  const std::string path = (dir / "calibration.json").string();
  const HardwareKey key =
      MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "fp32", "dflash2");
  CalibrationConfig config;
  config.mxfp4_split_target = 640;
  config.prefill_chunk_tokens = 1024;
  config.draft_tokens = 4;
  CalibrationFile file;
  file.Set(MakeEntry(key, config, 61.5, 1234.0));

  ASSERT_TRUE(file.Save(path).has_value());
  auto loaded = CalibrationFile::Load(path);
  ASSERT_TRUE(loaded.has_value()) << tessera::ToString(loaded.error());
  const CalibrationEntry* entry = loaded->Find(key);
  ASSERT_NE(entry, nullptr);
  EXPECT_EQ(entry->config.mxfp4_split_target, 640u);
  EXPECT_EQ(entry->config.prefill_chunk_tokens, 1024u);
  EXPECT_EQ(entry->config.draft_tokens, 4u);
  EXPECT_DOUBLE_EQ(entry->decode_tps, 61.5);
  EXPECT_DOUBLE_EQ(entry->prefill_tps, 1234.0);
  EXPECT_EQ(entry->date, "2026-10-10");

  // An unknown key has no entry.
  const HardwareKey other =
      MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "fp32", "none");
  EXPECT_EQ(loaded->Find(other), nullptr);

  // A second entry with a different key coexists; the same key replaces.
  loaded->Set(MakeEntry(other, config, 1.0, 2.0));
  EXPECT_EQ(loaded->Entries().size(), 2u);
  loaded->Set(MakeEntry(key, config, 99.0, 100.0));
  EXPECT_EQ(loaded->Entries().size(), 2u);
  EXPECT_DOUBLE_EQ(loaded->Find(key)->decode_tps, 99.0);
}



// Flat candidates: every value measures the same, so the default is kept.

// A gain below kMinGain is noise: the default is kept.

// A real gain in the split target is found and kept.
TEST(CalibrateTest, SweepFindsSplitTargetGain) {
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  OnlySweep(options, Setting::MxFp4SplitTarget);
  auto measure = [](Setting, const CalibrationConfig& config, std::size_t, bool)
      -> std::expected<Measurement, StatusCode> {
    Measurement m;
    m.decode_tps = config.mxfp4_split_target == 640 ? 1.5 : 1.0;
    m.prefill_tps = m.decode_tps;
    return m;
  };
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.mxfp4_split_target, 640u);
  bool kept = false;
  for (const auto& point : outcome->points) {
    if (point.setting == Setting::MxFp4SplitTarget && point.value == 640) {
      kept = point.kept;
    }
  }
  EXPECT_TRUE(kept);
}

// A real gain in the draft block is found and kept.

// The interleaved confirmation rejects a win that was a single burst of
// noise: the sweep sees a fast candidate once, then the confirmation sees
// it slow and keeps the default.
TEST(CalibrateTest, SweepConfirmationRejectsNoisyWin) {
  constexpr std::uint64_t kWinner = 640;
  // The winner is fast for its first four calls (warm + three prompts, the
  // initial sweep) then matches the default.
  std::map<std::uint64_t, std::size_t> calls;
  auto measure = [&calls](Setting, const CalibrationConfig& config, std::size_t, bool)
      -> std::expected<Measurement, StatusCode> {
    const std::uint64_t id = config.mxfp4_split_target;
    const std::size_t call = calls[id]++;
    Measurement m;
    m.decode_tps = (id == kWinner && call < 4) ? 2.0 : 1.0;
    m.prefill_tps = m.decode_tps;
    return m;
  };
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  OnlySweep(options, Setting::MxFp4SplitTarget);
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.mxfp4_split_target, 320u);
}

// A measure failure aborts the sweep with the error status.

// A not-applicable setting is reported, not failed.

// The report lists the sweep, the chosen settings, the memory-bound
// context, and an example run command that reproduces the choices.

// A real gain in the split-K cap is found and kept.

// A real gain in the attention split is found and kept.

// A calibrated value must equal the default: the greedy token sequence is
// identical, because calibration changes only speed. The fp8 tensor-core
// MXFP4 split-K reassociates the accumulation, so on ROCm the logits move
// within the fp8 reference-exponent scheme's own noise (the scalar path
// already differs from the default tensor-core path by a comparable
// amount) while the greedy token is unchanged. The split is inert on
// Vulkan, which has no fp8 tensor-core kernel, so the logits match there.
// Skips cleanly with no model or device.
TEST(CalibrateTest, CalibratedSplitMatchesDefaultWithinTolerance) {
  const char* dir = std::getenv("TESSERA_TEST_MXFP4_DIR");
  if (dir == nullptr) {
    GTEST_SKIP() << "TESSERA_TEST_MXFP4_DIR not set";
  }
  std::unique_ptr<tessera::Engine> engine;
  tessera::testing::MakeEngineOrSkip(engine);
  auto model = engine->LoadModel(tessera::ModelOptions{dir, 1024});
  ASSERT_TRUE(model.has_value()) << tessera::ToString(model.error());

  tessera::GenerateOptions options;
  options.max_completion_tokens = 8;
  options.prompt_tokens = {760, 6511, 314, 9338, 369};
  auto default_ids = engine->Generate(**model, options);
  ASSERT_TRUE(default_ids.has_value())
      << tessera::ToString(default_ids.error());
  options.mxfp4_split_target = 640;
  auto calibrated_ids = engine->Generate(**model, options);
  ASSERT_TRUE(calibrated_ids.has_value())
      << tessera::ToString(calibrated_ids.error());
  EXPECT_EQ(*default_ids, *calibrated_ids);

  // The first-step logits stay within the per-backend tolerance: exact on
  // Vulkan (the split is inert), and within the fp8 scheme's noise on ROCm.
  tessera::core::DecodeCache default_cache;
  auto default_logits =
      tessera::core::DecodeLogits(engine->Owner(), **model, default_cache, 0);
  ASSERT_TRUE(default_logits.has_value())
      << tessera::ToString(default_logits.error());
  tessera::core::DecodeCache calibrated_cache;
  calibrated_cache.tuning.mxfp4_split_target = 640;
  auto calibrated_logits =
      tessera::core::DecodeLogits(engine->Owner(), **model, calibrated_cache, 0);
  ASSERT_TRUE(calibrated_logits.has_value())
      << tessera::ToString(calibrated_logits.error());
  ASSERT_EQ(default_logits->size(), calibrated_logits->size());
  double max_abs = 0.0;
  for (std::size_t i = 0; i < default_logits->size(); ++i) {
    max_abs = std::max(
        max_abs, std::fabs(static_cast<double>((*default_logits)[i] -
                                               (*calibrated_logits)[i])));
  }
  const std::string_view backend = engine->Owner().Name();
  const double tolerance = backend == "vulkan" ? 1.0e-4 : 1.5;
  EXPECT_LE(max_abs, tolerance) << "backend " << backend;
}
