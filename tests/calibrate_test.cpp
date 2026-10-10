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
using tessera::CandidateValues;
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

TEST(CalibrateTest, CandidateValuesListsDedupeAndBounds) {
  const auto split = CandidateValues(Setting::MxFp4SplitTarget, 320);
  EXPECT_EQ(split, (std::vector<std::size_t>{120, 320, 640}));
  const auto prefill = CandidateValues(Setting::PrefillChunkTokens, 512);
  EXPECT_EQ(prefill, (std::vector<std::size_t>{512, 768, 1024, 1536, 2048}));
  const auto prefill_default = CandidateValues(Setting::PrefillChunkTokens, 768);
  EXPECT_EQ(prefill_default,
            (std::vector<std::size_t>{512, 768, 1024, 1536, 2048}));
  // The checkpoint value is appended and deduped.
  const auto draft = CandidateValues(Setting::DraftTokens, 8);
  EXPECT_EQ(draft, (std::vector<std::size_t>{4, 8}));
  const auto draft_checkpoint = CandidateValues(Setting::DraftTokens, 16);
  EXPECT_EQ(draft_checkpoint, (std::vector<std::size_t>{4, 8, 16}));
  // A bound drops candidates above it.
  const auto bounded = CandidateValues(Setting::DraftTokens, 7, 7);
  EXPECT_EQ(bounded, (std::vector<std::size_t>{4, 7}));
  // Zero is never a candidate.
  const auto no_default = CandidateValues(Setting::DraftTokens, 0);
  EXPECT_EQ(no_default, (std::vector<std::size_t>{4, 8}));
}

TEST(CalibrateTest, MaxContextForKvScalesWithKvType) {
  TransformerConfig config;
  config.layers = 64;
  config.attention.kv_heads = 4;
  config.attention.head_dim = 256;
  config.hybrid = true;
  config.full_attention_interval = 4;  // 16 full-attention layers
  // Q8: 16 * 4 * 256 * 2 = 32768 bytes per token.
  const std::uint64_t available = 32768ull * 1000;
  EXPECT_EQ(MaxContextForKv(available, config, KvCacheType::Q8), 1000u);
  // F32 is 4x the bytes, so a quarter of the context.
  EXPECT_EQ(MaxContextForKv(available, config, KvCacheType::F32), 250u);
  // Q4 is half of Q8.
  EXPECT_EQ(MaxContextForKv(available, config, KvCacheType::Q4), 2000u);
  // A vanilla config uses every layer: 64 * 4 * 256 * 2.
  TransformerConfig vanilla;
  vanilla.layers = 64;
  vanilla.attention.kv_heads = 4;
  vanilla.attention.head_dim = 256;
  EXPECT_EQ(MaxContextForKv(131072ull * 500, vanilla, KvCacheType::Q8), 500u);
  // No K/V geometry cannot size a cache.
  TransformerConfig empty;
  EXPECT_EQ(MaxContextForKv(available, empty, KvCacheType::Q8), 0u);
}

TEST(CalibrateTest, MedianOfHandlesEmptyOddAndEven) {
  EXPECT_DOUBLE_EQ(MedianOf(std::vector<double>{}), 0.0);
  const std::vector<double> odd = {3.0, 1.0, 2.0};
  EXPECT_DOUBLE_EQ(MedianOf(odd), 2.0);
  const std::vector<double> even = {4.0, 1.0, 3.0, 2.0};
  EXPECT_DOUBLE_EQ(MedianOf(even), 2.5);
}

TEST(CalibrateTest, PickBestEmptyMissingDefaultTieAndGain) {
  EXPECT_FALSE(
      PickBest(std::vector<std::size_t>{}, 1, std::vector<double>{}).has_value());
  const std::vector<std::size_t> candidates = {1, 2};
  // A default absent from the list returns the best.
  const std::vector<double> gain = {1.0, 1.5};
  EXPECT_EQ(PickBest(candidates, 9, gain), 2u);
  // A tie keeps the default.
  const std::vector<double> tie = {2.0, 2.0};
  EXPECT_EQ(PickBest(candidates, 1, tie), 1u);
  // A real gain picks the winner.
  const std::vector<double> real = {1.0, 1.5};
  EXPECT_EQ(PickBest(candidates, 1, real), 2u);
  // A gain below kMinGain is noise and keeps the default.
  const std::vector<double> noise = {1.0, 1.0 + kMinGain * 0.5};
  EXPECT_EQ(PickBest(candidates, 1, noise), 1u);
  // A length mismatch is rejected.
  EXPECT_FALSE(
      PickBest(candidates, 1, std::vector<double>{1.0}).has_value());
}

TEST(CalibrateTest, HardwareKeyChangesWithEveryField) {
  const HardwareKey base =
      MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "fp32", "dflash2");
  const HardwareKey same =
      MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "fp32", "dflash2");
  EXPECT_EQ(base, same);
  EXPECT_NE(base, MakeHardwareKey("vulkan", "R9700", "Qwen3.8-27B", 4096,
                                  "fp32", "dflash2"));
  EXPECT_NE(base, MakeHardwareKey("rocm", "M100", "Qwen3.8-27B", 4096, "fp32",
                                  "dflash2"));
  EXPECT_NE(base, MakeHardwareKey("rocm", "R9700", "other", 4096, "fp32",
                                  "dflash2"));
  EXPECT_NE(base, MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 8192,
                                  "fp32", "dflash2"));
  EXPECT_NE(base, MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "q8",
                                  "dflash2"));
  EXPECT_NE(base, MakeHardwareKey("rocm", "R9700", "Qwen3.8-27B", 4096, "fp32",
                                  "none"));
  EXPECT_EQ(base.ToString(), same.ToString());
}

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

TEST(CalibrateTest, CalibrationFileRejectsMissingAndMalformed) {
  const auto dir = FreshTempDir("tessera_tests_calibration_bad");
  const std::string missing = (dir / "absent.json").string();
  auto absent = CalibrationFile::Load(missing);
  ASSERT_FALSE(absent.has_value());
  EXPECT_EQ(absent.error(), StatusCode::FileNotFound);

  const std::string bad = (dir / "bad.json").string();
  WriteString(bad, "this is not json");
  auto malformed = CalibrationFile::Load(bad);
  ASSERT_FALSE(malformed.has_value());
  EXPECT_EQ(malformed.error(), StatusCode::MalformedFile);

  const std::string wrong = (dir / "wrong.json").string();
  WriteString(wrong, R"({"version":1})");
  auto no_entries = CalibrationFile::Load(wrong);
  ASSERT_FALSE(no_entries.has_value());
  EXPECT_EQ(no_entries.error(), StatusCode::MalformedFile);

  const std::string partial = (dir / "partial.json").string();
  WriteString(partial, R"({"version":1,"entries":[{"backend":"rocm"}]})");
  auto truncated = CalibrationFile::Load(partial);
  ASSERT_FALSE(truncated.has_value());
  EXPECT_EQ(truncated.error(), StatusCode::MalformedFile);
}

TEST(CalibrateTest, ApplySavedCalibrationKeepsExplicitValues) {
  const HardwareKey key =
      MakeHardwareKey("rocm", "R9700", "m", 4096, "fp32", "none");
  CalibrationConfig saved_config;
  saved_config.mxfp4_split_target = 640;
  saved_config.prefill_chunk_tokens = 1024;
  saved_config.draft_tokens = 4;
  const CalibrationEntry saved = MakeEntry(key, saved_config, 1.0, 1.0);

  CalibrationConfig explicit_config;
  explicit_config.prefill_chunk_tokens = 256;  // explicit wins
  const CalibrationConfig merged =
      tessera::ApplySavedCalibration(explicit_config, saved);
  EXPECT_EQ(merged.prefill_chunk_tokens, 256u);
  EXPECT_EQ(merged.mxfp4_split_target, 640u);
  EXPECT_EQ(merged.draft_tokens, 4u);
}

// Flat candidates: every value measures the same, so the default is kept.
TEST(CalibrateTest, SweepKeepsDefaultWhenFlat) {
  StandInEngine engine;
  engine.key = [](const CalibrationConfig& c) {
    return static_cast<std::uint64_t>(c.mxfp4_split_target);
  };
  engine.rate = [](std::size_t) { return 50.0; };
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  options.sweep_prefill_chunk = false;
  auto outcome = tessera::RunCalibrationSweep(options, AsMeasure(engine));
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.mxfp4_split_target, 320u);
}

// A gain below kMinGain is noise: the default is kept.
TEST(CalibrateTest, SweepKeepsDefaultBelowMinGain) {
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  options.sweep_prefill_chunk = false;
  // Only the 640 candidate is a hair faster, inside the noise floor.
  auto measure = [](Setting, const CalibrationConfig& config, std::size_t, bool)
      -> std::expected<Measurement, StatusCode> {
    Measurement m;
    m.decode_tps = config.mxfp4_split_target == 640 ? 1.0 + kMinGain * 0.5
                                                    : 1.0;
    m.prefill_tps = m.decode_tps;
    return m;
  };
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.mxfp4_split_target, 320u);
}

// A real gain in the split target is found and kept.
TEST(CalibrateTest, SweepFindsSplitTargetGain) {
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  options.sweep_prefill_chunk = false;
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
TEST(CalibrateTest, SweepFindsDraftGain) {
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.draft_tokens = 8;
  options.sweep_split_target = false;
  options.sweep_prefill_chunk = false;
  options.sweep_draft_tokens = true;
  auto measure = [](Setting, const CalibrationConfig& config, std::size_t, bool)
      -> std::expected<Measurement, StatusCode> {
    Measurement m;
    m.decode_tps = config.draft_tokens == 4 ? 2.0 : 1.0;
    m.prefill_tps = m.decode_tps;
    return m;
  };
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.draft_tokens, 4u);
}

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
  options.sweep_prefill_chunk = false;
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->config.mxfp4_split_target, 320u);
}

// A measure failure aborts the sweep with the error status.
TEST(CalibrateTest, SweepFailureReturnsError) {
  int calls = 0;
  auto measure = [&calls](Setting, const CalibrationConfig&, std::size_t,
                          bool) -> std::expected<Measurement, StatusCode> {
    if (++calls >= 3) {
      return std::unexpected(StatusCode::DeviceError);
    }
    return Measurement{1.0, 1.0};
  };
  SweepOptions options;
  options.prompt_count = 3;
  options.defaults.mxfp4_split_target = 320;
  options.sweep_prefill_chunk = false;
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_FALSE(outcome.has_value());
  EXPECT_EQ(outcome.error(), StatusCode::DeviceError);
}

// A not-applicable setting is reported, not failed.
TEST(CalibrateTest, SweepReportsNotApplicable) {
  auto measure = [](Setting, const CalibrationConfig&, std::size_t, bool)
      -> std::expected<Measurement, StatusCode> {
    return Measurement{1.0, 1.0};
  };
  SweepOptions options;
  options.prompt_count = 1;
  options.sweep_split_target = false;
  options.sweep_prefill_chunk = false;
  options.sweep_draft_tokens = false;
  auto outcome = tessera::RunCalibrationSweep(options, measure);
  ASSERT_TRUE(outcome.has_value()) << tessera::ToString(outcome.error());
  EXPECT_EQ(outcome->points.size(), 0u);
  EXPECT_EQ(outcome->not_applicable.size(), 3u);
}

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
