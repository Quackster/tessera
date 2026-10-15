#include <gtest/gtest.h>

#include <cstddef>
#include <expected>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "test_helpers.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/calibrate_sweep.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

// The calibration sweep front end: the fixed prompt set, the measure
// callback that splits decode from prefill, the chosen-rate medians, and the
// concurrent-session helpers.

using tessera::BuildSweepMeasure;
using tessera::CalibrationConfig;
using tessera::CalibrationPrompts;
using tessera::CalibrationPromptsFromFile;
using tessera::CalibrationReport;
using tessera::ConcurrentRate;
using tessera::DefaultCalibrationPrompts;
using tessera::FormatCalibrationReport;
using tessera::GenerateOptions;
using tessera::GenerateOutcome;
using tessera::kDecodeMeasureTokens;
using tessera::KvBytesPerToken;
using tessera::KvCacheType;
using tessera::MaxConcurrentSessions;
using tessera::MeasureChosenRates;
using tessera::MeasureFn;
using tessera::Setting;
using tessera::StatusCode;
using tessera::SummarizeConcurrentSessions;
using tessera::TimedGenerateFn;
using tessera::TransformerConfig;
using tessera::testing::FreshTempDir;
using tessera::testing::WriteString;

// The default prompt set is the three fixed, non-empty prompts the sweep
// repeats; the longest one drives the attention-split decision.
TEST(CalibrateSweepTest, DefaultPromptsAreFixedAndLong) {
  const std::vector<std::string> prompts = DefaultCalibrationPrompts();
  ASSERT_EQ(prompts.size(), 3u);
  for (const std::string& prompt : prompts) {
    EXPECT_FALSE(prompt.empty());
  }
  CalibrationPrompts set;
  set.prefill = {{1, 2, 3}, {4, 5}};
  EXPECT_EQ(set.count(), 2u);
  EXPECT_EQ(set.longest(), 3u);
}

// A prompts file is read one prompt per non-empty line; an empty file is
// rejected.
TEST(CalibrateSweepTest, PromptsFromFileReadsNonEmptyLines) {
  const auto dir = FreshTempDir("tessera_tests_calibration_prompts");
  const std::string path = (dir / "prompts.txt").string();
  WriteString(path, "first prompt\n\nsecond prompt\r\n");
  auto prompts = CalibrationPromptsFromFile(path);
  ASSERT_TRUE(prompts.has_value());
  ASSERT_EQ(prompts->size(), 2u);
  EXPECT_EQ((*prompts)[0], "first prompt");
  EXPECT_EQ((*prompts)[1], "second prompt");

  const std::string empty = (dir / "empty.txt").string();
  WriteString(empty, "\n\n");
  EXPECT_FALSE(CalibrationPromptsFromFile(empty).has_value());
  EXPECT_FALSE(
      CalibrationPromptsFromFile((dir / "missing.txt").string()).has_value());
}

// The measure callback uses the long prompt and reports prefill tps for the
// prefill setting, and the short prompt and decode tps for a decode setting.
TEST(CalibrateSweepTest, SweepMeasureSplitsDecodeAndPrefill) {
  CalibrationPrompts prompts;
  prompts.prefill = {{1, 2, 3}, {4, 5, 6}};
  prompts.decode = {{1, 2}, {4, 5}};
  std::vector<std::size_t> seen;
  const TimedGenerateFn generate =
      [&seen](std::span<const std::uint32_t> prompt,
              const GenerateOptions& options)
      -> std::expected<GenerateOutcome, StatusCode> {
    seen.push_back(prompt.size());
    GenerateOutcome out;
    out.prompt_tokens = prompt.size();
    out.produced = options.max_completion_tokens;
    out.decode_ms = 2.0;
    out.prefill_ms = 4.0;
    return out;
  };
  const MeasureFn measure =
      BuildSweepMeasure(generate, prompts, KvCacheType::F32);
  CalibrationConfig config;
  auto prefill = measure(Setting::PrefillChunkTokens, config, 0, false);
  ASSERT_TRUE(prefill.has_value());
  auto decode = measure(Setting::MxFp4SplitTarget, config, 0, false);
  ASSERT_TRUE(decode.has_value());
  ASSERT_EQ(seen.size(), 2u);
  EXPECT_EQ(seen[0], 3u);  // the long prompt
  EXPECT_EQ(seen[1], 2u);  // the short prompt
  EXPECT_DOUBLE_EQ(prefill->prefill_tps, 3.0 * 1000.0 / 4.0);
  EXPECT_DOUBLE_EQ(decode->decode_tps,
                   static_cast<double>(kDecodeMeasureTokens) * 1000.0 / 2.0);
}

// MeasureChosenRates takes the median decode and prefill rates over the
// prompts.
TEST(CalibrateSweepTest, MeasureChosenRatesMediansOverPrompts) {
  CalibrationPrompts prompts;
  prompts.prefill = {{1, 2, 3}, {4, 5, 6, 7, 8}, {9, 10, 11, 12, 13, 14, 15}};
  prompts.decode = {{1, 2}, {4, 5, 6, 7}, {8, 9, 10, 11, 12, 13}};
  const TimedGenerateFn generate =
      [](std::span<const std::uint32_t> prompt,
         const GenerateOptions& options)
      -> std::expected<GenerateOutcome, StatusCode> {
    GenerateOutcome out;
    out.prompt_tokens = prompt.size();
    out.produced = options.max_completion_tokens;
    out.decode_ms = 1.0;
    out.prefill_ms = 1.0;
    return out;
  };
  const MeasureFn measure =
      BuildSweepMeasure(generate, prompts, KvCacheType::F32);
  auto rates = MeasureChosenRates(measure, CalibrationConfig{}, prompts);
  ASSERT_TRUE(rates.has_value()) << tessera::ToString(rates.error());
  // Decode: produced = kDecodeMeasureTokens over 1 ms for every prompt.
  EXPECT_DOUBLE_EQ(rates->first,
                   static_cast<double>(kDecodeMeasureTokens) * 1000.0);
  // Prefill: prompt tokens 3, 5, 7 over 1 ms -> median 5000.
  EXPECT_DOUBLE_EQ(rates->second, 5000.0);
}

// The KV bytes per token scale by cache type over the full-attention
// layers only; an incomplete config sizes no cache.
TEST(CalibrateSweepTest, KvBytesPerTokenScalesByType) {
  TransformerConfig config;
  config.layers = 4;
  config.attention.kv_heads = 8;
  config.attention.head_dim = 128;
  // 4 layers * 8 kv heads * 128 head dim * 2 (key and value) = 8192.
  EXPECT_EQ(KvBytesPerToken(config, KvCacheType::F32), 8192u * 4);
  EXPECT_EQ(KvBytesPerToken(config, KvCacheType::F16), 8192u * 2);
  EXPECT_EQ(KvBytesPerToken(config, KvCacheType::Q8), 8192u);
  EXPECT_EQ(KvBytesPerToken(config, KvCacheType::Q4), 8192u / 2);

  TransformerConfig empty;
  EXPECT_EQ(KvBytesPerToken(empty, KvCacheType::Q8), 0u);
}

// The concurrent session count is the available bytes over the per-session
// cache; a zero input fits nothing.
TEST(CalibrateSweepTest, MaxConcurrentSessionsFitsByMemory) {
  // 100 bytes per token at a 1000-token context is 100000 bytes a session.
  EXPECT_EQ(MaxConcurrentSessions(1000000, 100, 1000), 10u);
  EXPECT_EQ(MaxConcurrentSessions(99999, 100, 1000), 0u);
  EXPECT_EQ(MaxConcurrentSessions(0, 100, 1000), 0u);
  EXPECT_EQ(MaxConcurrentSessions(1000000, 0, 1000), 0u);
  EXPECT_EQ(MaxConcurrentSessions(1000000, 100, 0), 0u);
}

// The concurrency row totals every session's tokens over the slowest
// session's wall time; a bad input yields a zeroed row.
TEST(CalibrateSweepTest, SummarizeConcurrentSessionsTotalsOverWallTime) {
  const std::vector<std::size_t> tokens = {32, 30, 28};
  const std::vector<double> ms = {1000.0, 1000.0, 2000.0};
  const ConcurrentRate rate = SummarizeConcurrentSessions(3, tokens, ms);
  EXPECT_EQ(rate.sessions, 3u);
  EXPECT_DOUBLE_EQ(rate.total_tps, 45.0);  // 90 tokens over 2000 ms
  EXPECT_DOUBLE_EQ(rate.per_session_tps, 15.0);

  EXPECT_DOUBLE_EQ(SummarizeConcurrentSessions(2, tokens, ms).total_tps, 0.0);
  EXPECT_EQ(SummarizeConcurrentSessions(0, {}, {}).sessions, 0u);
}

// The report prints the concurrent-session rows and the largest fitting
// count, or "not measured" when the benchmark did not run.
TEST(CalibrateSweepTest, ReportPrintsConcurrency) {
  CalibrationReport report;
  report.backend = "rocm";
  report.model_path = "/models/m";
  report.file = "/tmp/c.json";
  report.key_context = 4096;
  const std::string empty = FormatCalibrationReport(report);
  EXPECT_NE(empty.find("Concurrency"), std::string::npos);
  EXPECT_NE(empty.find("not measured"), std::string::npos);

  ConcurrentRate rate;
  rate.sessions = 2;
  rate.total_tps = 40.0;
  rate.per_session_tps = 20.0;
  report.concurrent.push_back(rate);
  report.max_concurrent = 3;
  const std::string text = FormatCalibrationReport(report);
  EXPECT_NE(text.find("2 session(s)"), std::string::npos);
  EXPECT_NE(text.find("40.00 tok/s total"), std::string::npos);
  EXPECT_NE(text.find("20.00 tok/s each"), std::string::npos);
  EXPECT_NE(text.find("3 at 4096 key token(s)"), std::string::npos);
}
