#include <gtest/gtest.h>

#include <cstdlib>
#include <expected>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "cli_common.hpp"
#include "cli_helpers.hpp"
#include "cli_options.hpp"
#include "tessera/defaults.hpp"
#include "tessera/hybrid.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

namespace {

using tessera::cli::CliOptions;
using tessera::cli::ParseCliOptions;

CliOptions ParseOrDie(std::initializer_list<std::string_view> args) {
  std::vector<std::string_view> list(args);
  CliOptions options;
  auto parsed = ParseCliOptions(list, &options);
  EXPECT_TRUE(parsed) << (parsed ? "" : parsed.error());
  return options;
}

std::string ParseError(std::initializer_list<std::string_view> args) {
  std::vector<std::string_view> list(args);
  CliOptions options;
  auto parsed = ParseCliOptions(list, &options);
  EXPECT_FALSE(parsed);
  return parsed ? std::string("<no error>") : parsed.error();
}

TEST(CliOptionsTest, DefaultsMatchTheDocumentedOnes) {
  const CliOptions options = ParseOrDie({});
  EXPECT_TRUE(options.model_path.empty());
  EXPECT_EQ(options.context, tessera::kDefaultContext);
  EXPECT_EQ(options.draft_block, tessera::kDefaultDraftBlock);
  EXPECT_EQ(options.prefill_chunk, tessera::kDefaultPrefillChunk);
  EXPECT_EQ(options.kv_type, tessera::KvCacheType::F32);
  EXPECT_FALSE(options.kv_type_set);
  EXPECT_FALSE(options.kv_offload);
  EXPECT_TRUE(options.enable_thinking);
  EXPECT_FALSE(options.thinking_flag_set);
  EXPECT_EQ(options.reasoning_effort, "xhigh");
  EXPECT_FALSE(options.reasoning_effort_set);
  EXPECT_EQ(options.host, tessera::kDefaultServeHost);
  EXPECT_EQ(options.port, tessera::kDefaultServePort);
  EXPECT_EQ(options.stats_interval_seconds,
            tessera::kDefaultStatsIntervalSeconds);
  EXPECT_TRUE(options.auto_title);
  EXPECT_EQ(options.dp_groups, 1u);
  EXPECT_TRUE(options.gpus.empty());
  EXPECT_FALSE(options.sample);
  EXPECT_FALSE(options.sampling_explicit);
  EXPECT_EQ(options.gpu_layers, 0u);
  EXPECT_FALSE(options.gpu_layers_help);
}

TEST(CliOptionsTest, EveryFlagLandsInItsField) {
  const CliOptions options = ParseOrDie({"--model",
                                         "m.gguf",
                                         "--draft",
                                         "d",
                                         "--mtp",
                                         "mtp",
                                         "--prompt-text",
                                         "one",
                                         "--prompt-text",
                                         "two",
                                         "--mmproj",
                                         "mp.gguf",
                                         "--image",
                                         "img.ppm",
                                         "--calibration",
                                         "cal.json",
                                         "--prompts",
                                         "p.txt",
                                         "--quiet",
                                         "--verbose",
                                         "--no-chat",
                                         "--no-thinking",
                                         "--reasoning-effort",
                                         "low",
                                         "--host",
                                         "0.0.0.0",
                                         "--port",
                                         "9000",
                                         "--no-auto-title",
                                         "--api-key",
                                         "k1",
                                         "--api-key",
                                         "k2",
                                         "--allow-origin",
                                         "*",
                                         "--stats-interval",
                                         "7",
                                         "--speculate",
                                         "--dp",
                                         "3",
                                         "--sample",
                                         "--temperature",
                                         "0.7",
                                         "--top-p",
                                         "0.9",
                                         "--top-k",
                                         "5",
                                         "--min-p",
                                         "0.05",
                                         "--presence-penalty",
                                         "0.1",
                                         "--repetition-penalty",
                                         "1.2",
                                         "--seed",
                                         "42",
                                         "--max-completion-tokens",
                                         "64",
                                         "--max-thinking-tokens",
                                         "32",
                                         "--context",
                                         "4096",
                                         "--draft-block",
                                         "8",
                                         "--prefill-chunk",
                                         "128",
                                         "--split-target",
                                         "11",
                                         "--split-cap",
                                         "3",
                                         "--attention-split",
                                         "9",
                                         "--draft-context",
                                         "2048",
                                         "--gpu-layers",
                                         "12",
                                         "--gpus",
                                         "1,2",
                                         "--calibrate-limit",
                                         "5",
                                         "--concurrent",
                                         "--kv-q8",
                                         "--kv-offload"});
  EXPECT_EQ(options.model_path, "m.gguf");
  EXPECT_EQ(options.draft_path, "d");
  EXPECT_EQ(options.mtp_path, "mtp");
  ASSERT_EQ(options.prompt_texts.size(), 2u);
  EXPECT_EQ(options.prompt_texts[0], "one");
  EXPECT_EQ(options.prompt_texts[1], "two");
  EXPECT_EQ(options.mmproj_path, "mp.gguf");
  EXPECT_EQ(options.image_path, "img.ppm");
  EXPECT_EQ(options.calibration_path, "cal.json");
  EXPECT_EQ(options.prompts_path, "p.txt");
  EXPECT_TRUE(options.quiet);
  EXPECT_TRUE(options.verbose);
  EXPECT_TRUE(options.no_chat);
  EXPECT_FALSE(options.enable_thinking);
  EXPECT_TRUE(options.thinking_flag_set);
  EXPECT_EQ(options.reasoning_effort, "low");
  EXPECT_TRUE(options.reasoning_effort_set);
  EXPECT_EQ(options.host, "0.0.0.0");
  EXPECT_EQ(options.port, 9000);
  EXPECT_FALSE(options.auto_title);
  ASSERT_EQ(options.api_keys.size(), 2u);
  EXPECT_EQ(options.api_keys[1], "k2");
  ASSERT_EQ(options.allow_origins.size(), 1u);
  EXPECT_EQ(options.allow_origins[0], "*");
  EXPECT_EQ(options.stats_interval_seconds, 7u);
  EXPECT_TRUE(options.speculate);
  EXPECT_EQ(options.dp_groups, 3u);
  EXPECT_TRUE(options.sample);
  EXPECT_FLOAT_EQ(options.sampling.temperature, 0.7f);
  EXPECT_FLOAT_EQ(options.sampling.top_p, 0.9f);
  EXPECT_EQ(options.sampling.top_k, 5);
  EXPECT_FLOAT_EQ(options.sampling.min_p, 0.05f);
  EXPECT_FLOAT_EQ(options.sampling.presence_penalty, 0.1f);
  EXPECT_FLOAT_EQ(options.sampling.repetition_penalty, 1.2f);
  EXPECT_TRUE(options.sampling_explicit);
  EXPECT_EQ(options.seed, 42u);
  EXPECT_EQ(options.max_completion_tokens, 64u);
  EXPECT_EQ(options.max_thinking_tokens, 32u);
  EXPECT_EQ(options.context, 4096u);
  EXPECT_EQ(options.draft_block, 8u);
  EXPECT_EQ(options.prefill_chunk, 128u);
  EXPECT_EQ(options.split_target, 11u);
  EXPECT_EQ(options.split_cap, 3u);
  EXPECT_EQ(options.attention_split, 9u);
  EXPECT_EQ(options.draft_context, 2048u);
  EXPECT_EQ(options.gpu_layers, 12u);
  ASSERT_EQ(options.gpus.size(), 2u);
  EXPECT_EQ(options.gpus[0], 1);
  EXPECT_EQ(options.gpus[1], 2);
  EXPECT_EQ(options.calibrate_limit, 5u);
  EXPECT_TRUE(options.concurrent);
  EXPECT_EQ(options.kv_type, tessera::KvCacheType::Q8);
  EXPECT_TRUE(options.kv_type_set);
  EXPECT_TRUE(options.kv_offload);
}

TEST(CliOptionsTest, KvFlagsSelectAndMarkTheType) {
  for (const auto& [flag, type] :
       {std::pair{"--kv-f16", tessera::KvCacheType::F16},
        std::pair{"--kv-q8", tessera::KvCacheType::Q8},
        std::pair{"--kv-q4", tessera::KvCacheType::Q4},
        std::pair{"--kv-fp8", tessera::KvCacheType::FP8}}) {
    const CliOptions options = ParseOrDie({flag});
    EXPECT_EQ(options.kv_type, type);
    EXPECT_TRUE(options.kv_type_set);
  }
}

TEST(CliOptionsTest, GpuLayersForms) {
  EXPECT_EQ(ParseOrDie({"--gpu-layers", "all"}).gpu_layers, 0u);
  const CliOptions help = ParseOrDie({"--gpu-layers", "help"});
  EXPECT_TRUE(help.gpu_layers_help);
  EXPECT_EQ(help.gpu_layers, 0u);
  EXPECT_EQ(ParseOrDie({"--gpu-layers", "7"}).gpu_layers, 7u);
}

TEST(CliOptionsTest, ParseErrorsCarryTheFlag) {
  EXPECT_EQ(ParseError({"--model"}), "argument '--model' needs a value");
  EXPECT_EQ(ParseError({"--context", "abc"}),
            "invalid value 'abc' for --context");
  EXPECT_EQ(ParseError({"--context", "12x"}),
            "invalid value '12x' for --context");
  EXPECT_EQ(ParseError({"--context", "99999999999999999999"}),
            "invalid value '99999999999999999999' for --context");
  EXPECT_EQ(ParseError({"--seed", "-1"}), "invalid value '-1' for --seed");
  EXPECT_EQ(ParseError({"--frobnicate"}), "unknown argument '--frobnicate'");
  EXPECT_EQ(ParseError({"--tiled-min-rows"}),
            "argument '--tiled-min-rows' needs a value");
}

TEST(CliOptionsTest, DiagnosticFlagsMapToEnvironment) {
  ParseOrDie({"--wmma"});
  const char* wmma = std::getenv("TESSERA_MXFP4_WMMA");
  ASSERT_NE(wmma, nullptr);
  EXPECT_EQ(std::string(wmma), "1");
  ParseOrDie({"--no-wmma"});
  wmma = std::getenv("TESSERA_MXFP4_WMMA");
  ASSERT_NE(wmma, nullptr);
  EXPECT_EQ(std::string(wmma), "0");
  ParseOrDie({"--tiled-min-rows", "100"});
  const char* rows = std::getenv("TESSERA_TILED_MIN_ROWS");
  ASSERT_NE(rows, nullptr);
  EXPECT_EQ(std::string(rows), "100");
  EXPECT_TRUE(tessera::cli::DiagnosticFlagTakesValue("--tiled-min-rows"));
  EXPECT_FALSE(tessera::cli::DiagnosticFlagTakesValue("--wmma"));
  unsetenv("TESSERA_MXFP4_WMMA");
  unsetenv("TESSERA_TILED_MIN_ROWS");
}

TEST(CliOptionsTest, SelectedDevicesDefaultsToTheFirst) {
  EXPECT_EQ(tessera::cli::SelectedDevices(ParseOrDie({})),
            (std::vector<int>{0}));
  EXPECT_EQ(tessera::cli::SelectedDevices(ParseOrDie({"--gpus", "1,2"})),
            (std::vector<int>{1, 2}));
}

TEST(CliOptionsTest, SplitLayoutActiveFollowsTheLayoutFlags) {
  EXPECT_FALSE(tessera::cli::SplitLayoutActive(ParseOrDie({})));
  EXPECT_TRUE(tessera::cli::SplitLayoutActive(ParseOrDie({"--pipeline"})));
  EXPECT_TRUE(tessera::cli::SplitLayoutActive(ParseOrDie({"--tp"})));
  EXPECT_TRUE(tessera::cli::SplitLayoutActive(ParseOrDie({"--cp"})));
  EXPECT_TRUE(tessera::cli::SplitLayoutActive(ParseOrDie({"--spec-split"})));
  EXPECT_TRUE(tessera::cli::SplitLayoutActive(ParseOrDie({"--hybrid"})));
}

TEST(CliOptionsTest, ValidateLayoutFlagsAcceptsTheValidCombinations) {
  EXPECT_TRUE(tessera::cli::ValidateLayoutFlags(ParseOrDie({})));
  EXPECT_TRUE(tessera::cli::ValidateLayoutFlags(ParseOrDie({"--tp"})));
  EXPECT_TRUE(
      tessera::cli::ValidateLayoutFlags(ParseOrDie({"--hybrid", "--tp"})));
  EXPECT_TRUE(
      tessera::cli::ValidateLayoutFlags(ParseOrDie({"--hybrid", "--cp"})));
  EXPECT_TRUE(tessera::cli::ValidateLayoutFlags(
      ParseOrDie({"--spec-split", "--speculate"})));
}

TEST(CliOptionsTest, ValidateLayoutFlagsRejectsConflicts) {
  auto message = [](std::initializer_list<std::string_view> args) {
    auto valid = tessera::cli::ValidateLayoutFlags(ParseOrDie(args));
    EXPECT_FALSE(valid);
    return valid ? std::string("<ok>") : valid.error();
  };
  EXPECT_EQ(message({"--tp", "--cp"}),
            "--pipeline, --tp, --cp and --spec-split are mutually exclusive; "
            "pass at most one");
  EXPECT_EQ(message({"--hybrid", "--pipeline"}),
            "--hybrid combines with --tp or --cp, not with --pipeline or "
            "--spec-split");
  EXPECT_EQ(message({"--hybrid", "--spec-split", "--speculate"}),
            "--hybrid combines with --tp or --cp, not with --pipeline or "
            "--spec-split");
  EXPECT_EQ(message({"--spec-split"}),
            "--spec-split needs --speculate (the MTP head)");
  EXPECT_EQ(message({"--spec-split", "--speculate", "--draft", "d"}),
            "--spec-split runs the MTP head; it does not take --draft");
}

TEST(CliOptionsTest, StrategyLabelFollowsDraftThenSpeculate) {
  EXPECT_EQ(tessera::cli::CliStrategyLabel(ParseOrDie({})), "none");
  EXPECT_EQ(tessera::cli::CliStrategyLabel(ParseOrDie({"--speculate"})), "mtp");
  EXPECT_EQ(tessera::cli::CliStrategyLabel(
                ParseOrDie({"--draft", "d", "--speculate"})),
            "dflash2");
}

TEST(CliOptionsTest, EnvDefaultsFillOnlyUnsetValues) {
  setenv("TESSERA_API_KEY", "env-key", 1);
  setenv("TESSERA_CALIBRATION", "env-cal.json", 1);
  CliOptions empty;
  tessera::cli::ApplyCliEnvDefaults(&empty);
  ASSERT_EQ(empty.api_keys.size(), 1u);
  EXPECT_EQ(empty.api_keys[0], "env-key");
  EXPECT_EQ(empty.calibration_path, "env-cal.json");
  CliOptions explicit_values =
      ParseOrDie({"--api-key", "flag-key", "--calibration", "flag-cal.json"});
  tessera::cli::ApplyCliEnvDefaults(&explicit_values);
  ASSERT_EQ(explicit_values.api_keys.size(), 1u);
  EXPECT_EQ(explicit_values.api_keys[0], "flag-key");
  EXPECT_EQ(explicit_values.calibration_path, "flag-cal.json");
  unsetenv("TESSERA_API_KEY");
  unsetenv("TESSERA_CALIBRATION");
}

TEST(HybridGroupsTest, SplitsGroupMajor) {
  tessera::HybridOptions options;
  options.devices = {0, 1, 2, 3};
  options.dp_groups = 2;
  options.mp_kind = tessera::HybridMpKind::Tensor;
  auto groups = tessera::HybridGroups(options);
  ASSERT_TRUE(groups) << tessera::ToString(groups.error());
  ASSERT_EQ(groups->size(), 2u);
  EXPECT_EQ((*groups)[0].index, 0u);
  EXPECT_EQ((*groups)[0].devices, (std::vector<int>{0, 1}));
  EXPECT_EQ((*groups)[1].index, 1u);
  EXPECT_EQ((*groups)[1].devices, (std::vector<int>{2, 3}));
}

TEST(HybridGroupsTest, OneGroupKeepsEveryDeviceInRankOrder) {
  tessera::HybridOptions options;
  options.devices = {2, 0, 1};
  options.dp_groups = 1;
  options.mp_kind = tessera::HybridMpKind::Context;
  auto groups = tessera::HybridGroups(options);
  ASSERT_TRUE(groups) << tessera::ToString(groups.error());
  ASSERT_EQ(groups->size(), 1u);
  EXPECT_EQ((*groups)[0].devices, (std::vector<int>{2, 0, 1}));
}

TEST(HybridGroupsTest, SingletonGroupsWhenDpCoversEveryDevice) {
  tessera::HybridOptions options;
  options.devices = {0, 1, 2, 3};
  options.dp_groups = 4;
  auto groups = tessera::HybridGroups(options);
  ASSERT_TRUE(groups) << tessera::ToString(groups.error());
  ASSERT_EQ(groups->size(), 4u);
  for (std::size_t g = 0; g < groups->size(); ++g) {
    EXPECT_EQ((*groups)[g].index, g);
    EXPECT_EQ((*groups)[g].devices, (std::vector<int>{static_cast<int>(g)}));
  }
}

TEST(HybridGroupsTest, RejectsBadDeviceCounts) {
  tessera::HybridOptions options;
  options.devices = {0, 1, 2, 3};
  options.dp_groups = 3;
  EXPECT_FALSE(tessera::HybridGroups(options));
  options.dp_groups = 0;
  EXPECT_FALSE(tessera::HybridGroups(options));
  options.dp_groups = 1;
  options.devices = {};
  EXPECT_FALSE(tessera::HybridGroups(options));
}

TEST(EncodePromptTextTest, NoChatEncodesTheRawText) {
  const tessera::Tokenizer tokenizer({"a", "b"}, {1, 1}, {});
  bool render_called = false;
  auto ids = tessera::cli::EncodePromptText(
      "ab", true, tokenizer,
      [&]() -> std::expected<std::string, tessera::StatusCode> {
        render_called = true;
        return "b";
      });
  ASSERT_TRUE(ids) << tessera::ToString(ids.error());
  EXPECT_EQ(*ids, (std::vector<std::uint32_t>{0, 1}));
  EXPECT_FALSE(render_called);
}

TEST(EncodePromptTextTest, RenderedTextIsEncoded) {
  const tessera::Tokenizer tokenizer({"a", "b"}, {1, 1}, {});
  auto ids = tessera::cli::EncodePromptText(
      "ab", false, tokenizer,
      []() -> std::expected<std::string, tessera::StatusCode> { return "b"; });
  ASSERT_TRUE(ids) << tessera::ToString(ids.error());
  EXPECT_EQ(*ids, (std::vector<std::uint32_t>{1}));
}

TEST(EncodePromptTextTest, UnsupportedTemplateFallsBackToRawText) {
  const tessera::Tokenizer tokenizer({"a", "b"}, {1, 1}, {});
  auto ids = tessera::cli::EncodePromptText(
      "ab", false, tokenizer,
      []() -> std::expected<std::string, tessera::StatusCode> {
        return std::unexpected(tessera::StatusCode::UnsupportedFeature);
      });
  ASSERT_TRUE(ids) << tessera::ToString(ids.error());
  EXPECT_EQ(*ids, (std::vector<std::uint32_t>{0, 1}));
}

TEST(EncodePromptTextTest, TemplateErrorsPropagate) {
  const tessera::Tokenizer tokenizer({"a", "b"}, {1, 1}, {});
  auto ids = tessera::cli::EncodePromptText(
      "ab", false, tokenizer,
      []() -> std::expected<std::string, tessera::StatusCode> {
        return std::unexpected(tessera::StatusCode::InvalidArgument);
      });
  EXPECT_FALSE(ids);
  EXPECT_EQ(ids.error(), tessera::StatusCode::InvalidArgument);
}

}  // namespace
