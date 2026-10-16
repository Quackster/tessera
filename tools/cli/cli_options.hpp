#pragma once

#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cli_common.hpp"
#include "tessera/defaults.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

// The one argument parser every tessera-cli command shares. `run`, `serve`
// and `calibrate` accept the same flag set: the command front ends decide
// what applies to them and report a flag they cannot honor, instead of
// rejecting it as unknown (AGENTS.md rule 2: one canonical flag mapping,
// so a new flag lands in one place).

namespace tessera::cli {

// Every option the CLI understands, one field per flag, defaulted to the
// same value the flag has when omitted. A field pair like
// `kv_type`/`kv_type_set` distinguishes "the flag was not passed" (the
// command applies its own default) from "the flag was passed".
struct CliOptions {
  // Model and data paths.
  std::string model_path;        // --model
  std::string draft_path;        // --draft (DFlash2 checkpoint)
  std::string mtp_path;          // --mtp (fetch-mtp / spec-split target)
  std::string mmproj_path;       // --mmproj
  std::string image_path;        // --image
  std::string calibration_path;  // --calibration (env TESSERA_CALIBRATION)
  std::string prompts_path;      // --prompts (calibrate)
  std::vector<std::string> prompt_texts;  // --prompt-text, repeatable

  // Logging and chat rendering.
  bool quiet = false;           // --quiet
  bool verbose = false;         // --verbose
  bool no_chat = false;         // --no-chat
  bool enable_thinking = true;  // --enable-thinking | --no-thinking
  bool thinking_flag_set = false;
  std::string reasoning_effort = "xhigh";  // --reasoning-effort
  bool reasoning_effort_set = false;

  // Serve.
  std::string host = kDefaultServeHost;  // --host
  std::uint16_t port = kDefaultServePort;
  bool auto_title = true;                  // --no-auto-title
  std::vector<std::string> api_keys;       // --api-key, repeatable
  std::vector<std::string> allow_origins;  // --allow-origin, repeatable
  std::size_t stats_interval_seconds = kDefaultStatsIntervalSeconds;

  // Layout selection. serve honors every route; run rejects none and
  // applies the route it selects (see each command).
  bool speculate = false;                   // --speculate (MTP)
  bool pipeline_mode = false;               // --pipeline
  bool tp_mode = false;                     // --tp
  bool cp_mode = false;                     // --cp
  bool spec_split_mode = false;             // --spec-split
  bool hybrid_mode = false;                 // --hybrid
  std::size_t dp_groups = 1;                // --dp
  std::vector<std::size_t> pipeline_split;  // --pipeline-split

  // Decode shape.
  tessera::SamplingOptions sampling;
  bool sample = false;  // --sample
  bool sampling_explicit = false;
  tessera::KvCacheType kv_type = tessera::KvCacheType::F32;
  bool kv_type_set = false;
  std::uint64_t seed = 0;
  std::size_t max_completion_tokens = 0;
  std::size_t max_thinking_tokens = 0;
  std::size_t context = kDefaultContext;
  std::size_t draft_block = kDefaultDraftBlock;
  std::size_t prefill_chunk = kDefaultPrefillChunk;
  std::size_t split_target = 0;
  std::size_t split_cap = 0;
  std::size_t attention_split = 0;
  std::size_t draft_context = 0;
  std::size_t gpu_layers = 0;
  bool gpu_layers_help = false;  // --gpu-layers help
  std::vector<int> gpus;         // --gpus

  // Calibrate.
  std::size_t calibrate_limit = 0;  // --calibrate-limit
  bool concurrent = false;          // --concurrent
};

namespace detail {

// The error for a flag value that is not the expected shape.
[[nodiscard]] inline std::string BadValue(std::string_view flag,
                                          std::string_view value) {
  return "invalid value '" + std::string(value) + "' for " + std::string(flag);
}

}  // namespace detail

// The value of `flag`, which sits at args[i + 1]; advances `i` past it.
[[nodiscard]] inline std::expected<std::string_view, std::string> StringValue(
    std::span<const std::string_view> args, std::size_t& i,
    std::string_view flag) {
  if (i + 1 >= args.size()) {
    return std::unexpected("argument '" + std::string(flag) +
                           "' needs a value");
  }
  return args[++i];
}

// A whole unsigned number (rejects trailing text and overflow).
[[nodiscard]] inline std::expected<std::uint64_t, std::string> ParseUnsigned(
    std::string_view text, std::string_view flag) {
  std::uint64_t value = 0;
  const char* const begin = text.data();
  const char* const end = text.data() + text.size();
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    return std::unexpected(detail::BadValue(flag, text));
  }
  return value;
}

// A real number (strtod syntax, nothing trailing).
[[nodiscard]] inline std::expected<double, std::string> ParseReal(
    std::string_view text, std::string_view flag) {
  const std::string owned(text);
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(owned.c_str(), &end);
  if (end == owned.c_str() || *end != '\0' || errno == ERANGE) {
    return std::unexpected(detail::BadValue(flag, text));
  }
  return value;
}

// The size value of `flag` at args[i + 1]; advances `i` past it.
[[nodiscard]] inline std::expected<std::size_t, std::string> SizeValue(
    std::span<const std::string_view> args, std::size_t& i,
    std::string_view flag) {
  auto text = StringValue(args, i, flag);
  if (!text) {
    return std::unexpected(text.error());
  }
  auto value = ParseUnsigned(*text, flag);
  if (!value) {
    return std::unexpected(value.error());
  }
  return static_cast<std::size_t>(*value);
}

// The real value of `flag` at args[i + 1]; advances `i` past it.
[[nodiscard]] inline std::expected<double, std::string> RealValue(
    std::span<const std::string_view> args, std::size_t& i,
    std::string_view flag) {
  auto text = StringValue(args, i, flag);
  if (!text) {
    return std::unexpected(text.error());
  }
  return ParseReal(*text, flag);
}

// Parse the arguments after the command word. Every flag is accepted for
// every command; the first problem is reported as a message without a
// prefix, so the caller prints it with its own usage text.
//
// Usage:
//   CliOptions options;
//   auto parsed = ParseCliOptions(args, &options);
[[nodiscard]] inline std::expected<void, std::string> ParseCliOptions(
    std::span<const std::string_view> args, CliOptions* options) {
  for (std::size_t i = 0; i < args.size(); ++i) {
    const std::string_view arg = args[i];
    if (arg == "--model") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->model_path = *value;
    } else if (arg == "--draft") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->draft_path = *value;
    } else if (arg == "--mtp") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->mtp_path = *value;
    } else if (arg == "--prompt-text") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->prompt_texts.emplace_back(*value);
    } else if (arg == "--mmproj") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->mmproj_path = *value;
    } else if (arg == "--image") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->image_path = *value;
    } else if (arg == "--quiet") {
      options->quiet = true;
    } else if (arg == "--verbose") {
      options->verbose = true;
    } else if (arg == "--no-chat") {
      options->no_chat = true;
    } else if (arg == "--enable-thinking") {
      options->enable_thinking = true;
      options->thinking_flag_set = true;
    } else if (arg == "--no-thinking") {
      options->enable_thinking = false;
      options->thinking_flag_set = true;
    } else if (arg == "--reasoning-effort") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->reasoning_effort = *value;
      options->reasoning_effort_set = true;
    } else if (arg == "--api-key") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->api_keys.emplace_back(*value);
    } else if (arg == "--allow-origin") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->allow_origins.emplace_back(*value);
    } else if (arg == "--host") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->host = *value;
    } else if (arg == "--no-auto-title") {
      options->auto_title = false;
    } else if (arg == "--port") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      if (*value > 65535) {
        return std::unexpected(detail::BadValue(arg, args[i]));
      }
      options->port = static_cast<std::uint16_t>(*value);
    } else if (arg == "--stats-interval") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->stats_interval_seconds = *value;
    } else if (arg == "--context") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->context = *value;
    } else if (arg == "--gpus") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      try {
        options->gpus = ParseGpuList(*value);
      } catch (const std::exception&) {
        return std::unexpected(detail::BadValue(arg, *value));
      }
    } else if (arg == "--pipeline-split") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      try {
        options->pipeline_split = ParseSizeList(*value);
      } catch (const std::exception&) {
        return std::unexpected(detail::BadValue(arg, *value));
      }
    } else if (arg == "--calibration") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->calibration_path = *value;
    } else if (arg == "--prompts") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->prompts_path = *value;
    } else if (arg == "--calibrate-limit") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->calibrate_limit = *value;
    } else if (arg == "--concurrent") {
      options->concurrent = true;
    } else if (arg == "--draft-block") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->draft_block = *value;
    } else if (arg == "--prefill-chunk") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->prefill_chunk = *value;
    } else if (arg == "--split-target") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->split_target = *value;
    } else if (arg == "--split-cap") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->split_cap = *value;
    } else if (arg == "--attention-split") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->attention_split = *value;
    } else if (arg == "--draft-context") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->draft_context = *value;
    } else if (arg == "--gpu-layers") {
      auto value = StringValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      // 0 (the default) uploads everything; `all` also keeps everything
      // resident; `help` sizes the model without loading it.
      if (*value == "help") {
        options->gpu_layers_help = true;
      } else if (*value == "all") {
        options->gpu_layers = 0;
      } else {
        auto layers = ParseUnsigned(*value, arg);
        if (!layers) {
          return std::unexpected(layers.error());
        }
        options->gpu_layers = static_cast<std::size_t>(*layers);
      }
    } else if (arg == "--speculate") {
      options->speculate = true;
    } else if (arg == "--pipeline") {
      options->pipeline_mode = true;
    } else if (arg == "--tp") {
      options->tp_mode = true;
    } else if (arg == "--cp") {
      options->cp_mode = true;
    } else if (arg == "--spec-split") {
      options->spec_split_mode = true;
    } else if (arg == "--hybrid") {
      options->hybrid_mode = true;
    } else if (arg == "--dp") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->dp_groups = *value;
    } else if (arg == "--sample") {
      options->sample = true;
    } else if (MatchKvType(arg, options->kv_type)) {
      options->kv_type_set = true;
    } else if (arg == "--temperature") {
      auto value = RealValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.temperature = static_cast<float>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--top-p") {
      auto value = RealValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.top_p = static_cast<float>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--top-k") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.top_k = static_cast<int>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--min-p") {
      auto value = RealValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.min_p = static_cast<float>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--presence-penalty") {
      auto value = RealValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.presence_penalty = static_cast<float>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--repetition-penalty") {
      auto value = RealValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->sampling.repetition_penalty = static_cast<float>(*value);
      options->sampling_explicit = true;
    } else if (arg == "--seed") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->seed = *value;
    } else if (arg == "--max-completion-tokens") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->max_completion_tokens = *value;
    } else if (arg == "--max-thinking-tokens") {
      auto value = SizeValue(args, i, arg);
      if (!value) {
        return std::unexpected(value.error());
      }
      options->max_thinking_tokens = *value;
    } else {
      // Diagnostic flags map to environment overrides; the canonical
      // mapping lives in ApplyDiagnosticFlag (cli_common.hpp). A
      // value-taking flag consumes its value first.
      std::optional<std::string_view> value;
      if (DiagnosticFlagTakesValue(arg)) {
        auto text = StringValue(args, i, arg);
        if (!text) {
          return std::unexpected(text.error());
        }
        value = *text;
      }
      if (!ApplyDiagnosticFlag(arg, value)) {
        return std::unexpected("unknown argument '" + std::string(arg) + "'");
      }
    }
  }
  return {};
}

// The selected devices: --gpus, or the first device when unset.
[[nodiscard]] inline std::vector<int> SelectedDevices(
    const CliOptions& options) {
  return options.gpus.empty() ? std::vector<int>{0} : options.gpus;
}

// True when a flag selects a multi-device layout.
[[nodiscard]] inline bool SplitLayoutActive(const CliOptions& options) {
  return options.pipeline_mode || options.tp_mode || options.cp_mode ||
         options.spec_split_mode || options.hybrid_mode;
}

// At most one layout route. `--hybrid` combines with `--tp` or `--cp` to
// pick the model-parallel kind inside each group. `--spec-split` needs
// `--speculate` and refuses `--draft`, which selects DFlash2.
[[nodiscard]] inline std::expected<void, std::string> ValidateLayoutFlags(
    const CliOptions& options) {
  const int routes = (options.pipeline_mode ? 1 : 0) +
                     (options.tp_mode ? 1 : 0) + (options.cp_mode ? 1 : 0) +
                     (options.spec_split_mode ? 1 : 0);
  if (routes > 1) {
    return std::unexpected(
        "--pipeline, --tp, --cp and --spec-split are mutually exclusive; "
        "pass at most one");
  }
  if ((options.pipeline_mode || options.spec_split_mode) &&
      options.hybrid_mode) {
    return std::unexpected(
        "--hybrid combines with --tp or --cp, not with --pipeline or "
        "--spec-split");
  }
  if (options.spec_split_mode && !options.speculate) {
    return std::unexpected("--spec-split needs --speculate (the MTP head)");
  }
  if (options.spec_split_mode && !options.draft_path.empty()) {
    return std::unexpected(
        "--spec-split runs the MTP head; it does not take --draft");
  }
  return {};
}

// The calibration strategy label the flags select: "dflash2", "mtp" or
// "none". One canonical mapping, because the label is part of the
// calibration key on disk, so a drift would orphan entries.
[[nodiscard]] inline std::string_view CliStrategyLabel(
    const CliOptions& options) {
  if (!options.draft_path.empty()) {
    return "dflash2";
  }
  return options.speculate ? "mtp" : "none";
}

// Apply the environment fallbacks the flags share: TESSERA_API_KEY and
// TESSERA_CALIBRATION fill an unset --api-key and --calibration.
inline void ApplyCliEnvDefaults(CliOptions* options) {
  if (options->api_keys.empty()) {
    if (const char* key = std::getenv("TESSERA_API_KEY");
        key != nullptr && *key != '\0') {
      options->api_keys.emplace_back(key);
    }
  }
  if (options->calibration_path.empty()) {
    if (const char* env = std::getenv("TESSERA_CALIBRATION");
        env != nullptr && *env != '\0') {
      options->calibration_path = env;
    }
  }
}

}  // namespace tessera::cli
