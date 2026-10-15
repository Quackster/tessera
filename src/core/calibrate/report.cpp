#include "tessera/calibrate.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <string>

// The human-readable calibration report: the sweep, the chosen settings,
// the concurrency rows, and the example run commands that reproduce them.
// One canonical implementation (AGENTS.md rules 2 and 3).

namespace tessera {

ExampleCommands BuildExampleCommands(const CalibrationReport& report) {
  // The draft checkpoint the run must attach for a dflash2 entry: without
  // it the example would run a different strategy than the sweep used.
  const std::string draft_flag =
      (report.strategy == "dflash2" && !report.draft_path.empty())
          ? " --draft " + report.draft_path
          : "";
  // Both commands run at the memory-bound max context: the sweep tunes
  // speed knobs, and the bound is the cache size worth running with.
  ExampleCommands commands;
  commands.explicit_flags =
      "tessera-cli run --model " + report.model_path + draft_flag + " " +
      report.kv_flag + report.parallel_flags + "\\\n      --context " +
      std::to_string(report.max_context);
  commands.explicit_flags += " --prefill-chunk " +
                             std::to_string(report.chosen.prefill_chunk_tokens);
  if (report.split_applicable) {
    commands.explicit_flags += " --split-target " +
                               std::to_string(report.chosen.mxfp4_split_target);
    commands.explicit_flags +=
        " --split-cap " + std::to_string(report.chosen.mxfp4_split_cap);
  }
  if (report.attention_applicable) {
    commands.explicit_flags +=
        " --attention-split " + std::to_string(report.chosen.attention_split);
  }
  if (report.draft_attached) {
    commands.explicit_flags +=
        " --draft-block " + std::to_string(report.chosen.draft_tokens);
    commands.explicit_flags +=
        " --draft-context " + std::to_string(report.chosen.draft_context);
  }
  if (report.strategy == "mtp") {
    commands.explicit_flags += " --speculate";
  }
  commands.explicit_flags += " --prompt-text \"Hello\"";
  commands.from_file = "tessera-cli run --model " + report.model_path +
                       draft_flag + " " + report.kv_flag +
                       report.parallel_flags + "\\\n      --context " +
                       std::to_string(report.max_context) +
                       (report.strategy == "mtp" ? " --speculate" : "") +
                       " --calibration " + report.file +
                       " --prompt-text \"Hello\"";
  return commands;
}

std::string FormatCalibrationReport(const CalibrationReport& report) {
  const auto row = [](const std::string& label, const std::string& value,
                      std::size_t width) {
    std::string line = label;
    if (line.size() < width) {
      line.append(width - line.size(), ' ');
    }
    return line + value + "\n";
  };
  const auto default_for = [&](Setting setting) -> std::size_t {
    switch (setting) {
      case Setting::MxFp4SplitTarget:
        return report.defaults.mxfp4_split_target;
      case Setting::MxFp4SplitCap:
        return report.defaults.mxfp4_split_cap;
      case Setting::PrefillChunkTokens:
        return report.defaults.prefill_chunk_tokens;
      case Setting::AttentionSplit:
        return report.defaults.attention_split;
      case Setting::DraftTokens:
        return report.defaults.draft_tokens;
      case Setting::DraftContext:
        return report.defaults.draft_context;
    }
    return 0;
  };
  char rate[32];
  const auto format_rate = [&](double value) {
    std::snprintf(rate, sizeof(rate), "%.2f", value);
    return std::string(rate);
  };

  std::string out;
  out += "Tessera calibration report\n";
  out += "==========================\n";
  out += row("backend", report.backend, 14);
  out += row("device", report.device, 14);
  out += row("model", report.model, 14);
  out += row("kv cache", report.kv_type, 14);
  out += row("strategy", report.strategy, 14);
  out += row("parallel", report.parallel, 14);
  out += row("key context", std::to_string(report.key_context), 14);
  out += row("max context", std::to_string(report.max_context) + " token(s), " +
                                report.kv_type + " memory bound",
             14);
  out += "\nSweep\n-----\n";
  for (Setting setting :
       {Setting::MxFp4SplitTarget, Setting::MxFp4SplitCap,
        Setting::PrefillChunkTokens, Setting::AttentionSplit,
        Setting::DraftTokens, Setting::DraftContext}) {
    out += std::string(ToString(setting)) + "\n";
    bool any = false;
    for (const SweepPoint& point : report.points) {
      if (point.setting != setting) {
        continue;
      }
      any = true;
      std::string note;
      if (point.value == default_for(setting)) {
        note += " (default)";
      }
      if (point.kept) {
        note += " (kept)";
      }
      out += row("  " + std::to_string(point.value),
                 format_rate(point.tps) + " tok/s" + note, 8);
    }
    if (!any) {
      const bool skipped =
          std::find(report.skipped.begin(), report.skipped.end(), setting) !=
          report.skipped.end();
      out += skipped ? "  skipped (point budget)\n"
                     : "  not applicable on " + report.backend + "\n";
    }
  }
  out += "\nChosen\n------\n";
  out += row("mxfp4_split_target",
             std::to_string(report.chosen.mxfp4_split_target), 21);
  out += row("mxfp4_split_cap",
             std::to_string(report.chosen.mxfp4_split_cap), 21);
  out += row("prefill_chunk_tokens",
             std::to_string(report.chosen.prefill_chunk_tokens), 21);
  out += row("attention_split",
             std::to_string(report.chosen.attention_split), 21);
  out += row("draft_tokens",
             report.chosen.draft_tokens == 0
                 ? std::string("(none)")
                 : std::to_string(report.chosen.draft_tokens),
             21);
  out += row("draft_context",
             report.chosen.draft_context == 0
                 ? std::string("(checkpoint)")
                 : std::to_string(report.chosen.draft_context),
             21);
  out += row("decode", format_rate(report.decode_tps) + " tok/s", 21);
  out += row("prefill", format_rate(report.prefill_tps) + " tok/s", 21);
  out += "\nConcurrency\n-----------\n";
  if (report.concurrent.empty()) {
    out += "  not measured\n";
  } else {
    for (const ConcurrentRate& rate : report.concurrent) {
      out += row("  " + std::to_string(rate.sessions) + " session(s)",
                 format_rate(rate.total_tps) + " tok/s total, " +
                     format_rate(rate.per_session_tps) + " tok/s each",
                 20);
    }
  }
  out += row("max sessions",
             std::to_string(report.max_concurrent) + " at " +
                 std::to_string(report.key_context) + " key token(s)",
             20);
  out += "\nWrote " + report.file + "\n";
  const ExampleCommands commands = BuildExampleCommands(report);
  out += "\nExample\n-------\n";
  out += "  " + commands.explicit_flags + "\n";
  out += "  # or apply the saved entry directly:\n";
  out += "  " + commands.from_file + "\n";
  return out;
}

}  // namespace tessera
