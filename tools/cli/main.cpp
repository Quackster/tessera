#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>

#include "calibrate_command.hpp"
#include "cli_engine.hpp"
#include "cli_helpers.hpp"
#include "cli_options.hpp"
#include "fetch_mtp_command.hpp"
#include "run_command.hpp"
#include "serve_command.hpp"
#include "tessera/engine.hpp"

namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

}  // namespace

// The CLI front end: dispatch on the command word, parse the shared flags,
// and hand off to the command. All parsing lives in cli_options.hpp and all
// work lives in the command modules; main stays a dispatcher.
int main(int argc, char** argv) {
  if (argc < 2) {
    tessera::cli::PrintUsage();
    return kExitUsage;
  }
  const std::string_view command = argv[1];
  if (command == "--list-gpus") {
    auto names = tessera::ListGpuNames();
    if (!names) {
      std::fprintf(stderr, "cli: cannot list GPUs (%s)\n",
                   tessera::ToString(names.error()).data());
      return kExitError;
    }
    for (std::size_t i = 0; i < names->size(); ++i) {
      std::printf("gpu %zu: %s\n", i, (*names)[i].c_str());
    }
    return kExitOk;
  }
  if (command == "fetch-mtp") {
    return tessera::cli::RunFetchMtpCommand(argc, argv);
  }
  if (command != "run" && command != "serve" && command != "calibrate") {
    tessera::cli::PrintUsage();
    return kExitUsage;
  }
  std::vector<std::string_view> args;
  args.reserve(static_cast<std::size_t>(argc) - 2);
  for (int i = 2; i < argc; ++i) {
    args.emplace_back(argv[i]);
  }
  tessera::cli::CliOptions options;
  auto parsed = tessera::cli::ParseCliOptions(args, &options);
  if (!parsed) {
    // The parse error carries no prefix, so each command shows its usage.
    std::fprintf(stderr, "cli: %s\n", parsed.error().c_str());
    if (command == "calibrate") {
      tessera::cli::PrintCalibrateUsage();
    } else {
      tessera::cli::PrintUsage();
    }
    return kExitUsage;
  }
  tessera::cli::ApplyCliEnvDefaults(&options);
  if (command == "calibrate") {
    return tessera::cli::RunCalibrateCommand(options);
  }
  if (options.gpu_layers_help) {
    // Size the model without loading it. The flag needs --model for the
    // weight bytes and one device for the free-memory query.
    if (options.model_path.empty()) {
      std::fprintf(stderr, "cli: --model is required\n");
      tessera::cli::PrintUsage();
      return kExitUsage;
    }
    auto handled = tessera::cli::HandleGpuLayersHelp(options);
    if (handled) {
      return *handled;
    }
  }
  if (command == "serve") {
    return tessera::cli::RunServeCommand(options);
  }
  return tessera::cli::RunRunCommand(options);
}
