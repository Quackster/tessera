#include "fetch_mtp_command.hpp"

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

#include "tessera/mtp_fetch.hpp"
#include "tessera/types.hpp"

// The fetch-mtp subcommand. It calls the library's inventory, fetch and
// verify entry points; all range reads, digest checks and resume logic live
// in models/qwen4exp/mtp_fetch.cpp (AGENTS.md, Where Things Live).

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;
// `verify`'s exit code: a tensor is missing or corrupt.
constexpr int kExitBad = 3;

void PrintFetchUsage() {
  std::fprintf(
      stderr,
      "usage: tessera-cli fetch-mtp --out <dir> [inventory|fetch|verify] "
      "[--only <substr>]\n"
      "  --out <dir>   directory the tensors and inventory are written under\n"
      "  inventory     read the checkpoint headers and write mtp-inventory.json\n"
      "  fetch         download the MTP tensors (default)\n"
      "  verify        check the fetched tensors against the pinned digests\n"
      "  --only <s>    limit fetch/verify to tensors whose name contains <s>\n");
}

}  // namespace

int RunFetchMtpCommand(int argc, char** argv) {
  std::string mode = "fetch";
  std::string out;
  std::string only;
  for (int i = 2; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--out" && i + 1 < argc) {
      out = argv[++i];
    } else if (arg == "--only" && i + 1 < argc) {
      only = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      PrintFetchUsage();
      return kExitOk;
    } else if (arg == "inventory" || arg == "fetch" || arg == "verify") {
      mode = std::string(arg);
    } else {
      std::fprintf(stderr, "fetch-mtp: unknown argument '%s'\n", argv[i]);
      PrintFetchUsage();
      return kExitUsage;
    }
  }
  if (out.empty()) {
    std::fprintf(stderr, "fetch-mtp: --out <dir> is required\n");
    PrintFetchUsage();
    return kExitUsage;
  }
  const std::filesystem::path dir(out);
  const tessera::MtpFetchOptions options{dir, only};
  if (mode == "inventory") {
    auto result = tessera::MtpInventory(options);
    if (!result) {
      std::fprintf(stderr, "fetch-mtp: inventory failed (%s)\n",
                   tessera::ToString(result.error()).data());
      return kExitError;
    }
    std::printf("fetch-mtp: wrote %s\n",
                (dir / "mtp-inventory.json").c_str());
    return kExitOk;
  }
  if (mode == "verify") {
    auto result = tessera::MtpVerify(dir);
    if (!result) {
      std::fprintf(stderr, "fetch-mtp: verify failed (%s)\n",
                   tessera::ToString(result.error()).data());
      return kExitError;
    }
    for (const std::string& name : *result) {
      std::fprintf(stderr, "fetch-mtp: MTP tensor missing or corrupt: %s\n",
                   name.c_str());
    }
    if (!result->empty()) {
      return kExitBad;
    }
    std::printf("fetch-mtp: all fetched tensors match the pinned digests\n");
    return kExitOk;
  }
  auto result = tessera::MtpFetch(options);
  if (!result) {
    std::fprintf(stderr, "fetch-mtp: fetch failed (%s)\n",
                 tessera::ToString(result.error()).data());
    return kExitError;
  }
  std::printf("fetch-mtp: wrote %s\n", (dir / "mtp-manifest.json").c_str());
  return kExitOk;
}

}  // namespace tessera::cli
