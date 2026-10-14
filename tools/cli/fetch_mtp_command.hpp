#pragma once

// The `fetch-mtp` subcommand: download the Qwen3.8-Flash-Next MTP head from
// the BF16 checkpoint over HTTP. All logic lives in the library
// (models/qwen4exp/mtp_fetch.hpp); this is the thin front end.

namespace tessera::cli {

// Run `tessera-cli fetch-mtp ...`. `argv[1]` is "fetch-mtp" and the flags
// follow. Returns a process exit code (3 when `verify` finds a bad tensor).
int RunFetchMtpCommand(int argc, char** argv);

}  // namespace tessera::cli
