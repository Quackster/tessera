#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "tessera/calibrate.hpp"
#include "tessera/engine.hpp"
#include "tessera/runner.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

// Shared front-end helpers used by the run, serve and calibrate commands:
// the model summary line, the usage text, and the calibration lookup. The
// heavy logic stays in the library.

namespace tessera {

class Engine;
class Model;

namespace cli {

// One summary line plus the first few tensor names for a loaded model, so
// both `run` and the deferred `serve` loader report it identically.
void LogModelSummary(Engine& engine, Model& model);

// The run/serve usage text (also points at the calibrate command).
void PrintUsage();

// One stderr line for a flag a command accepts but cannot honor on the
// selected route: "cli: <flag> ignored: <reason>". One formatter, so every
// command reports an inert flag the same way.
void WarnIgnoredFlag(std::string_view flag, std::string_view reason);

// On-disk bytes of a model path: the file size for a GGUF, the sum of
// the regular files for an MXFP4 directory. Used to refuse a device or a
// host that cannot hold the weights before the load starts.
[[nodiscard]] std::uint64_t PathSizeBytes(const std::string& path);

// Render one source prompt through a chat template and encode it. A model
// without a chat template (UnsupportedFeature) encodes the raw text, so a
// --no-chat or template-less run still works; a template that exists but
// fails to render is an error, so a thinking request never silently runs
// as raw text. `render` runs the model's or layout's template with the
// thinking flags already applied.
[[nodiscard]] std::expected<std::vector<std::uint32_t>, StatusCode>
EncodePromptText(
    std::string_view source, bool no_chat, const Tokenizer& tokenizer,
    const std::function<std::expected<std::string, StatusCode>()>& render);

// Fill `*sampling` with the per-architecture recommended sampler for the
// loaded model and report whether the caller should enable sampling. A
// thinking model loops under greedy decode, so its architecture (keyed by
// general.architecture and the thinking mode) supplies a sampler. Does
// nothing and returns false when the user passed explicit sampling flags
// or the architecture has no recommendation.
[[nodiscard]] bool ApplyRecommendedSampling(const Model& model, bool thinking,
                                            bool explicit_sampling,
                                            SamplingOptions* sampling);

// Resolve the calibration configuration for a loaded model: build the
// hardware key, look the entry up in the calibration file, and merge it
// under the caller's explicit values (an explicit nonzero value wins). A
// missing file is not an error (no calibration yet); a malformed file is
// reported and ignored.
[[nodiscard]] CalibrationConfig ResolveCalibration(
    Engine& engine, Model& model, std::size_t context, KvCacheType kv_type,
    std::string_view strategy, const std::string& path,
    const CalibrationConfig& explicit_config);

// Resolve `--gpu-layers` for a model before it is loaded. An explicit
// nonzero value always wins; otherwise the calibration file's stored value
// for the model's key applies. This runs before Model::Load because paging
// is a load-time setting, so the model identity comes from its metadata
// (Model::ProbeName), not a load. A missing file or key, an unreadable
// path, or an empty `path` leaves the explicit value. An MoE target larger
// than VRAM needs the nonzero result to load at all.
[[nodiscard]] std::size_t ResolveCalibrationGpuLayers(
    Engine& engine, const std::string& model_path, std::size_t context,
    KvCacheType kv_type, std::string_view strategy, const std::string& path,
    std::size_t explicit_layers);

// Print the `--gpu-layers help` sizing advice for a model path without
// loading it: the offloadable layer count, the KV cache cost at
// `context`, and the largest fitting layer count on this engine's
// device. `kv_offload` keeps the full-attention KV cache in system RAM,
// so the advice does not reserve device space for it. Returns the
// process exit code (0 on advice, 1 on error).
int LogGpuLayersAdvice(Engine& engine, const std::string& model_path,
                       std::size_t context, KvCacheType kv_type,
                       std::string_view strategy, bool kv_offload = false);

// Refuse a paged load whose resident set plus KV cache cannot fit the
// engine's device before the load starts, logging what is needed and
// what fits. `kv_offload` keeps the full-attention KV cache in system
// RAM, so it is not counted against the device. True when the load may
// proceed (or when the backend reports no memory size, so nothing can
// be judged). `device` names the GPU in the refusal message.
bool CheckGpuLayersFit(Engine& engine, const std::string& model_path,
                       std::size_t context, KvCacheType kv_type,
                       std::string_view strategy, std::size_t gpu_layers,
                       int device, bool kv_offload = false);

// Refuse a replica load whose paged weights cannot fit system RAM, before
// the load starts. `held_bytes` is the paged host bytes of the replicas
// already loaded; on a proceed result `*replica_bytes` is this replica's
// paged host bytes, so the caller can accumulate across devices. Returns
// true when the load may proceed: a full upload holds no host weights, and
// an unreportable host figure skips the check. `strategy` selects the MTP
// head the same way the fit check does; `device` names the GPU.
bool CheckHostRamFit(Engine& engine, const std::string& model_path,
                     std::size_t gpu_layers, std::string_view strategy,
                     std::uint64_t held_bytes, int device,
                     std::uint64_t* replica_bytes);

// One concurrent CLI session: the streamed tokens plus the outcome.
// `outcome` holds the error when the session failed.
struct ConcurrentSessionResult {
  std::vector<std::uint32_t> tokens;
  std::expected<GenerateOutcome, StatusCode> outcome{
      std::unexpected(StatusCode::InvalidArgument)};
};

// Run each request on the pool at once, streaming per session. The
// shared scheduler time-slices the device, so two sessions decode
// together instead of one waiting for the other. `on_token` runs on
// the session thread and must be safe across sessions.
// Empty requests return empty results.
[[nodiscard]] std::vector<ConcurrentSessionResult> RunConcurrentSessions(
    const RunnerPool& pool, std::span<const GenerateOptions> requests,
    std::function<void(std::size_t, std::uint32_t)> on_token = {});

}  // namespace cli
}  // namespace tessera
