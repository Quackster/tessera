#include "run_command.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <expected>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cli_common.hpp"
#include "cli_engine.hpp"
#include "cli_helpers.hpp"
#include "run_single.hpp"
#include "tessera/context_parallel.hpp"
#include "tessera/engine.hpp"
#include "tessera/hybrid.hpp"
#include "tessera/pipeline.hpp"
#include "tessera/spec_split.hpp"
#include "tessera/speculative.hpp"
#include "tessera/tensor_parallel.hpp"
#include "tessera/tokenizer.hpp"
#include "tessera/types.hpp"

namespace tessera::cli {
namespace {

constexpr int kExitOk = 0;
constexpr int kExitError = 1;
constexpr int kExitUsage = 2;

// The flags a route cannot honor: accepted (one parser for every
// command) but reported once instead of ignored in silence.
void WarnInertFlags(const CliOptions& options) {
  const bool split = SplitLayoutActive(options);
  if (options.hybrid_mode) {
    if (options.speculate || !options.draft_path.empty()) {
      WarnIgnoredFlag("--speculate/--draft",
                      "--hybrid runs greedy without a speculative strategy");
    }
  } else if (options.dp_groups != 1) {
    WarnIgnoredFlag("--dp", "--dp selects the groups of --hybrid");
  }
  if (!options.pipeline_mode && !options.pipeline_split.empty()) {
    WarnIgnoredFlag("--pipeline-split", "--pipeline-split needs --pipeline");
  }
  if ((options.tp_mode || options.cp_mode || options.hybrid_mode) &&
      options.gpu_layers != 0) {
    WarnIgnoredFlag("--gpu-layers",
                    "--gpu-layers pages the single-device, --pipeline and "
                    "--spec-split routes");
  }
  if (split && (!options.image_path.empty() || !options.mmproj_path.empty())) {
    WarnIgnoredFlag("--image/--mmproj",
                    "the vision path runs on the single-device route");
  }
}

// One greedy generation through a split layout's streaming face, decoded
// to stdout as it arrives. Shared by the pipeline and spec-split routes.
int StreamLayout(
    GenerateOptions& request,
    const std::function<std::expected<GenerateOutcome, StatusCode>(
        GenerateOptions&, const std::function<bool(std::uint32_t)>&)>& run,
    const Tokenizer& tokenizer) {
  auto streamed = run(request, [&](std::uint32_t token) {
    auto piece = tokenizer.Decode(std::span<const std::uint32_t>(&token, 1));
    if (piece) {
      std::fwrite(piece->data(), 1, piece->size(), stdout);
      std::fflush(stdout);
    }
    return true;
  });
  if (!streamed) {
    std::fprintf(stderr, "cli: generation failed (%s)\n",
                 ToString(streamed.error()).data());
    return kExitError;
  }
  std::fputc('\n', stdout);
  std::fflush(stdout);
  return kExitOk;
}

}  // namespace

int RunRunCommand(const CliOptions& options) {
  if (options.model_path.empty()) {
    std::fprintf(stderr, "cli: --model is required\n");
    PrintUsage();
    return kExitUsage;
  }
  if (auto valid = ValidateLayoutFlags(options); !valid) {
    std::fprintf(stderr, "cli: %s\n", valid.error().c_str());
    PrintUsage();
    return kExitUsage;
  }
  WarnInertFlags(options);

  const std::vector<int> devices = SelectedDevices(options);
  const std::string routing_hint = DeviceRoutingHint(
      devices, DeviceRouteKind::kRunSingle, SplitLayoutActive(options));
  if (!routing_hint.empty()) {
    std::fprintf(stderr, "%s\n", routing_hint.c_str());
  }

  if (options.pipeline_mode) {
    // Layer-split the model across the selected devices for one request.
    if (options.prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --pipeline needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = options.prompt_texts.front();
    auto pipeline = Pipeline::Create(MakeCliPipelineOptions(options, devices));
    if (!pipeline) {
      std::fprintf(stderr, "cli: pipeline create failed (%s)\n",
                   ToString(pipeline.error()).data());
      return kExitError;
    }
    // Attach the drafter to the pipeline: --draft selects DFlash2, else
    // --speculate selects MTP on the last stage, which owns the MTP head.
    if (options.speculate || !options.draft_path.empty()) {
      const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
        return MakeCliStrategy(options);
      };
      auto attached =
          (*pipeline)->AttachSpeculative(factory, CliStrategyOptions(options));
      if (!attached) {
        std::fprintf(stderr, "cli: pipeline strategy attach failed (%s)\n",
                     ToString(attached.error()).data());
        return kExitError;
      }
    }
    for (std::size_t stage = 0; stage < (*pipeline)->StageCount(); ++stage) {
      ApplyCliLogLevel((*pipeline)->StageEngine(stage), options);
    }
    const Tokenizer* tokenizer = (*pipeline)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto ids = EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
      return (*pipeline)->ChatPrompt(prompt_text, options.enable_thinking,
                                     options.reasoning_effort);
    });
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed (%s)\n",
                   ToString(ids.error()).data());
      return kExitError;
    }
    GenerateOptions gen;
    gen.prompt_tokens = std::move(*ids);
    gen.max_completion_tokens = options.max_completion_tokens;
    gen.kv_type = options.kv_type;
    // Stream over the pipeline so each token reaches the console as it is
    // generated. A blocking Generate would return only after every step
    // (by default the whole remaining context), which looks hung after
    // the prefill line.
    return StreamLayout(
        gen,
        [&](GenerateOptions& request,
            const std::function<bool(std::uint32_t)>& on_token) {
          return (*pipeline)->GenerateStreaming(request, on_token);
        },
        *tokenizer);
  }

  if (options.tp_mode && !options.hybrid_mode) {
    // Tensor-parallel: split the attention and FFN across the selected
    // devices and run greedy generation with a per-rank thread and the
    // rank reduce. MoE routed experts split by expert dim on the same
    // ranks; each rank runs its local experts and reduces the sum.
    if (options.prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --tp needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = options.prompt_texts.front();
    TensorParallelOptions tp_options;
    tp_options.model_path = options.model_path;
    tp_options.devices = devices;
    tp_options.context = options.context;
    tp_options.kv_type = options.kv_type;
    auto tp = TensorParallel::Create(tp_options);
    if (!tp) {
      std::fprintf(stderr, "cli: tensor-parallel create failed (%s)\n",
                   ToString(tp.error()).data());
      return kExitError;
    }
    if (options.speculate || !options.draft_path.empty()) {
      const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
        return MakeCliStrategy(options);
      };
      auto attached =
          (*tp)->AttachSpeculative(factory, CliStrategyOptions(options));
      if (!attached) {
        std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                     ToString(attached.error()).data());
        return kExitError;
      }
    }
    for (std::size_t rank = 0; rank < (*tp)->RankCount(); ++rank) {
      ApplyCliLogLevel((*tp)->RankEngine(rank), options);
    }
    const Tokenizer* tokenizer = (*tp)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto ids = EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
      return (*tp)->ChatPrompt(prompt_text, options.enable_thinking,
                               options.reasoning_effort);
    });
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed (%s)\n",
                   ToString(ids.error()).data());
      return kExitError;
    }
    GenerateOptions gen;
    gen.prompt_tokens = std::move(*ids);
    gen.max_completion_tokens = options.max_completion_tokens;
    gen.kv_type = options.kv_type;
    return StreamLayout(
        gen,
        [&](GenerateOptions& request,
            const std::function<bool(std::uint32_t)>& on_token) {
          return (*tp)->GenerateStreaming(request.prompt_tokens, request,
                                          on_token);
        },
        *tokenizer);
  }

  if (options.cp_mode && !options.hybrid_mode) {
    // Context-parallel: split the key range across the selected devices
    // and run greedy generation with a per-rank thread and a per-layer
    // partial combine.
    if (options.prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --cp needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = options.prompt_texts.front();
    ContextParallelOptions cp_options;
    cp_options.model_path = options.model_path;
    cp_options.devices = devices;
    cp_options.context = options.context;
    cp_options.kv_type = options.kv_type;
    auto cp = ContextParallel::Create(cp_options);
    if (!cp) {
      std::fprintf(stderr, "cli: context-parallel create failed (%s)\n",
                   ToString(cp.error()).data());
      return kExitError;
    }
    if (options.speculate || !options.draft_path.empty()) {
      const auto factory = [&]() -> std::unique_ptr<SpeculativeStrategy> {
        return MakeCliStrategy(options);
      };
      auto attached =
          (*cp)->AttachSpeculative(factory, CliStrategyOptions(options));
      if (!attached) {
        std::fprintf(stderr, "cli: strategy attach failed (%s)\n",
                     ToString(attached.error()).data());
        return kExitError;
      }
    }
    for (std::size_t rank = 0; rank < (*cp)->RankCount(); ++rank) {
      ApplyCliLogLevel((*cp)->RankEngine(rank), options);
    }
    const Tokenizer* tokenizer = (*cp)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto ids = EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
      return (*cp)->ChatPrompt(prompt_text, options.enable_thinking,
                               options.reasoning_effort);
    });
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed (%s)\n",
                   ToString(ids.error()).data());
      return kExitError;
    }
    GenerateOptions gen;
    gen.prompt_tokens = std::move(*ids);
    gen.max_completion_tokens = options.max_completion_tokens;
    gen.kv_type = options.kv_type;
    return StreamLayout(
        gen,
        [&](GenerateOptions& request,
            const std::function<bool(std::uint32_t)>& on_token) {
          return (*cp)->GenerateStreaming(request.prompt_tokens, request,
                                          on_token);
        },
        *tokenizer);
  }

  if (options.spec_split_mode) {
    // Speculative split: the target runs on the first --gpus device and
    // the MTP draft head on the second.
    if (options.prompt_texts.size() != 1) {
      std::fprintf(stderr,
                   "cli: --spec-split needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = options.prompt_texts.front();
    SpecSplitOptions split_options;
    split_options.model_path = options.model_path;
    split_options.devices = devices;
    split_options.context = options.context;
    split_options.kv_type = options.kv_type;
    split_options.draft_block = options.draft_block;
    split_options.prefill_chunk = options.prefill_chunk;
    split_options.mtp_path = options.mtp_path;
    split_options.gpu_layers = options.gpu_layers;
    auto split = SpecSplit::Create(split_options);
    if (!split) {
      std::fprintf(stderr, "cli: speculative-split create failed (%s)\n",
                   ToString(split.error()).data());
      return kExitError;
    }
    ApplyCliLogLevel((*split)->TargetEngine(), options);
    const Tokenizer* tokenizer = (*split)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    auto ids = EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
      return (*split)->ChatPrompt(prompt_text, options.enable_thinking,
                                  options.reasoning_effort);
    });
    if (!ids) {
      std::fprintf(stderr, "cli: prompt encode failed (%s)\n",
                   ToString(ids.error()).data());
      return kExitError;
    }
    GenerateOptions gen;
    gen.prompt_tokens = std::move(*ids);
    gen.max_completion_tokens = options.max_completion_tokens;
    gen.kv_type = options.kv_type;
    return StreamLayout(
        gen,
        [&](GenerateOptions& request,
            const std::function<bool(std::uint32_t)>& on_token) {
          return (*split)->GenerateStreaming(request.prompt_tokens, request,
                                             on_token);
        },
        *tokenizer);
  }

  if (options.hybrid_mode) {
    // Hybrid: data-parallel groups of model-parallel teams behind one
    // device mesh. The one prompt runs on every group; --tp/--cp selects
    // the model-parallel route inside each multi-rank group.
    if (options.prompt_texts.size() != 1) {
      std::fprintf(stderr, "cli: --hybrid needs exactly one --prompt-text\n");
      return kExitUsage;
    }
    const std::string& prompt_text = options.prompt_texts.front();
    HybridOptions hybrid_options;
    hybrid_options.model_path = options.model_path;
    hybrid_options.devices = devices;
    hybrid_options.dp_groups = options.dp_groups;
    hybrid_options.mp_kind =
        options.cp_mode ? HybridMpKind::Context : HybridMpKind::Tensor;
    hybrid_options.context = options.context;
    hybrid_options.kv_type = options.kv_type;
    auto hybrid = Hybrid::Create(hybrid_options);
    if (!hybrid) {
      std::fprintf(stderr, "cli: hybrid create failed (%s)\n",
                   ToString(hybrid.error()).data());
      return kExitError;
    }
    const Tokenizer* tokenizer = (*hybrid)->GetTokenizer();
    if (tokenizer == nullptr) {
      std::fprintf(stderr, "cli: this model has no tokenizer\n");
      return kExitError;
    }
    std::vector<std::vector<std::uint32_t>> prompts;
    prompts.reserve(options.dp_groups);
    for (std::size_t group = 0; group < options.dp_groups; ++group) {
      auto encoded =
          EncodePromptText(prompt_text, options.no_chat, *tokenizer, [&] {
            return (*hybrid)->ChatPrompt(prompt_text, options.enable_thinking,
                                         options.reasoning_effort);
          });
      if (!encoded) {
        std::fprintf(stderr, "cli: prompt encode failed\n");
        return kExitError;
      }
      prompts.push_back(std::move(*encoded));
    }
    auto outputs =
        (*hybrid)->GenerateMany(prompts, options.max_completion_tokens);
    if (!outputs) {
      std::fprintf(stderr, "cli: hybrid generation failed (%s)\n",
                   ToString(outputs.error()).data());
      return kExitError;
    }
    for (std::size_t i = 0; i < outputs->size(); ++i) {
      auto text = tokenizer->Decode((*outputs)[i]);
      std::fprintf(stdout, "batch %zu: %s\n", i, text ? text->c_str() : "");
    }
    std::fflush(stdout);
    return kExitOk;
  }

  return RunSingleDeviceGeneration(options);
}

}  // namespace tessera::cli
