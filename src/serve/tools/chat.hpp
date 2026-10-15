#pragma once

#include <cstddef>
#include <expected>
#include <functional>

#include "core/json.hpp"
#include "tessera/engine.hpp"
#include "tessera/types.hpp"

// Tool-aware OpenAI chat handler: renders the request tools into the
// model template, generates one turn, and answers with text or with
// OpenAI-form `tool_calls`.
namespace tessera {

class Model;
class Tokenizer;

namespace serve {

class ResponseWriter;
struct SessionMessage;

// One turn's generation, decoupled from the device: a single-device lease
// supplies its engine's GenerateStreaming, a split runner supplies the
// pool's least-loaded Submit. Both stream tokens and honor the callback's
// false return.
using ToolGenerate = std::function<std::expected<GenerateOutcome, StatusCode>(
    const GenerateOptions&, const std::function<bool(std::uint32_t)>&)>;

// True when the body carries a non-empty tool list that is not disabled
// by `tool_choice: "none"`.
[[nodiscard]] bool WantsTools(const core::Json& body);

// Run one tool-aware chat turn and write the OpenAI chat response
// (buffered or SSE). Falls back to a plain text answer when the model
// emits no tool call. `generate` runs the turn on the serving backend
// (single device or layer split); `model` backs the chat template and the
// context limit. `kv_type` is the runner's KV storage type, read onto the
// request like every other serve turn. `out_turn` (null to skip) receives
// the assistant turn for session history: text content, reasoning and the
// OpenAI-form tool_calls dump. True when the turn generated (and
// `out_turn`, when given, is valid); the error response is already sent
// otherwise.
[[nodiscard]] bool ChatWithTools(const ToolGenerate& generate, Model& model,
                                 const Tokenizer& tokenizer,
                                 ResponseWriter& writer,
                                 const core::Json& body,
                                 std::size_t default_max,
                                 SessionMessage* out_turn,
                                 KvCacheType kv_type);

}  // namespace serve

}  // namespace tessera
