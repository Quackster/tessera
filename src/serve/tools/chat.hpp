#pragma once

#include <cstddef>

#include "core/json.hpp"

// Tool-aware OpenAI chat handler: renders the request tools into the
// model template, generates one turn, and answers with text or with
// OpenAI-form `tool_calls`.
namespace tessera {

class Engine;
class Model;
class Tokenizer;

namespace serve {

class ResponseWriter;
struct SessionMessage;

// True when the body carries a non-empty tool list that is not disabled
// by `tool_choice: "none"`.
[[nodiscard]] bool WantsTools(const core::Json& body);

// Run one tool-aware chat turn and write the OpenAI chat response
// (buffered or SSE). Falls back to a plain text answer when the model
// emits no tool call. `out_turn` (null to skip) receives the assistant
// turn for session history: text content, reasoning and the OpenAI-form
// tool_calls dump. True when the turn generated (and `out_turn`, when
// given, is valid); the error response is already sent otherwise.
[[nodiscard]] bool ChatWithTools(Engine& engine, Model& model,
                                const Tokenizer& tokenizer,
                                ResponseWriter& writer, const core::Json& body,
                                std::size_t default_max,
                                SessionMessage* out_turn);

}  // namespace serve

}  // namespace tessera
