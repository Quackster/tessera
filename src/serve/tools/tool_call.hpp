#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"

// Qwen-style XML tool calls for the OpenAI chat endpoint. The model
// emits `<tool_call><function=name>...` blocks; clients send OpenAI-form
// history (`tool_calls` with JSON-string arguments, `role: tool`
// results). This module converts both directions.
namespace tessera::serve {

// One parsed model tool call. `arguments` is a compact JSON object.
struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments;
};

// Parse complete `<tool_call>...</tool_call>` blocks from generated
// `text`, in order. `text_before` receives the text before the first
// block (reasoning stripped, trimmed); the whole text when no block
// parses. Truncated and malformed blocks are skipped.
[[nodiscard]] std::vector<ToolCall> ParseToolCalls(std::string_view text,
                                                   std::string* text_before);

// The tool list from the request body to render, or nullptr for none.
// Sets `error` when `tools` is not an array or `tool_choice` is neither
// a string nor an object.
[[nodiscard]] const core::Json* EffectiveTools(const core::Json& body,
                                               std::string* error);

// Copy `body.messages` into `out`, converting assistant `tool_calls`
// arguments from JSON strings to objects for the chat template.
// Passes tool messages through (string content required). Sets `error`
// and returns false on malformed history.
[[nodiscard]] bool NormalizeToolHistory(const core::Json& body,
                                        core::Json* out, std::string* error);

}  // namespace tessera::serve
