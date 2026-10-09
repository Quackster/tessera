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
// parses. `reasoning` (null to skip) receives the `<think>` span text
// (trimmed), so the client can still see the reasoning that never
// belongs in the reply content. Truncated and malformed blocks are
// skipped.
[[nodiscard]] std::vector<ToolCall> ParseToolCalls(std::string_view text,
                                                    std::string* text_before,
                                                    std::string* reasoning = nullptr);

// Incremental think/content splitter for live session streaming.
// Feed detokenized pieces in order; each push returns the new
// reasoning/content text since the last call, each word exactly once
// in exactly one field. While the prompt ends with an open <think>
// (think_expected) and no closer arrived, all text is pending thought
// and no content deltas flow. A trailing unclosed span is likewise
// withheld until it closes (or Finish flushes it), so span markup
// never leaks into the content. Finish flushes the remainder, so
// concatenated deltas always equal SplitThink(all).
class ThinkStreamer {
 public:
  explicit ThinkStreamer(bool think_expected);
  struct Deltas {
    std::string reasoning;
    std::string content;
  };
  [[nodiscard]] Deltas Push(std::string_view piece);
  // End-of-turn remainder (an unclosed trailing span stays content,
  // same as the post-hoc split).
  [[nodiscard]] Deltas Finish();

 private:
  const bool think_expected_;
  bool resolved_ = false;
  bool withhold_ = true;
  std::string raw_;
  std::string emitted_reasoning_;
  std::string emitted_content_;
};

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
