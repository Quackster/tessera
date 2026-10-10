#include "serve/tools/tool_call.hpp"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tessera::serve {

namespace {

// Qwen tool-call XML markers emitted by the model.
constexpr char kCallOpen[] = "<tool_call>";
constexpr char kCallClose[] = "</tool_call>";
constexpr char kFunctionOpen[] = "<function=";
constexpr char kFunctionClose[] = "</function>";
constexpr char kParameterOpen[] = "<parameter=";
constexpr char kParameterClose[] = "</parameter>";
constexpr char kThinkOpen[] = "<think>";
constexpr char kThinkClose[] = "</think>";
constexpr char kCallIdPrefix[] = "call_";
constexpr std::size_t kMissing = std::string_view::npos;

std::string Trim(std::string_view text) {
  std::size_t begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string_view::npos) {
    return {};
  }
  return std::string(
      text.substr(begin, text.find_last_not_of(" \t\r\n") - begin + 1));
}

// Split <think>...</think> spans from the reply: reasoning holds the
// concatenated inner text, text the remainder. Reasoning must not leak
// into the reply content or the next turn's history, but the client
// still sees it through `reasoning_content`. The generation prompt
// already ends with an open <think>, so a leading span may have no
// opener: everything through its closer counts as reasoning. A span
// without a closer stays in the text (an unfinished turn keeps its
// words, same as before).
struct ThinkSplit {
  std::string reasoning;
  std::string text;
};

ThinkSplit SplitThink(std::string_view input) {
  ThinkSplit split{std::string(), std::string(input)};
  while (true) {
    const std::size_t close = split.text.find(kThinkClose);
    if (close == std::string::npos) {
      return split;
    }
    std::size_t begin = 0;
    std::size_t inner = 0;
    if (close > 0) {
      const std::size_t open = split.text.rfind(kThinkOpen, close - 1);
      if (open != std::string::npos) {
        begin = open;
        inner = open + sizeof(kThinkOpen) - 1;
      }
    }
    split.reasoning += split.text.substr(inner, close - inner);
    split.text.erase(begin, close + sizeof(kThinkClose) - 1 - begin);
  }
}

// A parameter body is JSON when it parses (numbers, booleans, objects);
// otherwise it is a plain string.
core::Json ValueFromText(const std::string& body) {
  if (auto parsed = core::Json::Parse(body)) {
    return *parsed;
  }
  return core::Json::String(body);
}

// Parse one block between the call markers; nullopt when malformed.
std::optional<ToolCall> ParseBlock(std::string_view block,
                                   std::size_t index) {
  const std::size_t function = block.find(kFunctionOpen);
  if (function == kMissing) {
    return std::nullopt;
  }
  const std::size_t name_begin = function + sizeof(kFunctionOpen) - 1;
  const std::size_t name_end = block.find('>', name_begin);
  if (name_end == kMissing) {
    return std::nullopt;
  }
  const std::string name = Trim(block.substr(name_begin, name_end - name_begin));
  if (name.empty() || block.find(kFunctionClose, name_end) == kMissing) {
    return std::nullopt;
  }
  core::Json arguments = core::Json::Object();
  std::size_t pos = name_end + 1;
  while (true) {
    const std::size_t open = block.find(kParameterOpen, pos);
    if (open == kMissing) {
      break;
    }
    const std::size_t key_begin = open + sizeof(kParameterOpen) - 1;
    const std::size_t key_end = block.find('>', key_begin);
    const std::size_t close =
        key_end == kMissing
            ? kMissing
            : block.find(kParameterClose, key_end);
    if (key_end == kMissing || close == kMissing) {
      return std::nullopt;
    }
    const std::string key =
        Trim(block.substr(key_begin, key_end - key_begin));
    if (key.empty()) {
      return std::nullopt;
    }
    arguments.Set(key, ValueFromText(Trim(
                              block.substr(key_end + 1, close - key_end - 1))));
    pos = close + sizeof(kParameterClose) - 1;
  }
  ToolCall call;
  call.id = std::string(kCallIdPrefix) + std::to_string(index);
  call.name = name;
  call.arguments = arguments.Dump();
  return call;
}

// Convert one OpenAI-form history call to template form (arguments as
// an object). Copies id/type through untouched.
bool NormalizeCall(const core::Json& call, core::Json* out,
                   std::string* error) {
  if (!call.isObject()) {
    *error = "tool_calls entries must be objects";
    return false;
  }
  const core::Json* function = call.Find("function");
  if (function == nullptr || !function->isObject()) {
    *error = "tool_calls entries need a function object";
    return false;
  }
  const core::Json* name = function->Find("name");
  if (name == nullptr || !name->isString() || name->AsString().empty()) {
    *error = "tool calls need a function name";
    return false;
  }
  core::Json fixed_function = core::Json::Object();
  for (const auto& [key, value] : function->AsObject()) {
    if (key != "arguments") {
      fixed_function.Set(key, value);
    }
  }
  const core::Json* arguments = function->Find("arguments");
  if (arguments == nullptr) {
    fixed_function.Set("arguments", core::Json::Object());
  } else if (arguments->isObject()) {
    fixed_function.Set("arguments", *arguments);
  } else if (arguments->isString()) {
    auto parsed = core::Json::Parse(arguments->AsString());
    if (parsed == nullptr || !parsed->isObject()) {
      *error = "tool call arguments must parse to an object";
      return false;
    }
    fixed_function.Set("arguments", *parsed);
  } else {
    *error = "tool call arguments must be an object or a JSON string";
    return false;
  }
  core::Json entry = core::Json::Object();
  for (const auto& [key, value] : call.AsObject()) {
    if (key != "function") {
      entry.Set(key, value);
    }
  }
  entry.Set("function", std::move(fixed_function));
  *out = std::move(entry);
  return true;
}

}  // namespace

std::vector<ToolCall> ParseToolCalls(std::string_view text,
                                      std::string* text_before,
                                      std::string* reasoning) {
  const ThinkSplit split = SplitThink(text);
  const std::string_view body = split.text;
  if (reasoning != nullptr) {
    *reasoning = Trim(split.reasoning);
  }
  std::vector<ToolCall> calls;
  std::size_t pos = 0;
  bool first = true;
  while (true) {
    const std::size_t open = body.find(kCallOpen, pos);
    if (open == kMissing) {
      break;
    }
    if (first) {
      *text_before = Trim(body.substr(0, open));
      first = false;
    }
    const std::size_t close = body.find(kCallClose, open);
    if (close == kMissing) {
      break;
    }
    const std::size_t begin = open + sizeof(kCallOpen) - 1;
    if (auto call = ParseBlock(body.substr(begin, close - begin),
                               calls.size())) {
      calls.push_back(std::move(*call));
    }
    pos = close + sizeof(kCallClose) - 1;
  }
  if (first) {
    *text_before = Trim(split.text);
  }
  return calls;
}

bool ThinkExpected(std::string_view prompt, bool enable_thinking) {
  if (!enable_thinking) {
    return false;
  }
  std::size_t end = prompt.size();
  while (end > 0 && (prompt[end - 1] == ' ' || prompt[end - 1] == '\t' ||
                     prompt[end - 1] == '\r' || prompt[end - 1] == '\n')) {
    --end;
  }
  constexpr std::string_view kOpen = "<think>";
  return end >= kOpen.size() &&
         prompt.substr(end - kOpen.size(), kOpen.size()) == kOpen;
}

ThinkStreamer::ThinkStreamer(bool think_expected)
    : think_expected_(think_expected) {}

namespace {

// Shared leading bytes of the emitted text and the new split; the
// delta starts after them. A closing span can only remove markup, so
// the common prefix never un-emits words.
std::size_t CommonPrefix(std::string_view emitted, std::string_view text) {
  std::size_t n = 0;
  while (n < emitted.size() && n < text.size() && emitted[n] == text[n]) {
    ++n;
  }
  return n;
}

// A tail that may still grow into a think tag: a proper prefix of
// an opener or a closer. Complete openers use the unclosed span
// check in Push; this covers their split pieces ("<", "<t", "</" and
// so on). Anything else (like "a < b") streams at once.
bool IsThinkTagPrefix(std::string_view tail) {
  constexpr std::string_view kOpen = "<think>";
  constexpr std::string_view kClose = "</think>";
  const auto is_prefix = [](std::string_view tag, std::string_view text) {
    return text.size() < tag.size() &&
           tag.substr(0, text.size()) == text;
  };
  return !tail.empty() &&
         (is_prefix(kOpen, tail) || is_prefix(kClose, tail));
}

}  // namespace

ThinkStreamer::Deltas ThinkStreamer::Push(std::string_view piece) {
  raw_.append(piece);
  if (think_expected_ && !resolved_) {
    if (raw_.find(kThinkClose) == std::string::npos) {
      return {};
    }
    resolved_ = true;
  }
  const ThinkSplit split = SplitThink(raw_);
  // A trailing unclosed span may yet close: withhold it from the
  // content so span markup never leaks and its words never stream
  // twice (once raw, once as reasoning). The same holds for a split
  // opener still arriving piece by piece.
  std::string_view visible = split.text;
  if (withhold_) {
    const std::size_t open = split.text.rfind(kThinkOpen);
    if (open != std::string::npos &&
        split.text.find(kThinkClose, open) == std::string::npos) {
      visible = std::string_view(split.text).substr(0, open);
    } else {
      const std::size_t bracket = split.text.rfind('<');
      if (bracket != std::string::npos &&
          IsThinkTagPrefix(std::string_view(split.text).substr(bracket))) {
        visible = std::string_view(split.text).substr(0, bracket);
      }
    }
  }
  Deltas deltas;
  // Reasoning only gains closed spans at the end; visible content can
  // only grow past the withheld tail, so it diffs by common prefix
  // and keeps already streamed words.
  deltas.reasoning =
      split.reasoning.substr(CommonPrefix(emitted_reasoning_, split.reasoning));
  emitted_reasoning_ += deltas.reasoning;
  deltas.content =
      std::string(visible).substr(CommonPrefix(emitted_content_, visible));
  emitted_content_ += deltas.content;
  return deltas;
}

ThinkStreamer::Deltas ThinkStreamer::Finish() {
  resolved_ = true;
  withhold_ = false;
  return Push({});
}

const core::Json* EffectiveTools(const core::Json& body,
                                 std::string* error) {
  const core::Json* tools = body.Find("tools");
  if (tools == nullptr) {
    return nullptr;
  }
  if (!tools->isArray()) {
    *error = "tools must be an array";
    return nullptr;
  }
  if (tools->AsArray().empty()) {
    return nullptr;
  }
  if (const core::Json* choice = body.Find("tool_choice");
      choice != nullptr) {
    if (choice->isString() && choice->AsString() == "none") {
      return nullptr;
    }
    if (!choice->isString() && !choice->isObject()) {
      *error = "tool_choice must be a string or an object";
      return nullptr;
    }
  }
  return tools;
}

bool NormalizeToolHistory(const core::Json& body, core::Json* out,
                          std::string* error) {
  const core::Json* messages = body.Find("messages");
  if (messages == nullptr || !messages->isArray()) {
    *error = "messages must be an array";
    return false;
  }
  core::Json items = core::Json::Array();
  for (const core::Json& message : messages->AsArray()) {
    if (!message.isObject()) {
      *error = "messages entries must be objects";
      return false;
    }
    const core::Json* role = message.Find("role");
    if (role == nullptr || !role->isString()) {
      *error = "messages entries need a string role";
      return false;
    }
    if (role->AsString() == "tool") {
      const core::Json* content = message.Find("content");
      if (content == nullptr || !content->isString()) {
        *error = "tool messages need string content";
        return false;
      }
      items.Push(message);
      continue;
    }
    if (role->AsString() == "assistant") {
      const core::Json* calls = message.Find("tool_calls");
      if (calls != nullptr && calls->isArray()) {
        core::Json fixed = core::Json::Object();
        for (const auto& [key, value] : message.AsObject()) {
          if (key != "tool_calls") {
            fixed.Set(key, value);
          }
        }
        core::Json converted = core::Json::Array();
        for (const core::Json& call : calls->AsArray()) {
          core::Json entry = core::Json::Object();
          if (!NormalizeCall(call, &entry, error)) {
            return false;
          }
          converted.Push(std::move(entry));
        }
        fixed.Set("tool_calls", std::move(converted));
        items.Push(std::move(fixed));
        continue;
      }
    }
    items.Push(message);
  }
  *out = std::move(items);
  return true;
}

}  // namespace tessera::serve
