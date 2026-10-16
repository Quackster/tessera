#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "core/json.hpp"
#include "serve/jinja/jinja.hpp"

// Chat prompt rendering for the serving layer. Converts request JSON to
// template values and renders the model's chat template.
namespace tessera {

class Model;

namespace serve {

// Template values a request leaves unset: thinking on unless a request
// turns it off, and an empty reasoning effort to pick the template's
// highest level. The launcher can change both (`serve --no-thinking`).
struct TemplateDefaults {
  bool enable_thinking = true;
  std::string reasoning_effort;
};

// Convert a JSON value into a template value (recursive).
[[nodiscard]] jinja::Value ToJinja(const core::Json& json);

// The thinking flag a request selects: its own boolean when present, else
// the serve default. The renderer and the thinking streamer must agree on
// this, so both read it here.
[[nodiscard]] bool RequestEnableThinking(const core::Json& body,
                                         const TemplateDefaults& defaults);

// The chat-template arguments a request supplies: every entry of
// `chat_template_kwargs` (an object), plus a top-level `reasoning_effort`
// when the nested object does not set one. Values pass through unchanged,
// so the model template validates them. A body without either yields an
// empty list.
[[nodiscard]] std::vector<jinja::Kwarg> RequestTemplateKwargs(
    const core::Json& body);

// The reasoning-effort levels a chat template offers, in template order.
// Empty when the template never reads `reasoning_effort`. The scan reads
// the `not in ('xhigh', 'medium', 'low')` validation list; a template
// that reads reasoning_effort without such a list gets the standard three
// levels.
[[nodiscard]] std::vector<std::string> ReasoningEffortsFromTemplate(
    std::string_view tmpl);

// Render the chat template for a chat/messages request body into a
// prompt string. Converts OpenAI-form tool history (arguments as JSON
// strings) into the mappings the template expects, and honors
// `tool_choice: "none"` by rendering without tools. Iterates the request
// template arguments first (see RequestTemplateKwargs), so the caller's
// `messages`, `tools` and `add_generation_prompt` stay authoritative.
// Requests that omit `enable_thinking` or `reasoning_effort` use
// `defaults`. Empty `error` on success.
std::string RenderPrompt(const Model& model, const core::Json& body,
                         std::string* error,
                         const TemplateDefaults& defaults = {});

}  // namespace serve

}  // namespace tessera
