#pragma once

#include <string>

#include "core/json.hpp"
#include "serve/jinja/jinja.hpp"

// Chat prompt rendering for the serving layer. Converts request JSON to
// template values and renders the model's chat template.
namespace tessera {

class Model;

namespace serve {

// Convert a JSON value into a template value (recursive).
[[nodiscard]] jinja::Value ToJinja(const core::Json& json);

// Render the chat template for a chat/messages request body into a
// prompt string. Converts OpenAI-form tool history (arguments as JSON
// strings) into the mappings the template expects, and honors
// `tool_choice: "none"` by rendering without tools. Empty `error` on
// success.
std::string RenderPrompt(const Model& model, const core::Json& body,
                         std::string* error);

}  // namespace serve

}  // namespace tessera
