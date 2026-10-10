#pragma once

#include <memory>
#include <string>

#include "core/json.hpp"
#include "serve/session.hpp"

// Model-generated chat titles: after the first prompt of a new session
// the model names the chat window from the opening exchange, replacing
// the first-user-line fallback. Web sessions only; OpenAI sessions keep
// the first-line title.

namespace tessera {

class Engine;
class Model;
class Tokenizer;

namespace serve {

// Title prompt body: a direct instruction carrying the first user text,
// with thinking off so the answer is the title itself.
[[nodiscard]] core::Json TitlePromptBody(const std::string& user_text);

// Name a fresh default-titled session after its turn: ask the model for
// a short title when enabled (call with the session turn and the device
// slot held), else fall back to the first user line. No-op unless the
// title is still the default; never touches a renamed session or the
// history.
void MaybeAutoTitle(Engine& engine, Model& model, const Tokenizer& tokenizer,
                    const std::shared_ptr<Session>& session, bool enabled);

}  // namespace serve

}  // namespace tessera
