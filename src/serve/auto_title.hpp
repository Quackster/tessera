#pragma once

#include <memory>
#include <string>

#include "core/json.hpp"
#include "serve/session.hpp"
#include "tessera/runner.hpp"

// Model-generated chat titles: on the first prompt of a new session the
// model names the chat window from that prompt, replacing the
// first-user-line fallback. The name is set before the answer streams.
// Web sessions only; OpenAI sessions keep the first-line title.

namespace tessera {

class Engine;
class Model;
class Tokenizer;

namespace serve {

// Title prompt body: a direct instruction carrying the first user text,
// with thinking off so the answer is the title itself.
[[nodiscard]] core::Json TitlePromptBody(const std::string& user_text);

// Name a fresh default-titled session before its first answer: ask the
// model for a short title when enabled (call with the session turn and
// the device slot held), else fall back to the first user line. No-op
// unless the title is still the default; never touches a renamed
// session or the history. `kv_type` is the runner's KV storage type and
// `kv_offload` its KV cache location, read onto the title request like
// every other serve turn.
void MaybeAutoTitle(Engine& engine, Model& model, const Tokenizer& tokenizer,
                    const std::shared_ptr<Session>& session, bool enabled,
                    KvCacheType kv_type = KvCacheType::F32,
                    bool kv_offload = false);

// Pool form of the above: titles through the shared scheduler when
// the engine allows it, else a brief exclusive lease. Same result.
void MaybeAutoTitle(RunnerPool& pool, const std::shared_ptr<Session>& session,
                    bool enabled);

}  // namespace serve

}  // namespace tessera
