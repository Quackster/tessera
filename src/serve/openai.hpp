#pragma once

#include <cstddef>
#include <string>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/render.hpp"
#include "serve/session.hpp"
#include "tessera/calibrate.hpp"
#include "tessera/runner.hpp"

// OpenAI-format completions over the chat model: POST /v1/completions
// (raw prompt), POST /v1/chat/completions (messages) and POST
// /v1/messages (Anthropic shape). One instance is shared by all
// connection threads; each generation takes an idle runner from the
// pool, so parallel windows use parallel devices when available.
//
// Session management: a request carrying `session_id` continues the
// stored session with that id (404 unknown, 409 already generating).
// The incoming messages reconcile with the stored history by common
// prefix, so a client that resends the full history and a client that
// sends only new turns both converge; the assistant reply is appended.
// Without `session_id` the request is stateless. Responses keep the
// standard OpenAI shape either way.

namespace tessera {

class Tokenizer;

namespace serve {

void OpenAiComplete(RunnerPool& pool, const Tokenizer& tokenizer,
                    SessionStore& sessions, ResponseWriter& writer,
                    const core::Json& body, std::size_t default_max,
                    std::size_t stats_interval_seconds,
                    const TemplateDefaults& defaults = {});

void OpenAiChat(RunnerPool& pool, const Tokenizer& tokenizer,
                SessionStore& sessions, ResponseWriter& writer,
                const core::Json& body, std::size_t default_max, bool anthropic,
                std::size_t stats_interval_seconds,
                const TemplateDefaults& defaults = {});

}  // namespace serve

}  // namespace tessera
