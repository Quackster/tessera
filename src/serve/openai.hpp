#pragma once

#include <cstddef>
#include <mutex>
#include <string>

#include "core/json.hpp"
#include "serve/http.hpp"
#include "serve/session.hpp"
#include "tessera/calibrate.hpp"

// OpenAI-format completions over the chat model: POST /v1/completions
// (raw prompt), POST /v1/chat/completions (messages) and POST
// /v1/messages (Anthropic shape). One instance is shared by all
// connection threads; generations queue in arrival order on the single
// device, so parallel windows wait instead of failing.
//
// Session management: a request carrying `session_id` continues the
// stored session with that id (404 unknown, 409 already generating).
// The incoming messages reconcile with the stored history by common
// prefix, so a client that resends the full history and a client that
// sends only new turns both converge; the assistant reply is appended.
// Without `session_id` the request is stateless. Responses keep the
// standard OpenAI shape either way.

namespace tessera {

class Engine;
class Model;
class Tokenizer;

namespace serve {

void OpenAiComplete(Engine& engine, Model& model, const Tokenizer& tokenizer,
                    SessionStore& sessions, std::mutex& generation,
                    ResponseWriter& writer, const core::Json& body,
                    std::size_t default_max,
                    const CalibrationConfig& calibration = {});

void OpenAiChat(Engine& engine, Model& model, const Tokenizer& tokenizer,
                SessionStore& sessions, std::mutex& generation,
                ResponseWriter& writer, const core::Json& body,
                std::size_t default_max, bool anthropic,
                const CalibrationConfig& calibration = {});

}  // namespace serve

}  // namespace tessera
