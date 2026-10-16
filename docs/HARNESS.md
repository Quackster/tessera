# HARNESS.md: serve harness goal and progress

This file is the goal contract and the progress log for the serve harness work.
The running LLM reads this file at the start of a session and updates the
"Progress log" section after each check. Keep the file current.

## Goal

Harden the tessera serve harness on the ROCm build. The harness must prove
these outcomes:

1. Connectivity. A client reaches a served tessera endpoint. OpenCode is one
   such client.
2. Tool calls. The OpenAI tool path returns `tool_calls` with the
   `finish_reason` `tool_calls`.
3. Thinking. The model thinking is returned in the `reasoning_content` field.
   The answer is returned in `content`. Thinking never leaks into `content`.
4. Parallel connections. Two connections work at the same time. Both streams
   run at the same time. The second client does not fail, does not hang
   forever, and does not corrupt the first stream.
5. Targets. Every check runs against all three targets in the next section.
6. Flash-Next defect. The defect that breaks parallel connections on the
   Flash-Next pipeline split is root-caused and fixed, with a regression test.

## Working rules

- Follow AGENTS.md. Rule 20 has force: one heavy run at a time.
- Run one server in the foreground. Free its memory before the next server.
- Run the whole parallel test in one call, not one call per client.
- Rebuild the target backend before every test run, so a stale build never
  tests old code:

```
cmake -B cmake-build-rocm -DTESSERA_BACKEND=rocm && cmake --build cmake-build-rocm -j
```

- No completion caps unless the goal states one. The server stops at the
  declared end-of-turn token. Stream the whole output and watch for garbage
  (AGENTS.md rule 19).

## Targets

The served model id comes from `GET /v1/models`. The default port is 8080.

### T1: Qwen3.8-27B GGUF (unsloth UD-Q4_K_M)

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --gpus 0 --host 127.0.0.1
```

For two real streams at once, serve two replicas, one per GPU:

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --gpus 0,1 --host 127.0.0.1
```

### T2: Qwen3.8-27B MXFP4 with the DFlash2 draft

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --gpus 0 --host 127.0.0.1
```

### T3: Qwen3.8-Flash-Next GSQ-RCO IQ3_S on the two-GPU pipeline split

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --mtp ~/models/Qwen3.8-Flash-Next-MTP \
  --speculate --pipeline --pipeline-split 24,24 --gpus 0,1 \
  --context 262144 --kv-q8 --gpu-layers 18 --host 127.0.0.1
```

## Checks

Wait for `GET /health` to return `ok` before a chat check. The server binds
before the model loads, so health answers `503` while loading.

### C1: connect

```
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/v1/models
```

Pass when health is `"ok"` and the model list holds the served model.

### C2: chat

```
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Reply with one short sentence."}],"stream":false}'
```

Pass when the reply has a non-empty `content`.

### C3: thinking in reasoning_content

Send a request with thinking on. Pass when the `message` has a non-empty
`reasoning_content` and the `content` holds the answer, and the thinking text
is not inside `content`.

### C4: tool calls

```
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the weather in Paris? Use the tool."}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "description":"Get the weather for a city",
       "parameters":{"type":"object","properties":{"city":{"type":"string"}},
       "required":["city"]}}}],
       "tool_choice":"auto","stream":false}'
```

Pass when the reply has `tool_calls` with `function.name` `get_weather` and
`finish_reason` `tool_calls`.

### C5: two connections at once, streaming at once

Start two streaming requests at the same time and confirm both finish and both
stream. The start time and the end time of each stream show the overlap.

```
out=/tmp/opencode/harness; mkdir -p "$out"
req() { curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data "$1"; }
body(){ printf '{"messages":[{"role":"user","content":"%s"}],"stream":true}' "$1"; }
req "$(body 'Count from one to twenty.')" > "$out/a.sse" 2>"$out/a.err" &
pa=$!
req "$(body 'Name five colors.')" > "$out/b.sse" 2>"$out/b.err" &
pb=$!
wait $pa; sa=$?; wait $pb; sb=$?
echo "exit a=$sa b=$sb"
grep -c '^data: ' "$out/a.sse" "$out/b.sse"
```

Pass when both exit codes are 0, both files hold SSE `data:` lines, both files
end with `data: [DONE]`, and neither stream carries an error chunk.

Record the failure text in the progress log when a check fails.

## Progress log

Add one entry per finished check. State the target, the check, the exact
command, and the observed result. Mark a defect open until its fix and its
regression test land.

- 2026-10-15 P0: created this file. Goal recorded. No check run yet.
- 2026-10-15 C1/C2 (T3, running server, before this goal): `/health` returns
  `ok`, `/v1/models` lists `Qwen3.8-Flash-Next GSQ-RCO`. Pass.
- 2026-10-15 C2/C3 (T3, running server): a non-stream chat returns
  `reasoning_content` with the thinking and `content` with the answer. Pass.
- 2026-10-15 C5 (T3, running server): two and four concurrent streaming
  requests all return 200, all end with `data: [DONE]`, and none carry an
  error. But they do not run at the same time. Two requests: single
  `count to twenty` is 21.9 s, the pair is 37.0 s. Four `name three colors`
  requests take 58.9 s. The wall time equals the sum, so the turns run one
  after the other.
- open: Flash-Next T3 does not serve two connections at once. Root cause:
  `PipelineRunner::Submit` holds the runner mutex for the whole turn
  (`src/core/runner.cpp:122`), so a second connection waits. A single-device
  model batches concurrent turns through `Scheduler`
  (`src/core/scheduler.cpp`), so it streams them at once. The split pipeline
  has no such batched driving path in serve. The overlapped multi-sequence
  path `Pipeline::GenerateMany` exists but is greedy, non-streaming, and
  without a speculative strategy, so serve cannot reuse it as is.

## Fix: concurrent pipeline turns

Status: implemented, awaiting device verification.

The split pipeline now serves concurrent turns. Each turn keeps its own
state, and the pipeline interleaves the turns between tokens.

- `Pipeline::Impl::SequenceState` holds one cache per stage, the boundary
  handoff buffers and one draft strategy, so a turn never shares KV or
  draft state with another.
- `Pipeline::Impl` holds a small pool of sequences (`kMaxConcurrentTurns`
  plus the single-turn default), a device mutex, and the strategy factory.
- `Pipeline::GenerateStreamingConcurrent` takes a free sequence, holds the
  device mutex for the turn, and yields the device every
  `kYieldTokens` tokens. It re-activates its sequence after each reacquire,
  because the other turn repointed the stages at its own sequence.
- `PipelineRunner::Submit` calls `GenerateStreamingConcurrent`, so the two
  serve turns interleave instead of running one to completion.
- The single-turn paths (`Generate`, `GenerateTimed`, `GenerateStreaming`)
  keep the default sequence and are unchanged.
- Regression test: `EngineTest.ShardedPipelineServesTwoTurnsAtOnceWhenProvided`.

Per-stage KV memory multiplies by the concurrent turn count, so a long
context bounds how many turns fit; the pool is two turns.

Files touched: `include/tessera/pipeline.hpp`, `src/core/pipeline_impl.hpp`,
`src/core/pipeline.cpp`, `src/core/pipeline_spec.cpp`, `src/core/runner.cpp`,
`tests/engine_test.cpp`.

