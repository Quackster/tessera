# BUGS.md: open defects

This file tracks open defects. Add one entry per defect. Keep the
reproduction exact. Mark an entry fixed only after an end to end run
passes.

## BUG-4: a served turn runs to the context limit

Status: open.

A served chat turn does not stop. A simple request keeps generating until
the remaining context is full. On one R9700 the process has produced more
than 130000 tokens for one prompt and is still running.

### Observed

The live process:

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --gpus 0 --draft-block 4 --speculate --context 262144 --kv-q8 \
  --host 0.0.0.0 --port 8080
```

Facts gathered read only:

- The process is pid 88140. It started at 16:42.
- `GET /slots` shows two turns busy, both with one message only. The
  assistant turn is not persisted yet. `total_slots` is 1, so the second
  turn waits behind the first.
- The active socket has sent about 21 MB over about 272000 segments and
  is still sending. The rate is about 9 tokens per second. About 135000
  deltas left the server, and the turn continues.
- `POST /v1/messages/count_tokens` for the pelican prompt returns 66
  input tokens. So the large count is all output.
- The auto title turn stops. It sets thinking off and caps at 32 tokens.
  So the model stops when thinking is bounded.

### Root cause

Three defects combine. Fix all three.

1. The turn has no cap. `kDefaultMaxCompletionTokens` is 0
   (`include/tessera/defaults.hpp:56`), and the web UI sends 0 for both
   budgets (`src/serve/web/index.html:35`). `EffectiveMaxTokens` then
   returns `context - prompt` (`src/core/model.cpp:1756`). The only
   terminators left are the model stop token and the context end.

2. The model stays in its think block, and nothing closes it. The
   rendered generation prompt ends with `<think>` (`chat_template.jinja`
   line 168), so `ThinkExpected` is true
   (`src/serve/tools/tool_call.cpp:220`). Thinking is on by default with
   `reasoning_effort=xhigh`, which adds the "think carefully" system
   instruction. `ThinkBudget` is only armed when `max_thinking_tokens` is
   above 0 (`src/core/spec_driver.cpp:185`), and every default is 0. So
   the think block is unlimited. The generation loop then stops only on a
   declared stop token (`src/core/spec_driver.cpp:259`) or on non-finite
   logits. The model never emits `<|im_end|>` (id 248046), so the loop
   runs to the context end.

3. The running binary is stale. It has no `ThinkStreamer::Fold` symbol,
   so it runs the old `Push` that rescans and reallocates the whole
   accumulated text every token (`docs/PERFORMANCE.md`, item 1). The
   process started at 16:42. The fixed `tool_call.cpp` is dated 16:54 and
   the rebuilt binary 16:56. The fix is correct (the equivalence harness
   reports `mismatches=0`), but the serving process does not have it.
   That is why the long turn still crawls.

### Secondary path

The sampled MTP speculative path can also fail to stop. It can degenerate
to token 0, which is `!`, and never emit a stop token. A single turn run
of the same model with `--speculate` and sampling produced
`!!!!!!!!!!!!!!!` (see the run log for "List three colors."). If the live
output is a wall of `!` and not reasoning, this is the trigger in place of
item 2. The server terminal or the web UI shows which one it is.

### Fix

1. Restart serve onto the rebuilt binary so the incremental
   `ThinkStreamer` is active.
2. Give the think block a real end. Arm the think budget by default, or
   make the engine force the close when the model does not. Match the
   reference runtime before choosing the value.
3. Add the missing loop guard. Stop a turn that repeats the same token or
   the same span, as rule 19 of `AGENTS.md` requires. Non-finite logits
   are already caught. Finite garbage is not.
