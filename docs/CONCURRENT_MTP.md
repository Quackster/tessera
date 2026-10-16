# CONCURRENT_MTP.md: concurrent MTP on the layer-split pipeline

This file is the goal and the progress log for concurrent MTP support on the
multi-GPU layer-split pipeline. Update the progress log after each step.

## Goal

Two connections on one served layer-split model each run MTP speculation at
the same time, and each returns output equal to the single-device reference.
No loop, no OOM. The full ROCm suite passes.

The goal is not passed unless MTP works with BOTH GPUs in use, as equally as
possible, and with the full 262144-token KV cache enabled and sized for both
concurrent turns at once.

Both conditions are required. First, both cards must carry the model and do
MTP work. The split must stay balanced, close to the 24/24 layer split of the
target command. One card doing the work while the other idles does not pass.
Second, the full 262144-token KV cache must be supported for both concurrent
turns at the same time. A result that only fits, or only runs, at a smaller
context does not pass. The KV cache must be maximized for the 262144 context,
not trimmed to make the run fit.

If the two-turn MTP state does not fit at `--context 262144 --gpu-layers 18`,
the memory layout must change (for example a per-card cache split or context
parallel, `--cp`, and a per-stage `--gpu-layers` split such as
`--gpu-layers 24,24`) until the full context works, or the smallest fitted
configuration is documented with the reason. A configuration that fits only
by moving nearly all work onto one card does not pass.

Partial credit is not a pass: both concurrent sessions must run MTP on both
cards, at the full context, and return the single-device reference output.

Target command:

```
./cmake-build-rocm/tessera-cli serve \
  --model ~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --mtp ~/models/Qwen3.8-Flash-Next-MTP \
  --speculate --pipeline --pipeline-split 24,24 --gpus 0,1 \
  --context 262144 --kv-q8 --gpu-layers 18 --host 0.0.0.0
```

Model: Qwen3.8-Flash-Next, GGUF `general.architecture = qwen4exp`, IQ3_S,
MTP draft head, layer-split 24/24 over two R9700s. Paged weights
(`--gpu-layers 18`) hold the rest in system RAM.

## Starting state

- Commit `21f6df9` added concurrent serving for the split pipeline. Each turn
  gets its own `Pipeline::Impl::SequenceState` (a cache per stage plus
  boundary buffers). A pipeline device mutex serializes stage work, and
  `Pipeline::GenerateStreamingConcurrent` releases it every few tokens so two
  turns interleave.
- Commit `b377e5a` settled the plain path: concurrent turns decode without
  the draft (interleaving two MTP turns corrupts the draft and verify state),
  the yield sleeps one millisecond (a bare `yield` serialized the turns), and
  an idle turn frees its caches. `serve --pipeline` does not attach the
  drafter or load the MTP head.
- Concurrent MTP is not implemented. The API and the CLI drop the draft on
  concurrent turns.

## Evidence

- Single MTP turn on the rebuilt pipeline: coherent.
- Two concurrent plain turns: coherent and overlapping.
- Two concurrent MTP turns with per-cache heads (`--context 8192
  --gpu-layers 8`): the third 1.6 GiB MTP head load fails
  (`hipMalloc failed for 1677721600 bytes`), turn 1 returns
  `{"error":"generation failed"}`, turn 2 garbles. `oom=1`, `mtp_loads=3`.
- Serialized MTP turns (`TESSERA_PIPELINE_SERIAL=1`, same flags): coherent,
  `oom=0`. The first turn frees its caches (and its head) before the second
  loads, so only one head is live at a time.
- Two concurrent MTP turns with ONE shared read-only head (this is the P3
  fix applied early to isolate memory from correctness): `oom=0`,
  `shared_loads=1`, but BOTH turns still garble
  (`...Five colors了 separatedITED了 commas!!!...`). So the shared head fixes
  the memory, and a real interleaving defect remains. The corruption is not
  caused by the OOM.
- DRAIN TEST (`TESSERA_PIPELINE_SYNC_YIELD=1`, which synchronizes every
  stage right before the yield releases the device lock): two concurrent MTP
  turns are COHERENT, and each output matches its single-turn run token for
  token (`red, blue, green, yellow, purple` / `Apple, Banana, Orange, Grape,
  Mango`), `oom=0`. So the corruption comes from device work that is still
  in flight when the lock is released between tokens. The fix direction is
  to make the yield point quiescent (and to drain before per-turn buffers are
  freed), not to touch the draft math.
- `docs/BUG.md` tracks pipeline MTP as broken even for a single turn
  (BUG-1/BUG-3). Confirm this first; a queued fix sits on an unverified path.

## Hypotheses

The corrupt path is the one MTP uses and plain decode does not.

1. Shared mutable state in the multi-row batched verify. MTP uses
   `Pipeline::Impl::ForwardRows` -> `Architecture::ForwardBatch`
   (`src/models/qwen4exp/trunk_batch.cpp`); plain decode uses the single-row
   `Step`/`Forward`. The `Qwen4ExpState` batch buffers look per-cache, but
   find what is not.
2. The device yield releases the lock while a batched verify's device work is
   still in flight, so the second turn's work interleaves on the same stages.
3. The MTP head loads per decode cache (about 1.6 GiB on the last stage), so
   it does not fit twice at `--gpu-layers 18` / 262144. A correct fix needs
   one shared, read-only head with per-turn draft state only.

## Plan

1. Re-enable the draft for the concurrent path behind a small, removable
   change and reproduce two concurrent MTP turns at a `--gpu-layers` low
   enough to fit two heads.
2. Bisect: disable the yield so the two turns fully serialize. Correct output
   means the yield is the cause; garbage means shared state in
   `ForwardRows`/`ForwardBatch`.
3. Read the batched forward and the MTP draft (`pipeline_spec.cpp`,
   `mtp.cpp`, `trunk_batch.cpp`, `state.hpp`, `src/spec/`) for state shared
   across sequences. Compare with how the single-device `Scheduler` isolates
   concurrent jobs.
4. Implement the fix: one shared read-only MTP head, all draft KV, scratch
   and verify state per turn.

## Rules

- One heavy run at a time (AGENTS.md rule 20). Bind every client with
  `curl -m`; wait only on the client PIDs, never bare `wait` while a server
  runs in the background.
- Root-cause fixes only (rule 8). Add or extend
  `EngineTest.ShardedPipelineServesTwoTurnsAtOnceWhenProvided` to cover two
  concurrent MTP turns against the single-device greedy output.
- Rebuild ROCm serially (gcc-15 crashes under parallel builds). The
  `~/.local/bin/cmake` wrapper is broken; use `/usr/bin/cmake` and
  reconfigure before a test run.

## Progress log

- 2026-10-16: created this file. Goal recorded. No step run yet.
- 2026-10-16: single-turn pipeline MTP confirmed coherent at `--context 8192
  --gpu-layers 8`.
- 2026-10-16: bisect. Serial (`TESSERA_PIPELINE_SERIAL=1`): coherent,
  `oom=0`. Interleaved with per-cache heads: OOM on the third 1.6 GiB head,
  turn 1 error, turn 2 garbage.
- 2026-10-16: shared read-only MTP head implemented
  (`Qwen4ExpArchitecture::SharedMtpWeights`, one load per model).
  Interleaved: `oom=0`, `shared_loads=1`, both turns still garble. Memory is
  not the correctness cause; the interleaving defect is still open.
- 2026-10-16: owner set the pass bar: MTP must work on both cards with the
  full 262144 context and a KV cache sized for both turns. `--cp` and a
  per-stage `--gpu-layers` split are candidate memory layouts if the full
  context does not fit.
- 2026-10-16: drain test. `TESSERA_PIPELINE_SYNC_YIELD=1` (sync every stage
  before the yield unlocks) makes two concurrent MTP turns coherent and
  identical to the single-turn runs. In-flight device work at the yield is
  the corruption source. Root fix to land: a quiescent yield point, plus a
  drain before per-turn caches are freed.
- 2026-10-16: owner raised the pass bar again: the goal also fails unless
  both GPUs are in use, as equally as possible, together with the full
  262144-token KV cache. Goal section updated.
- 2026-10-16: root fix landed. `Pipeline::Impl::SynchronizeStages()` drains
  every stage. The yield in `GenerateStreamingConcurrent` calls it before it
  releases the device lock, and `ReleaseSequenceCaches` calls it before it
  frees a turn's caches. Reproduced before the fix (rapid-fire, no pause,
  `--context 8192 --gpu-layers 8`, `max_completion_tokens=200`): 16/16 turns
  emitted a `!` loop. After the fix: 8/8 pairs coherent and deterministic,
  each matching its single-turn output.
- 2026-10-16: long-generation check after the fix. Two concurrent MTP turns,
  `max_completion_tokens=700`, two different prompts: both turns are
  token-for-token identical to their solo runs (700 tokens each, same text).
  The interleaving defect is closed at `--context 8192 --gpu-layers 8`.
- 2026-10-16: verification still open: the target command at
  `--context 262144 --gpu-layers 18` (both cards loaded and doing MTP work),
  the regression test, and the full ROCm suite.
- 2026-10-16: target command, full context, two concurrent MTP turns.
  Command: the target command with `--host 127.0.0.1`. Turn A is coherent
  (`red, blue, green, yellow, purple`). Turn B fails with
  `{"error":"generation failed"}`; the server logs
  `hipMalloc failed for 268435456 bytes (hipErrorOutOfMemory)` and
  `prefill failed: out_of_memory`. One MTP turn at this context fits (see
  `docs/BUG.md`); the second turn's per-turn state (its QSA cache) does not.
  The memory layout for two turns at 262144 is the open work.
