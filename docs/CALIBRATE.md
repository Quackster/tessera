# Tessera calibration

This file describes the calibration feature for tessera. Read it before you
start the implementation.

Calibration measures a few engine settings on the machine that runs the model
and keeps the fastest value. It only changes speed. It never changes the
produced tokens.

## Why calibrate

Some engine settings depend on the machine more than on the model. The shipped
defaults were measured on one machine. A different GPU, a different backend, or
a different context length can move the best value. The goal is to find the
value for the machine at hand without a code change.

A calibrated run must equal the default run. The non-speculative baseline
stays stable within the per-backend tolerance (`AGENTS.md`, architecture
rules). The output is the reference, the setting is the speed.

## Reference behavior in Strata

Strata is a hybrid CPU and GPU engine. Its tool `../Strata/tools/calibrate.py`
measures three settings that depend on the PC more than on the model:

- `--pcie-frac`: the share of the experts missing from VRAM that move over
  PCIe and run on the GPU instead of the CPU.
- `--spec-min-p`: how sure the draft layer must be to extend a verify window
  by one more guess.
- `--pool-workers`: the CPU threads that compute the experts.

The method in Strata is worth copying:

- It measures decode speed only. The prompt path streams every expert, so the
  tuned settings do not change it.
- It sweeps a fixed candidate list per setting.
- It keeps a setting only when the best value beats the default by more than
  `MIN_GAIN` (0.03). The adaptive expert tier and the operating system make a
  single measurement noisy by a few percent.
- It confirms the winning value against the default with an interleaved
  re-measurement, three times each.
- The first two settings are applied per request through a `strata_tune` key in
  the sampling map, so the sweep avoids a restart. The worker count needs a
  restart per value.
- It saves the result per machine and model in a settings file, and setup
  applies the saved result on a later install.

See `../Strata/setup.py` (`hardware_key`, `calibrate_config`,
`saved_calibration`, `setup_calibration`) for the persistence, and
`../Strata/tools/test_calibrate.py` for the stand-in engine test pattern.

## What tessera exposes today

Tessera is a GPU-only engine. There is no expert offload and no CPU expert
pool, so the Strata settings have no direct analog. The same idea maps to the
launch-shape and decode knobs below.

Hardware-dependent knobs in the tree today:

- The backend (Vulkan or ROCm). It is chosen at configure time with
  `TESSERA_BACKEND` and cannot change at run time.
- The GPU index (`EngineOptions::device_index`).
- `EngineOptions::prefill_chunk_tokens`, default auto `512`
  (`kDefaultPrefillChunkTokens`).
- `GenerateOptions::draft_tokens`, default from the draft checkpoint.
- `GenerateOptions::kv_type`, the KV cache type.
- The MXFP4 fp8 tensor-core GEMM split-K target and cap. Today these are
  environment-only: `TESSERA_MXFP4_SPLIT` (target `320`) and
  `TESSERA_MXFP4_SPLITCAP` (cap `4`), read in
  `src/models/qwen3_5/project.cpp`.
- Split-N attention chunks (`kSplitChunks = 16`, `kSplitMinKeys = 1024`),
  fixed.
- Diagnostic toggles: `TESSERA_MXFP4_WMMA`, `TESSERA_MXFP4_W4A8`,
  `TESSERA_TARGET_BF16`, `TESSERA_DFLASH2_CTX`.

## Settings to tune

| Setting | What it controls | Default | Candidates | Metric |
| --- | --- | --- | --- | --- |
| `mxfp4_split_target` | Split-K workgroups for the fp8 tensor-core MXFP4 GEMM. A small GPU is latency-bound with one split, a large GPU wants more. | `320` | `120`, `320`, `640` | decode tok/s |
| `prefill_chunk_tokens` | Tokens per prefill forward. A larger chunk cuts launch overhead and raises scratch. A smaller chunk bounds memory. | auto `512` | `256`, `512`, `1024` | prefill tok/s |
| `draft_tokens` | Draft block for speculative decode. A larger block trades acceptance against verify cost. | checkpoint value (DFlash2 block `8`) | `4`, `8`, and the checkpoint value | decode tok/s |

The first and third settings target decode speed, as in Strata. The second
targets prefill speed.

Backend scope: the split target acts only on ROCm, because the fp8 WMMA MXFP4
path is ROCm-only. On Vulkan the kernel does not exist, so the setting is
inert. Calibration must detect this and report the setting as not applicable,
not as a failure. Vulkan still calibrates the prefill chunk and the draft
block.

## Measurement protocol

- Measure decode speed as `produced_tokens / decode_ms`. Measure prefill speed
  as `prompt_tokens / prefill_ms`.
- Use greedy decode. Greedy is the deterministic reference baseline. A fixed
  candidate value must not change the tokens, only the time.
- Use three fixed prompts. The rate of one setting is the median of the rates
  over the prompts. A fixed prompt set is a file-local constant.
- Keep a setting only when the best candidate beats the default by more than
  `MIN_GAIN`. `MIN_GAIN` is a file-local constant, not a literal at the call
  site (`AGENTS.md` rule 13).
- Confirm the winner against the default with an interleaved re-measurement,
  three rounds each. Use the median of the rounds.
- Warm the engine before the first measurement, and warm each new configuration
  once before its sweep point.
- Run the whole sweep in one process with one model load. Do not load the model
  per candidate (`AGENTS.md` rule 20). Each candidate is a per-request value,
  so no restart is needed.
- Cap the run at five minutes. A longer run means the sweep is too heavy or a
  kernel regressed. Stop and fix the cause.

The default path is the floor. A candidate is kept only when it is faster and
its output matches the default within the per-backend tolerance. Do not trade
correctness for speed.

## Persistence and the hardware key

Store the result in a JSON calibration file. The path is runtime configuration,
never a hard-coded path (`AGENTS.md`, Model Data). Pass it with
`--calibration <path>`, or through a documented environment variable. The tool
reads and writes the same file.

Key an entry by the hardware and model identity, the analog of Strata's
`hardware_key`:

```
backend | device name | model name | context | kv type | strategy
```

The split target depends on the backend and the GPU. The draft block depends on
the model and the strategy. The context and the KV type change the cache and
the shapes. A change to any part of the key means the entry does not apply.

The file holds one entry per key, each with the chosen settings, the measured
tok/s, and the date. A later run with the same key reuses the entry only when
the caller passes `--calibration`. There is no implicit global lookup, so state
flows through the command line (`AGENTS.md`, engineering principles).

## Engine and CLI changes

### Public API

- Add the split-K target and cap to `GenerateOptions`, next to the existing
  `draft_tokens` and `prefill_chunk_tokens`. `0` means the built-in default.
  This is the per-request tuning channel, the analog of Strata's `strata_tune`.
- Move the two environment reads in `src/models/qwen3_5/project.cpp` to the
  state. The decode entry copies the request tuning into the architecture
  state. The architecture module reads the state. The core stays backend and
  model agnostic. Keep the environment variables as a diagnostic override and
  document that they win over the request value.
- `draft_tokens` and `prefill_chunk_tokens` already exist, so only the split
  fields are new.

### CLI

Add a `calibrate` subcommand to `tools/cli/main.cpp`:

```
tessera-cli calibrate --model <path> [--draft <dir>]
    [--context <n>] [--kv-*] [--calibration <file>]
    [--gpu <n>] [--prompts <file>]
```

The command loads the model once, warms it, sweeps the candidates, prints one
line per sweep point and a final report, and writes the calibration file. On
ROCm it sweeps the split target. On both backends it sweeps the prefill chunk
and, when a draft is attached, the draft block.

`run` and `serve` accept `--calibration <file>`. When the file has an entry for
the current key, the engine uses the saved values unless an explicit flag
overrides them. An explicit flag always wins.

All engine logic stays in the library. The CLI is a thin front end
(`AGENTS.md`, Where Things Live).

## Where the code lives

- The sweep, the candidate lists, the pick rule, the hardware key, and the
  calibration file reader and writer live in a new pair of files under
  `src/core/` or a small `src/core/calibrate/` folder, behind one module. Keep
  every file under 600 lines.
- The split-K value reaches the kernel through the Qwen3.5 architecture module,
  because the projection is architecture-owned. The core only carries the
  value.
- One canonical implementation of the pick rule, the candidate list, and the
  key. Do not copy them into the CLI. No trivial pass-through methods
  (`AGENTS.md` rules 2 and 3).
- Record the borrowed Strata approach in `CREDITS.md` in the same change
  (`AGENTS.md` rule 17).

## Tests

Add the tests to the single test target (`tests/`), not a new executable.

- A stand-in engine whose decode rate is a function of the settings, as in
  `../Strata/tools/test_calibrate.py`. Cover:
  - defaults kept when every candidate is flat,
  - a gain below `MIN_GAIN` is noise and keeps the default,
  - a real gain in the split target is found,
  - a real gain in the draft block is found,
  - the interleaved confirmation rejects a single noisy win.
- Candidate generation: the candidate list, the dedupe, and the bounds.
- The pick rule: empty input, a missing default, a tie, and the median.
- The hardware key: each field changes the key, and an identical machine and
  model matches.
- The calibration file: write, read, round trip, a malformed file, a missing
  file, and an unknown key.
- A failed run leaves the defaults in place.
- One device test that confirms a calibrated value equals the default within
  the per-backend tolerance. Skip cleanly with no device.

## Non-goals

- Do not calibrate the KV cache type. It changes numerics and is a user choice.
- Do not calibrate the backend. It is fixed at configure time.
- Do not calibrate sampling parameters. They change the output.
- Do not store model paths in the calibration file.
- Do not change the non-speculative baseline.

## Rollout checklist

1. Implement the settings, the module, the CLI command, and the tests.
2. Both backends build. The full suite passes on both.
3. State the backend used for the measurement in the report.
4. Update `docs/PROGRESS.md` and the `README.md` features table.
5. Move the item here from plan to done with the measured result.
6. Commit the milestone.
