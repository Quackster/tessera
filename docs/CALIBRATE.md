# Tessera calibration

Calibration measures engine settings on your machine and keeps the fastest value. It changes speed only. It never changes the produced tokens.

There are two modes. Use one mode per command. One command never runs both modes.

## Modes at a glance

| Mode | Command | Result |
| --- | --- | --- |
| Single GPU speed sweep | `calibrate --model <path>` without `--gpus` | Best speed settings plus memory bound |
| Parallel memory bound | `calibrate --model <path> --gpus 0,1 --tp` or `--cp` or `--pipeline` | Memory bound only, no speed settings |

## Mode 1: single GPU speed sweep

Use this mode to tune one GPU for one model.

Example:

```sh
# MXFP4 target with the DFlash2 draft.
./cmake-build-rocm/tessera-cli calibrate \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --calibration ~/.config/tessera/calibration.json

# Unsloth GGUF Q4_K_M with MTP speculation.
./cmake-build-rocm/tessera-cli calibrate \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --speculate \
  --calibration ~/.config/tessera/calibration.json
```

Useful flags:

* `--model <path>`: model file or directory. Required.
* `--gpu <n>`: device index. Default is 0.
* `--context <n>`: context length stored in the key. Default is 4096.
* `--kv-f16`, `--kv-q8`, `--kv-q4`, `--kv-fp8`: KV cache type. Default is int8.
* `--draft <dir>`: DFlash2 draft checkpoint. It enables the draft sweeps.
* `--speculate`: MTP draft from the target head. It enables the draft block sweep.
* `--prompts <file>`: prompts for the sweep, one per line. Default is three fixed prompts.
* `--calibrate-limit <n>`: test at most this many prompts per sweep point. Default is all prompts. A small value runs faster and still keeps the best of what it measured.
* `--calibration <file>`: file to read and write. You can also set `TESSERA_CALIBRATION`. One of the two is required.
* `--quiet`: hide progress logs.

What it tunes:

| Setting | Candidates | When it is swept |
| --- | --- | --- |
| `mxfp4_split_target` | `120`, `320`, `640` | ROCm only |
| `mxfp4_split_cap` | `2`, `4`, `8` | ROCm only |
| `prefill_chunk_tokens` | `512`, `768`, `1024`, `1536`, `2048` | Always |
| `attention_split` | `4`, `8`, `16`, `32` | Only when prompts reach 1024 keys |
| `draft_tokens` | `4`, `8`, plus checkpoint value | Only with `--draft` or `--speculate` |
| `draft_context` | `512`, `1024`, `2048` | Only with `--draft` |

How it measures:

* It loads the model one time in one process.
* It uses greedy decode. Greedy output is the reference.
* Decode settings use a short prompt and 32 decode tokens.
* The prefill setting uses long prompts and measures prompt tokens per second.
* The rate of one value is the median over the prompts.
* It keeps a value only when it beats the default by more than 0.03.
* It confirms a winner with three interleaved re-measurements.
* It warms the engine before the first point and warms each new value once.

What it stores:

* One entry under key `backend | device | model | context | kv type | strategy | none`.
* Strategy is `none`, `mtp`, or `dflash2`.
* The entry holds the chosen settings, the median decode rate, the median prefill rate, the memory bound context, and the date.
* The memory bound is informational. It equals free device memory minus a 2 GiB reserve, divided by KV bytes per token.
* The report prints each point, the chosen values, and an example `run` command.

## Mode 2: parallel memory bound

Use this mode to find the largest KV context for a multi GPU layout.

Example:

```sh
./cmake-build-rocm/tessera-cli calibrate \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --gpus 0,1 --tp \
  --calibration ~/.config/tessera/calibration.json
```

Rules:

* `--gpus` needs at least two devices, for example `0,1`.
* Pass exactly one route: `--tp`, `--cp`, or `--pipeline`.
* Do not pass `--draft` or `--speculate`. The entry always stores strategy `none`.
* The entry stores one layout: `tp2`, `cp2`, or `pipe2`.

How it measures:

* It builds the route and measures free memory and weight bytes on each rank.
* It combines the ranks. Tensor takes the minimum. Context multiplies the minimum by the rank count. Pipeline scales each stage by its layer share.
* It runs one short validation generate of 8 tokens. It measures no speed.
* It stores `max_context` and the date. Decode and prefill rates stay empty.

## Use a saved entry

Pass the same file to `run` or `serve`:

```sh
# MXFP4 target with the DFlash2 draft.
./cmake-build-rocm/tessera-cli run \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --calibration ~/.config/tessera/calibration.json \
  --prompt-text "Hello" --max-completion-tokens 32

# Unsloth GGUF Q4_K_M with MTP speculation.
./cmake-build-rocm/tessera-cli run \
  --model ~/models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --speculate \
  --calibration ~/.config/tessera/calibration.json \
  --prompt-text "Hello" --max-completion-tokens 32
```

The engine looks up the key for the current backend, device, model, context, KV type, strategy, and layout. An explicit flag always wins over a saved value. There is no global lookup. You must pass `--calibration`.

Only `calibrate` writes the file. `run` and `serve` only read it. A missing file on `run` or `serve` is not an error. It means no calibration exists yet. Create the parent directory before you run `calibrate`. The tool does not create it.

## What calibration does not do

* It does not tune speed for a parallel layout. The `--gpus` path stores memory only.
* It does not benchmark concurrent sessions. There is no sweep over 2, 3, or 4 sessions with tok/s per count.
* It does not report max KV cache under concurrent load.
* It does not store speed rates for parallel entries.
* It does not sweep the draft on the parallel path.
* It does not select the KV cache type. The KV type is your choice. It changes numerics.
* It does not select the backend. The backend is fixed at configure time.
* It does not tune sampling. Sampling changes output.
* It does not store model paths. The key stores the model name only.

Note: `run` with two or more `--prompt-text` flags runs concurrent sessions on one device. That path is a functional run. It is not a calibration sweep. It writes no file.

## Save and read the calibration log

The log is the console output of `calibrate`. It is separate from the JSON file. The JSON file applies settings. The log only records what happened.

Save it with `tee`:

```sh
./cmake-build-rocm/tessera-cli calibrate \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --calibration ~/.config/tessera/calibration.json \
  2>&1 | tee calibration.log
```

Progress lines go to stderr. The report goes to stdout. Redirect both, or the file misses the progress lines. `--quiet` hides the progress lines. The report still prints.

Read it top to bottom:

* Identity rows name the backend, device, model, KV type, strategy, key context, and memory bound.
* `Sweep` lists each tested value with its tok/s. `(default)` marks the built-in value. `(kept)` marks the winner.
* `Chosen` lists the values the run keeps.
* `Wrote` names the JSON file the entry went to.
* `Example` shows two `run` commands that reuse the result. The first spells the flags. The second applies the saved entry with `--calibration`.

## TODO

1. Add a speed sweep for parallel layouts. Sweep the same speed knobs through `--tp`, `--cp`, and `--pipeline` on the target GPUs. Store the winner under the layout key.
2. Add the draft sweep to the parallel path. Cover `--draft` and `--speculate` for each layout.
3. Add a concurrent session benchmark. Run 2, then 3, then 4 sessions on the calibrated model. Report total tok/s and per session tok/s for each count.
4. Report max KV cache under concurrent load. Record the largest session count that fits the requested context.
5. Store measured decode and prefill rates in parallel entries. Fill the rate fields that are empty today.
6. Keep diagnostic flags out of the sweep. `TESSERA_MXFP4_WMMA`, `TESSERA_MXFP4_W4A8`, `TESSERA_TARGET_BF16`, tiled GEMM thresholds, and the prefill attention budget stay manual.

## File format

The file is JSON. It holds one entry per key. A missing file means no calibration yet. A malformed file is rejected. Old files without the `parallel` field still load as `none`.
