# Calibrate command playbook

This file tells an LLM how to calibrate everything in one pass.
The caller supplies the model directory. The LLM runs the full matrix.
Read this file and `docs/CALIBRATE.md` before any run.

## Inputs

The caller supplies these values at the start of the session:

- `MODELS_DIR`: directory that holds the model variants.
- `CALIB_FILE`: calibration JSON file to read and write.
- Scope is Qwen 3.8 27B only. Other models are out of scope.

Expected layout under `MODELS_DIR`:

- `Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf`: GGUF target.
- `Qwen3.8-27B-MXFP4-MTPFP8`: MXFP4 target directory.
- `Qwen3.8-27B-DFlash2-FP8`: DFlash2 draft directory.

MTP needs no path. `--speculate` drafts from the target nextn head.
DFlash2 needs `--draft <DFlash2 directory>`.

## Key facts

One `calibrate` invocation does one path. It never does both paths.
Without `--gpus` it runs the single GPU speed sweep.
With `--gpus` it runs the parallel memory bound only.
The branch lives in `tools/cli/calibrate_command.cpp`.

The parallel path ignores `--draft` and `--speculate`.
It always stores strategy `none`.
It stores one entry per layout: `tp2`, `cp2`, or `pipe2`.
The single path stores `parallel: none` with strategy `none`, `mtp`, or `dflash2`.

One backend needs one binary. Use both binaries:

- `./cmake-build-rocm/tessera-cli`
- `./cmake-build-vulkan/tessera-cli`

Defaults are context `4096` and KV type int8. Keep both defaults.
All runs append entries to the same `CALIB_FILE`.

## Machine rules

Run one heavy command at a time. Never overlap two model loads.
Check GPU memory before each run with `rocm-smi`.
If GPU 0 is busy, run single GPU work on the free GPU with `--gpu 1`.
Parallel work needs both GPUs free. If one GPU is busy, stop and ask.
Never background a run. Never loop runs in one shell command.
Cap each command at 5 minutes. A longer run signals a defect.
The 79 GB Flash Next split is excluded. It does not fit in 2x32 GB.

## Single GPU matrix

Run 8 commands. Replace `$MODELS_DIR` and `$CALIB_FILE`. Set `--gpu` to the free GPU.

```sh
./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --calibration "$CALIB_FILE" --gpu 1

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --speculate --calibration "$CALIB_FILE" --gpu 1

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --calibration "$CALIB_FILE" --gpu 1

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --draft "$MODELS_DIR/Qwen3.8-27B-DFlash2-FP8" \
  --calibration "$CALIB_FILE" --gpu 1

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --calibration "$CALIB_FILE" --gpu 1

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --speculate --calibration "$CALIB_FILE" --gpu 1

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --calibration "$CALIB_FILE" --gpu 1

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --draft "$MODELS_DIR/Qwen3.8-27B-DFlash2-FP8" \
  --calibration "$CALIB_FILE" --gpu 1
```

## Parallel matrix

Run 12 commands. Each command needs `--gpus 0,1` and exactly one route flag.
Do not pass `--speculate` or `--draft` here. The entry is strategy `none`.

```sh
./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --tp --calibration "$CALIB_FILE"

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --cp --calibration "$CALIB_FILE"

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --pipeline --calibration "$CALIB_FILE"

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --tp --calibration "$CALIB_FILE"

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --cp --calibration "$CALIB_FILE"

./cmake-build-rocm/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --pipeline --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --tp --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --cp --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf" \
  --gpus 0,1 --pipeline --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --tp --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --cp --calibration "$CALIB_FILE"

./cmake-build-vulkan/tessera-cli calibrate \
  --model "$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8" \
  --gpus 0,1 --pipeline --calibration "$CALIB_FILE"
```

## Verify

Count the entries. Expect 20 entries in total.
Expect 8 single entries with `parallel: none`.
Expect 12 parallel entries with `tp2`, `cp2`, or `pipe2`.
Expect both backends `rocm` and `vulkan` present.
Expect strategies `none`, `mtp`, and `dflash2` on the single entries.
Use this check:

```sh
python3 -c "import json; f=json.load(open('$CALIB_FILE')); print(len(f['entries'])); print(sorted(set((e['backend'], e['model'], e['strategy'], e['parallel']) for e in f['entries'])))"
```

## If a run fails

Read the error line. It states the next action.
A busy GPU means wait and retry on the free GPU.
A missing model path means check `MODELS_DIR`.
A usage error means print the usage and fix the flags.
Never add a shim or a workaround. Fix the root cause.
