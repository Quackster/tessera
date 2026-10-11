#!/usr/bin/env bash
# Full calibration matrix for Qwen 3.8 27B (see docs/CALIBRATE_COMMAND.md).
# Runs the 8 single-GPU speed sweeps on GPU 0 and the 12 parallel memory-bound
# runs, then prints the best entry per key from the calibration file.
# Usage: MODELS_DIR=~/models CALIB_FILE=./calibration.json tools/calibrate_matrix.sh [--summary-only|--single-only|--parallel-only]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
CALIB_FILE="${CALIB_FILE:-$REPO_ROOT/calibration.json}"
GPU_SINGLE="${GPU_SINGLE:-0}"
ROCM_BIN="${ROCM_BIN:-$REPO_ROOT/cmake-build-rocm/tessera-cli}"
VULKAN_BIN="${VULKAN_BIN:-$REPO_ROOT/cmake-build-vulkan/tessera-cli}"
GGUF="$MODELS_DIR/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
MXFP4="$MODELS_DIR/Qwen3.8-27B-MXFP4-MTPFP8"
DRAFT="$MODELS_DIR/Qwen3.8-27B-DFlash2-FP8"
kBusyVramBytes=1073741824

MODE="all"
for arg in "$@"; do
  case "$arg" in
    --summary-only) MODE="summary" ;;
    --single-only) MODE="single" ;;
    --parallel-only) MODE="parallel" ;;
    *) echo "usage: $0 [--summary-only|--single-only|--parallel-only]" >&2; exit 2 ;;
  esac
done

check_bins() {
  for bin in "$ROCM_BIN" "$VULKAN_BIN"; do
    if [[ ! -x "$bin" ]]; then
      echo "missing binary: $bin (build it first)" >&2
      exit 1
    fi
  done
}

check_models() {
  for path in "$GGUF" "$MXFP4" "$DRAFT"; do
    if [[ ! -e "$path" ]]; then
      echo "missing model path: $path (check MODELS_DIR=$MODELS_DIR)" >&2
      exit 1
    fi
  done
}

# Print VRAM use per GPU; fail the parallel section when a GPU looks busy.
report_gpus() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showmeminfo vram 2>/dev/null | grep -E "GPU\[[01]\].*Used" || true
  else
    echo "rocm-smi not found, skip memory check"
  fi
}

require_gpus_free() {
  local used0 used1
  used0=$(rocm-smi --showid --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 {print $3}' | head -n 1 || echo 0)
  used1=$(rocm-smi --showid --showmeminfo vram --csv 2>/dev/null | awk -F, 'NR>1 {print $3}' | sed -n 2p || echo 0)
  if [[ "${used0:-0}" -gt "$kBusyVramBytes" || "${used1:-0}" -gt "$kBusyVramBytes" ]]; then
    echo "a GPU looks busy (used ${used0:-?} / ${used1:-?} bytes). Parallel work needs both GPUs free; stop and ask." >&2
    exit 1
  fi
}

run_single() {
  echo "--- calibrate $1 $2 (gpu $GPU_SINGLE)"
  report_gpus
  "$@"
}

run_parallel() {
  echo "--- calibrate $1 $2 $3 (gpus 0,1)"
  "$@"
}

summary() {
  CALIB_FILE="$CALIB_FILE" python3 - <<'EOF'
import json, os
path = os.environ["CALIB_FILE"]
with open(path) as f:
    data = json.load(f)
entries = data.get("entries", [])
print(f"entries: {len(entries)}")
print("keys (backend, model, strategy, parallel):")
for e in sorted(entries, key=lambda x: -x.get("decode_tps", 0)):
    print(f"  {e['backend']:6} {e['model']:22} {e['strategy']:8} {e['parallel']:6} "
          f"decode={e.get('decode_tps', 0):7.2f} prefill={e.get('prefill_tps', 0):7.2f} "
          f"split={e.get('mxfp4_split_target')}/{e.get('mxfp4_split_cap')} "
          f"chunk={e.get('prefill_chunk_tokens')} attn={e.get('attention_split')} "
          f"draft={e.get('draft_tokens')}/{e.get('draft_context')}")
singles = [e for e in entries if e.get("parallel") == "none"]
parallels = [e for e in entries if e.get("parallel") != "none"]
backends = sorted(set(e["backend"] for e in entries))
strategies = sorted(set(e["strategy"] for e in singles)) if singles else []
print(f"singles: {len(singles)} (expect 8), parallels: {len(parallels)} (expect 12)")
print(f"backends: {backends} (expect ['rocm', 'vulkan']), single strategies: {strategies} (expect ['dflash2', 'mtp', 'none'])")
best = max(entries, key=lambda x: x.get("decode_tps", 0)) if entries else None
if best:
    print(f"best decode: {best['backend']} {best['model']} {best['strategy']} {best['parallel']} at {best['decode_tps']:.2f} tok/s")
EOF
}

if [[ "$MODE" != "parallel" ]]; then
  check_bins
  check_models
fi

if [[ "$MODE" == "single" || "$MODE" == "all" ]]; then
  run_single rocm gguf-none "$ROCM_BIN" calibrate --model "$GGUF" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single rocm gguf-mtp "$ROCM_BIN" calibrate --model "$GGUF" --speculate --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single rocm mxfp4-none "$ROCM_BIN" calibrate --model "$MXFP4" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single rocm mxfp4-dflash2 "$ROCM_BIN" calibrate --model "$MXFP4" --draft "$DRAFT" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single vulkan gguf-none "$VULKAN_BIN" calibrate --model "$GGUF" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single vulkan gguf-mtp "$VULKAN_BIN" calibrate --model "$GGUF" --speculate --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single vulkan mxfp4-none "$VULKAN_BIN" calibrate --model "$MXFP4" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
  run_single vulkan mxfp4-dflash2 "$VULKAN_BIN" calibrate --model "$MXFP4" --draft "$DRAFT" --calibration "$CALIB_FILE" --gpu "$GPU_SINGLE"
fi

if [[ "$MODE" == "parallel" || "$MODE" == "all" ]]; then
  check_bins
  check_models
  require_gpus_free
  run_parallel rocm gguf tp "$ROCM_BIN" calibrate --model "$GGUF" --gpus 0,1 --tp --calibration "$CALIB_FILE"
  run_parallel rocm gguf cp "$ROCM_BIN" calibrate --model "$GGUF" --gpus 0,1 --cp --calibration "$CALIB_FILE"
  run_parallel rocm gguf pipeline "$ROCM_BIN" calibrate --model "$GGUF" --gpus 0,1 --pipeline --calibration "$CALIB_FILE"
  run_parallel rocm mxfp4 tp "$ROCM_BIN" calibrate --model "$MXFP4" --gpus 0,1 --tp --calibration "$CALIB_FILE"
  run_parallel rocm mxfp4 cp "$ROCM_BIN" calibrate --model "$MXFP4" --gpus 0,1 --cp --calibration "$CALIB_FILE"
  run_parallel rocm mxfp4 pipeline "$ROCM_BIN" calibrate --model "$MXFP4" --gpus 0,1 --pipeline --calibration "$CALIB_FILE"
  run_parallel vulkan gguf tp "$VULKAN_BIN" calibrate --model "$GGUF" --gpus 0,1 --tp --calibration "$CALIB_FILE"
  run_parallel vulkan gguf cp "$VULKAN_BIN" calibrate --model "$GGUF" --gpus 0,1 --cp --calibration "$CALIB_FILE"
  run_parallel vulkan gguf pipeline "$VULKAN_BIN" calibrate --model "$GGUF" --gpus 0,1 --pipeline --calibration "$CALIB_FILE"
  run_parallel vulkan mxfp4 tp "$VULKAN_BIN" calibrate --model "$MXFP4" --gpus 0,1 --tp --calibration "$CALIB_FILE"
  run_parallel vulkan mxfp4 cp "$VULKAN_BIN" calibrate --model "$MXFP4" --gpus 0,1 --cp --calibration "$CALIB_FILE"
  run_parallel vulkan mxfp4 pipeline "$VULKAN_BIN" calibrate --model "$MXFP4" --gpus 0,1 --pipeline --calibration "$CALIB_FILE"
fi

echo "=== best runs per key ==="
summary
