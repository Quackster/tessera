#!/usr/bin/env bash
# Container entrypoint for tessera. A container is configured entirely from
# its environment: each TESSERA_* variable below becomes a tessera-cli flag.
# TESSERA_COMMAND selects the subcommand (serve by default). An unset or
# empty variable adds no flag, so the CLI default applies. TESSERA_ARGS
# appends raw flags verbatim for anything not mapped here.
#
# A bad value exits 2 with a message that names the variable to fix.
set -euo pipefail

k_cli="${TESSERA_CLI:-/usr/local/bin/tessera-cli}"

# True for the usual truthy spellings of an on/off variable.
is_on() {
  case "${1,,}" in
    1 | true | yes | on) return 0 ;;
    *) return 1 ;;
  esac
}

args=()

# Append `--flag value` when the value is non-empty.
add() {
  if [[ -n "${2:-}" ]]; then
    args+=("$1" "$2")
  fi
}

# Append `--flag` when the value is truthy.
add_flag() {
  if is_on "${2:-}"; then
    args+=("$1")
  fi
}

# One of the --kv-* flags; the value names the storage type.
add_kv() {
  case "${1:-}" in
    "") ;;
    f16 | fp16 | half) args+=(--kv-f16) ;;
    q8 | int8) args+=(--kv-q8) ;;
    q4 | int4) args+=(--kv-q4) ;;
    fp8) args+=(--kv-fp8) ;;
    *)
      echo "entrypoint: TESSERA_KV='$1' is not one of f16, q8, q4, fp8" >&2
      exit 2
      ;;
  esac
}

# Paths.
add --model "${TESSERA_MODEL:-}"
add --draft "${TESSERA_DRAFT:-}"
add --mtp "${TESSERA_MTP:-}"
add --calibration "${TESSERA_CALIBRATION:-}"

# Devices and layout.
add --gpus "${TESSERA_GPUS:-}"
add_flag --speculate "${TESSERA_SPECULATE:-}"
add_flag --pipeline "${TESSERA_PIPELINE:-}"
add --pipeline-split "${TESSERA_PIPELINE_SPLIT:-}"
add_flag --tp "${TESSERA_TP:-}"
add_flag --cp "${TESSERA_CP:-}"
add_flag --spec-split "${TESSERA_SPEC_SPLIT:-}"
add_flag --hybrid "${TESSERA_HYBRID:-}"
add --dp "${TESSERA_DP:-}"

# Decode shape.
add --context "${TESSERA_CONTEXT:-}"
add --gpu-layers "${TESSERA_GPU_LAYERS:-}"
add --draft-block "${TESSERA_DRAFT_BLOCK:-}"
add --draft-context "${TESSERA_DRAFT_CONTEXT:-}"
add --prefill-chunk "${TESSERA_PREFILL_CHUNK:-}"
add --attention-split "${TESSERA_ATTENTION_SPLIT:-}"
add --split-target "${TESSERA_SPLIT_TARGET:-}"
add --split-cap "${TESSERA_SPLIT_CAP:-}"
add_kv "${TESSERA_KV:-}"
add_flag --kv-offload "${TESSERA_KV_OFFLOAD:-}"

# Server. The container binds every interface by default; the port is
# published with `docker run -p`.
add --host "${TESSERA_HOST:-0.0.0.0}"
add --port "${TESSERA_PORT:-8080}"
add --stats-interval "${TESSERA_STATS_INTERVAL:-}"
if [[ -n "${TESSERA_API_KEY:-}" ]]; then
  args+=(--api-key "$TESSERA_API_KEY")
fi
if [[ -n "${TESSERA_ALLOW_ORIGIN:-}" ]]; then
  IFS=', ' read -r -a origins <<<"$TESSERA_ALLOW_ORIGIN"
  for origin in "${origins[@]}"; do
    if [[ -n "$origin" ]]; then
      args+=(--allow-origin "$origin")
    fi
  done
fi
add_flag --no-auto-title "${TESSERA_NO_AUTO_TITLE:-}"
if is_on "${TESSERA_THINKING:-1}"; then
  args+=(--enable-thinking)
else
  args+=(--no-thinking)
fi
add --reasoning-effort "${TESSERA_REASONING_EFFORT:-}"

# Request defaults.
add --max-completion-tokens "${TESSERA_MAX_COMPLETION_TOKENS:-}"
add --max-thinking-tokens "${TESSERA_MAX_THINKING_TOKENS:-}"

# Logging.
add_flag --quiet "${TESSERA_QUIET:-}"
add_flag --verbose "${TESSERA_VERBOSE:-}"

# Raw escape hatch for flags with no variable above.
if [[ -n "${TESSERA_ARGS:-}" ]]; then
  read -r -a extra <<<"$TESSERA_ARGS"
  args+=("${extra[@]}")
fi

command="${TESSERA_COMMAND:-serve}"
if [[ "$command" == "serve" && -z "${TESSERA_MODEL:-}" ]]; then
  echo "entrypoint: TESSERA_MODEL is required for serve" >&2
  exit 2
fi

# Do not print the argument list: it can carry TESSERA_API_KEY.
echo "entrypoint: tessera-cli $command model=${TESSERA_MODEL:-<none>}"
exec "$k_cli" "$command" "${args[@]}"
