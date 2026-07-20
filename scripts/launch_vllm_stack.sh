#!/usr/bin/env bash
# Launch one shared Qwen vLLM endpoint on GPUs 0-3.
#
# Every research profile (reasoning, control, coding, formatting, and proof) uses this
# endpoint. Per-request chat-template settings in config.yml enable or disable thinking.
#
# Defaults:
#   GPUs 0,1,2,3 -> Qwen/Qwen3.6-35B-A3B, TP=4, port 8000, served as qwen-research
#
# Example overrides:
#   QWEN_MODEL=Qwen/Qwen3.6-35B-A3B QWEN_PORT=18000 REPLACE=1 \
#     ./scripts/launch_vllm_stack.sh

set -euo pipefail

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
command -v vllm >/dev/null 2>&1 || { echo "vllm is required on PATH" >&2; exit 1; }

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi
command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "python3 or python is required to parse QWEN_EXTRA_ARGS" >&2
  exit 1
}

HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-logs/vllm}"
REPLACE="${REPLACE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
SESSION="${QWEN_TMUX_SESSION:-vllm-qwen}"

QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.6-35B-A3B}"
QWEN_SERVED_MODEL_NAME="${QWEN_SERVED_MODEL_NAME:-qwen-research}"
QWEN_PORT="${QWEN_PORT:-8000}"
QWEN_GPUS="${QWEN_GPUS:-0,1,2,3}"
QWEN_TP="${QWEN_TP:-4}"
QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-32768}"
QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.88}"
QWEN_QUANTIZATION="${QWEN_QUANTIZATION:-none}"
QWEN_KV_CACHE_DTYPE="${QWEN_KV_CACHE_DTYPE:-auto}"
QWEN_DTYPE="${QWEN_DTYPE:-auto}"
DEFAULT_QWEN_EXTRA_ARGS="--reasoning-parser qwen3 --language-model-only --default-chat-template-kwargs '{\"enable_thinking\": true, \"preserve_thinking\": false}'"
QWEN_EXTRA_ARGS="${QWEN_EXTRA_ARGS:-$DEFAULT_QWEN_EXTRA_ARGS}"

mkdir -p "$LOG_DIR"

shell_join() {
  printf '%q ' "$@"
}

parse_extra_args_into_array() {
  local extra_args="$1"
  local -n out_array="$2"
  local tmp parsed_arg

  tmp="$(mktemp)"
  if ! EXTRA_ARGS="$extra_args" "$PYTHON_BIN" - <<'PY' >"$tmp"
import os
import shlex
import sys

try:
    args = shlex.split(os.environ["EXTRA_ARGS"])
except ValueError as exc:
    print(f"failed to parse QWEN_EXTRA_ARGS: {exc}", file=sys.stderr)
    sys.exit(2)

for arg in args:
    sys.stdout.buffer.write(arg.encode("utf-8") + b"\0")
PY
  then
    rm -f "$tmp"
    return 1
  fi

  while IFS= read -r -d '' parsed_arg; do
    out_array+=("$parsed_arg")
  done <"$tmp"
  rm -f "$tmp"
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$REPLACE" == "1" ]]; then
    echo "Killing existing tmux session: $SESSION"
    tmux kill-session -t "$SESSION"
  else
    echo "tmux session already exists: $SESSION (set REPLACE=1 to restart)" >&2
    exit 1
  fi
fi

cmd=(
  env "CUDA_VISIBLE_DEVICES=$QWEN_GPUS"
  vllm serve "$QWEN_MODEL"
  --served-model-name "$QWEN_SERVED_MODEL_NAME"
  --host "$HOST"
  --port "$QWEN_PORT"
  --tensor-parallel-size "$QWEN_TP"
  --max-model-len "$QWEN_MAX_MODEL_LEN"
  --gpu-memory-utilization "$QWEN_GPU_MEMORY_UTILIZATION"
)

if [[ -n "$QWEN_QUANTIZATION" && "$QWEN_QUANTIZATION" != "none" ]]; then
  cmd+=(--quantization "$QWEN_QUANTIZATION")
fi
if [[ -n "$QWEN_KV_CACHE_DTYPE" && "$QWEN_KV_CACHE_DTYPE" != "auto" ]]; then
  cmd+=(--kv-cache-dtype "$QWEN_KV_CACHE_DTYPE")
fi
if [[ -n "$QWEN_DTYPE" && "$QWEN_DTYPE" != "auto" ]]; then
  cmd+=(--dtype "$QWEN_DTYPE")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  cmd+=(--trust-remote-code)
fi
if [[ -n "$QWEN_EXTRA_ARGS" ]]; then
  parsed_extra_args=()
  parse_extra_args_into_array "$QWEN_EXTRA_ARGS" parsed_extra_args
  cmd+=("${parsed_extra_args[@]}")
fi

logfile="$LOG_DIR/$SESSION.log"
quoted_cmd="$(shell_join "${cmd[@]}")"
printf -v tmux_cmd 'set -o pipefail; echo %q; echo %q; %s 2>&1 | tee -a %q' \
  "[$(date -Is)] Starting $SESSION" \
  "Command: $quoted_cmd" \
  "$quoted_cmd" \
  "$logfile"

echo "Launching shared Qwen endpoint on CUDA_VISIBLE_DEVICES=$QWEN_GPUS"
echo "Model: $QWEN_MODEL"
echo "Served model name: $QWEN_SERVED_MODEL_NAME"
echo "OpenAI-compatible endpoint: http://127.0.0.1:$QWEN_PORT/v1"
echo "Command: $quoted_cmd"
tmux new-session -d -s "$SESSION" "$tmux_cmd"

cat <<EOF

Launched tmux session: $SESSION
Attach with:
  tmux attach -t $SESSION

All profiles in config.example.yml point to this one endpoint.
Restart with:
  REPLACE=1 $0
EOF
