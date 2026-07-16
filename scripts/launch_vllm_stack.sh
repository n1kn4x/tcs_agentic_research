#!/usr/bin/env bash
# Launch the vLLM stack in tmux.
#
# Default layout for 6x 32GB GPUs:
#   GPUs 0,1,2,3 -> deep-reasoner     : Qwen3.6-35B-A3B, port 8000, thinking + tools
#   GPU  4       -> routine-extractor : 7B/8B extraction/formatting model, port 8001
#   GPU  5       -> lean-prover       : Lean/code/math specialist, port 8003
#
# Qwen3.6 requires recent vLLM support (vllm>=0.19.0 is recommended by Qwen).
# Override models/ports/lengths with environment variables, for example:
#   DEEP_MODEL=Qwen/Qwen3.6-35B-A3B DEEP_MAX_MODEL_LEN=131072 \
#     ./scripts/launch_vllm_stack.sh
#
# If sessions already exist, set REPLACE=1 to kill and restart them.

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
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "python3 or python is required to parse *_EXTRA_ARGS" >&2; exit 1; }

HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-logs/vllm}"
REPLACE="${REPLACE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# Model defaults are intentionally easy to override. The deep endpoint uses
# Qwen3.6-35B-A3B (35B total / 3B activated) in text-only mode because this
# research stack does not use image/video inputs, freeing memory for KV cache.
DEEP_MODEL="${DEEP_MODEL:-Qwen/Qwen3.6-35B-A3B}"
ROUTINE_MODEL="${ROUTINE_MODEL:-Qwen/Qwen3-8B}"
PROOF_MODEL="${PROOF_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

DEEP_PORT="${DEEP_PORT:-8000}"
ROUTINE_PORT="${ROUTINE_PORT:-8001}"
PROOF_PORT="${PROOF_PORT:-8003}"

DEEP_GPUS="${DEEP_GPUS:-0,1,2,3}"
ROUTINE_GPUS="${ROUTINE_GPUS:-4}"
PROOF_GPUS="${PROOF_GPUS:-5}"

DEEP_TP="${DEEP_TP:-4}"
ROUTINE_TP="${ROUTINE_TP:-1}"
PROOF_TP="${PROOF_TP:-1}"

DEEP_MAX_MODEL_LEN="${DEEP_MAX_MODEL_LEN:-262144}"
ROUTINE_MAX_MODEL_LEN="${ROUTINE_MAX_MODEL_LEN:-32768}"
PROOF_MAX_MODEL_LEN="${PROOF_MAX_MODEL_LEN:-32768}"

DEEP_GPU_MEMORY_UTILIZATION="${DEEP_GPU_MEMORY_UTILIZATION:-0.88}"
ROUTINE_GPU_MEMORY_UTILIZATION="${ROUTINE_GPU_MEMORY_UTILIZATION:-0.85}"
PROOF_GPU_MEMORY_UTILIZATION="${PROOF_GPU_MEMORY_UTILIZATION:-0.85}"

# Set to fp8/awq/gptq/etc. Use "none" or empty to omit --quantization.
DEEP_QUANTIZATION="${DEEP_QUANTIZATION:-none}"
ROUTINE_QUANTIZATION="${ROUTINE_QUANTIZATION:-none}"
PROOF_QUANTIZATION="${PROOF_QUANTIZATION:-none}"

# Set to fp8/fp8_e5m2/fp8_e4m3 to reduce KV cache memory if needed.
DEEP_KV_CACHE_DTYPE="${DEEP_KV_CACHE_DTYPE:-auto}"
ROUTINE_KV_CACHE_DTYPE="${ROUTINE_KV_CACHE_DTYPE:-auto}"
PROOF_KV_CACHE_DTYPE="${PROOF_KV_CACHE_DTYPE:-auto}"

# Set to bfloat16/float16/etc. Use auto to omit --dtype.
DEEP_DTYPE="${DEEP_DTYPE:-auto}"
ROUTINE_DTYPE="${ROUTINE_DTYPE:-auto}"
PROOF_DTYPE="${PROOF_DTYPE:-auto}"

# Free-form additional vLLM args. Shell-style quoting is respected, so JSON
# values containing spaces can be written as single-quoted arguments. Examples:
#   DEEP_EXTRA_ARGS='--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder'
#   DEEP_EXTRA_ARGS='--speculative-config '\''{"method":"qwen3_next_mtp","num_speculative_tokens":2}'\'''
#   DEEP_EXTRA_ARGS='--hf-overrides '\''{"text_config": {"rope_parameters": {"rope_type": "yarn"}}}'\'''
DEFAULT_DEEP_EXTRA_ARGS="--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --language-model-only --default-chat-template-kwargs '{\"enable_thinking\": true, \"preserve_thinking\": true}'"
DEFAULT_ROUTINE_EXTRA_ARGS="--default-chat-template-kwargs '{\"enable_thinking\": false}'"
DEEP_EXTRA_ARGS="${DEEP_EXTRA_ARGS:-$DEFAULT_DEEP_EXTRA_ARGS}"
ROUTINE_EXTRA_ARGS="${ROUTINE_EXTRA_ARGS:-$DEFAULT_ROUTINE_EXTRA_ARGS}"
PROOF_EXTRA_ARGS="${PROOF_EXTRA_ARGS:-}"

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
    print(f"failed to parse *_EXTRA_ARGS: {exc}", file=sys.stderr)
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

launch_profile() {
  local session="$1"
  local cuda_devices="$2"
  local model="$3"
  local served_name="$4"
  local port="$5"
  local tp="$6"
  local max_model_len="$7"
  local gpu_memory_utilization="$8"
  local quantization="$9"
  local kv_cache_dtype="${10}"
  local dtype="${11}"
  local extra_args="${12}"

  if tmux has-session -t "$session" 2>/dev/null; then
    if [[ "$REPLACE" == "1" ]]; then
      echo "Killing existing tmux session: $session"
      tmux kill-session -t "$session"
    else
      echo "tmux session already exists: $session (set REPLACE=1 to restart)" >&2
      exit 1
    fi
  fi

  local -a cmd=(
    env "CUDA_VISIBLE_DEVICES=$cuda_devices"
    vllm serve "$model"
    --served-model-name "$served_name"
    --host "$HOST"
    --port "$port"
    --tensor-parallel-size "$tp"
    --max-model-len "$max_model_len"
    --gpu-memory-utilization "$gpu_memory_utilization"
  )

  if [[ -n "$quantization" && "$quantization" != "none" ]]; then
    cmd+=(--quantization "$quantization")
  fi
  if [[ -n "$kv_cache_dtype" && "$kv_cache_dtype" != "auto" ]]; then
    cmd+=(--kv-cache-dtype "$kv_cache_dtype")
  fi
  if [[ -n "$dtype" && "$dtype" != "auto" ]]; then
    cmd+=(--dtype "$dtype")
  fi
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
  fi
  if [[ -n "$extra_args" ]]; then
    local -a parsed_extra_args=()
    if ! parse_extra_args_into_array "$extra_args" parsed_extra_args; then
      echo "Failed to parse extra args for $session" >&2
      exit 1
    fi
    cmd+=("${parsed_extra_args[@]}")
  fi

  local logfile="$LOG_DIR/$session.log"
  local quoted_cmd tmux_cmd
  quoted_cmd="$(shell_join "${cmd[@]}")"
  printf -v tmux_cmd 'set -o pipefail; echo %q; echo %q; %s 2>&1 | tee -a %q' \
    "[$(date -Is)] Starting $session" \
    "Command: $quoted_cmd" \
    "$quoted_cmd" \
    "$logfile"

  echo "Launching $session on CUDA_VISIBLE_DEVICES=$cuda_devices, port $port, model $model"
  echo "Command in tmux session $session:"
  echo "  $quoted_cmd"
  tmux new-session -d -s "$session" "$tmux_cmd"
}

launch_profile \
  vllm-deep \
  "$DEEP_GPUS" \
  "$DEEP_MODEL" \
  deep-reasoner \
  "$DEEP_PORT" \
  "$DEEP_TP" \
  "$DEEP_MAX_MODEL_LEN" \
  "$DEEP_GPU_MEMORY_UTILIZATION" \
  "$DEEP_QUANTIZATION" \
  "$DEEP_KV_CACHE_DTYPE" \
  "$DEEP_DTYPE" \
  "$DEEP_EXTRA_ARGS"

launch_profile \
  vllm-routine \
  "$ROUTINE_GPUS" \
  "$ROUTINE_MODEL" \
  routine-extractor \
  "$ROUTINE_PORT" \
  "$ROUTINE_TP" \
  "$ROUTINE_MAX_MODEL_LEN" \
  "$ROUTINE_GPU_MEMORY_UTILIZATION" \
  "$ROUTINE_QUANTIZATION" \
  "$ROUTINE_KV_CACHE_DTYPE" \
  "$ROUTINE_DTYPE" \
  "$ROUTINE_EXTRA_ARGS"

launch_profile \
  vllm-proof \
  "$PROOF_GPUS" \
  "$PROOF_MODEL" \
  lean-prover \
  "$PROOF_PORT" \
  "$PROOF_TP" \
  "$PROOF_MAX_MODEL_LEN" \
  "$PROOF_GPU_MEMORY_UTILIZATION" \
  "$PROOF_QUANTIZATION" \
  "$PROOF_KV_CACHE_DTYPE" \
  "$PROOF_DTYPE" \
  "$PROOF_EXTRA_ARGS"

echo
echo "Launched vLLM tmux sessions:"
tmux ls | grep '^vllm-' || true
cat <<EOF

Attach to a session:
  tmux attach -t vllm-deep
  tmux attach -t vllm-routine
  tmux attach -t vllm-proof

Use the matching router config:
  tcs-research run --workspace workspaces/demo --config config.independent-reasoner-verifier.yml --max-iterations 3

Restart all sessions with:
  REPLACE=1 $0
EOF
