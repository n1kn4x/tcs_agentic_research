#!/usr/bin/env bash
# Launch the recommended independent-reasoner/verifier vLLM stack in tmux.
#
# Default layout for 6x 32GB GPUs:
#   GPUs 0,1 -> deep-reasoner       : primary 32B-class reasoner, port 8000
#   GPUs 2,3 -> verifier-reasoner   : independent 32B-class critic/verifier, port 8002
#   GPU  4   -> routine-extractor   : 7B/8B extraction/formatting model, port 8001
#   GPU  5   -> lean-prover         : Lean/code/math specialist, port 8003
#
# Override models/ports/lengths with environment variables, for example:
#   VERIFIER_MODEL=Qwen/Qwen3-32B VERIFIER_EXTRA_ARGS='--enable-reasoning --reasoning-parser qwen3' \
#     ./scripts/launch_vllm_independent_reasoner_verifier.sh
#
# If sessions already exist, set REPLACE=1 to kill and restart them.

set -euo pipefail

command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 1; }
command -v vllm >/dev/null 2>&1 || { echo "vllm is required on PATH" >&2; exit 1; }

HOST="${HOST:-0.0.0.0}"
LOG_DIR="${LOG_DIR:-logs/vllm}"
REPLACE="${REPLACE:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"

# Model defaults are intentionally easy to override. For the 32B endpoints,
# prefer FP8/AWQ/GPTQ checkpoints when available; BF16 32B models are tight on
# 2x32GB once KV cache and vLLM overhead are included.
DEEP_MODEL="${DEEP_MODEL:-Qwen/Qwen3-32B}"
VERIFIER_MODEL="${VERIFIER_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
ROUTINE_MODEL="${ROUTINE_MODEL:-Qwen/Qwen3-8B}"
PROOF_MODEL="${PROOF_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

DEEP_PORT="${DEEP_PORT:-8000}"
ROUTINE_PORT="${ROUTINE_PORT:-8001}"
VERIFIER_PORT="${VERIFIER_PORT:-8002}"
PROOF_PORT="${PROOF_PORT:-8003}"

DEEP_GPUS="${DEEP_GPUS:-0,1}"
VERIFIER_GPUS="${VERIFIER_GPUS:-2,3}"
ROUTINE_GPUS="${ROUTINE_GPUS:-4}"
PROOF_GPUS="${PROOF_GPUS:-5}"

DEEP_TP="${DEEP_TP:-2}"
VERIFIER_TP="${VERIFIER_TP:-2}"
ROUTINE_TP="${ROUTINE_TP:-1}"
PROOF_TP="${PROOF_TP:-1}"

DEEP_MAX_MODEL_LEN="${DEEP_MAX_MODEL_LEN:-65536}"
VERIFIER_MAX_MODEL_LEN="${VERIFIER_MAX_MODEL_LEN:-65536}"
ROUTINE_MAX_MODEL_LEN="${ROUTINE_MAX_MODEL_LEN:-32768}"
PROOF_MAX_MODEL_LEN="${PROOF_MAX_MODEL_LEN:-32768}"

DEEP_GPU_MEMORY_UTILIZATION="${DEEP_GPU_MEMORY_UTILIZATION:-0.88}"
VERIFIER_GPU_MEMORY_UTILIZATION="${VERIFIER_GPU_MEMORY_UTILIZATION:-0.88}"
ROUTINE_GPU_MEMORY_UTILIZATION="${ROUTINE_GPU_MEMORY_UTILIZATION:-0.85}"
PROOF_GPU_MEMORY_UTILIZATION="${PROOF_GPU_MEMORY_UTILIZATION:-0.85}"

# Set to fp8/awq/gptq/etc. Use "none" or empty to omit --quantization.
DEEP_QUANTIZATION="${DEEP_QUANTIZATION:-fp8}"
VERIFIER_QUANTIZATION="${VERIFIER_QUANTIZATION:-fp8}"
ROUTINE_QUANTIZATION="${ROUTINE_QUANTIZATION:-none}"
PROOF_QUANTIZATION="${PROOF_QUANTIZATION:-none}"

# Set to fp8/fp8_e5m2/fp8_e4m3 to reduce KV cache memory if needed.
DEEP_KV_CACHE_DTYPE="${DEEP_KV_CACHE_DTYPE:-auto}"
VERIFIER_KV_CACHE_DTYPE="${VERIFIER_KV_CACHE_DTYPE:-auto}"
ROUTINE_KV_CACHE_DTYPE="${ROUTINE_KV_CACHE_DTYPE:-auto}"
PROOF_KV_CACHE_DTYPE="${PROOF_KV_CACHE_DTYPE:-auto}"

# Set to bfloat16/float16/etc. Use auto to omit --dtype.
DEEP_DTYPE="${DEEP_DTYPE:-auto}"
VERIFIER_DTYPE="${VERIFIER_DTYPE:-auto}"
ROUTINE_DTYPE="${ROUTINE_DTYPE:-auto}"
PROOF_DTYPE="${PROOF_DTYPE:-auto}"

# Free-form additional vLLM args. Useful for reasoning parsers, for example:
#   DEEP_EXTRA_ARGS='--enable-reasoning --reasoning-parser qwen3'
#   VERIFIER_EXTRA_ARGS='--enable-reasoning --reasoning-parser deepseek_r1'
DEEP_EXTRA_ARGS="${DEEP_EXTRA_ARGS:-}"
VERIFIER_EXTRA_ARGS="${VERIFIER_EXTRA_ARGS:-}"
ROUTINE_EXTRA_ARGS="${ROUTINE_EXTRA_ARGS:-}"
PROOF_EXTRA_ARGS="${PROOF_EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR"

shell_join() {
  printf '%q ' "$@"
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
    # shellcheck disable=SC2206
    local -a parsed_extra_args=( $extra_args )
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
  vllm-verifier \
  "$VERIFIER_GPUS" \
  "$VERIFIER_MODEL" \
  verifier-reasoner \
  "$VERIFIER_PORT" \
  "$VERIFIER_TP" \
  "$VERIFIER_MAX_MODEL_LEN" \
  "$VERIFIER_GPU_MEMORY_UTILIZATION" \
  "$VERIFIER_QUANTIZATION" \
  "$VERIFIER_KV_CACHE_DTYPE" \
  "$VERIFIER_DTYPE" \
  "$VERIFIER_EXTRA_ARGS"

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
  tmux attach -t vllm-verifier
  tmux attach -t vllm-routine
  tmux attach -t vllm-proof

Use the matching router config:
  tcs-research run --workspace workspaces/demo --config config.independent-reasoner-verifier.yml --max-iterations 3

Restart all sessions with:
  REPLACE=1 $0
EOF
