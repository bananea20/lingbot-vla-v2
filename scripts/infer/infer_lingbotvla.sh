#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export CONDA_ENV="${CONDA_ENV:-lingbotvla}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/mnt/nfs_share/temp-data/frank_data/ckpt/lingbot-vla/s1_fridge_head_qrot_step60000_inference/global_step_60000}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$CHECKPOINT_ROOT/hf_ckpt}"
ROBO_NAME="${ROBO_NAME:-s1_fridge_head_qrot}"
TRAINING_CONFIG="${TRAINING_CONFIG:-$CHECKPOINT_ROOT/configs/s1_fridge_head_qrot.yaml}"
ROBOT_CONFIG="${ROBOT_CONFIG:-$CHECKPOINT_ROOT/configs/robot_config.yaml}"
NORM_PATH="${NORM_PATH:-$CHECKPOINT_ROOT/configs/s1_fridge_head_qrot.json}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
USE_LENGTH="${USE_LENGTH:-25}"
USE_COMPILE="${USE_COMPILE:-0}"
PRESERVE_HEAD_FROM_STATE="${PRESERVE_HEAD_FROM_STATE:-0}"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "conda.sh not found under ~/miniconda3 or ~/anaconda3" >&2
    exit 1
fi
conda activate "$CONDA_ENV"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory not found: $CHECKPOINT_DIR" >&2
    exit 1
fi

for file in "$TRAINING_CONFIG" "$ROBOT_CONFIG" "$NORM_PATH"; do
    if [[ ! -f "$file" ]]; then
        echo "Required inference config not found: $file" >&2
        exit 1
    fi
done

RUNTIME_ROOT="$(mktemp -d /tmp/lingbotvla_v2_s1_fridge_qrot.XXXXXX)"
trap 'rm -rf "$RUNTIME_ROOT"' EXIT

# The policy loader expects:
#   <run>/lingbotvla_cli.yaml
#   <run>/checkpoints/global_step_60000/hf_ckpt
# The exported inference bundle stores configs next to hf_ckpt, so build that
# expected layout in /tmp without touching the checkpoint bundle.
RUNTIME_RUN="$RUNTIME_ROOT/run"
mkdir -p "$RUNTIME_RUN/checkpoints/global_step_60000"
ln -s "$CHECKPOINT_DIR" "$RUNTIME_RUN/checkpoints/global_step_60000/hf_ckpt"
MODEL_PATH="$RUNTIME_RUN/checkpoints/global_step_60000/hf_ckpt"

python - "$TRAINING_CONFIG" "$RUNTIME_RUN/lingbotvla_cli.yaml" <<'PY'
import sys
import yaml

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r") as f:
    config = yaml.safe_load(f)

data = config.get("data", {})
for key in ("joints", "norm_type"):
    values = data.get(key)
    if isinstance(values, list):
        data[key] = [str(v) if isinstance(v, dict) else v for v in values]

with open(dst, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)
PY

mkdir -p "$RUNTIME_ROOT/configs/robot_configs" "$RUNTIME_ROOT/assets/norm_stats"
ln -s "$ROBOT_CONFIG" "$RUNTIME_ROOT/configs/robot_configs/$ROBO_NAME.yaml"
ln -s "$NORM_PATH" "$RUNTIME_ROOT/assets/norm_stats/$ROBO_NAME.json"

PRETRAINED_ROOT="${PRETRAINED_ROOT:-}"
if [[ -z "$PRETRAINED_ROOT" && -d "$REPO_ROOT/pretrained" ]]; then
    PRETRAINED_ROOT="$REPO_ROOT/pretrained"
fi

if [[ -n "$PRETRAINED_ROOT" ]]; then
    ln -s "$PRETRAINED_ROOT" "$RUNTIME_ROOT/pretrained"
fi

QWEN3VL_PATH="${QWEN3VL_PATH:-}"
if [[ -z "$QWEN3VL_PATH" && -f "$RUNTIME_ROOT/pretrained/lingbot_models/pretrained/Qwen3-VL-4B-Instruct/config.json" ]]; then
    export QWEN3VL_PATH="$RUNTIME_ROOT/pretrained/lingbot_models/pretrained/Qwen3-VL-4B-Instruct"
elif [[ -n "$QWEN3VL_PATH" ]]; then
    export QWEN3VL_PATH
fi

PYTHONPATH="$RUNTIME_ROOT:$REPO_ROOT:${PYTHONPATH:-}" python - <<'PY'
import importlib.util
import sys

missing = []
for module in ("transformers.models.qwen3_vl", "safetensors", "websockets", "scipy"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)

if missing:
    print(
        "LingBotVLA environment is missing required modules: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    print(
        "Use a separate env matching requirements.txt, for example "
        "CONDA_ENV=lingbotvla.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

args=(
    --model_path "$MODEL_PATH"
    --robo_name "$ROBO_NAME"
    --host "$HOST"
    --port "$PORT"
    --use_length "$USE_LENGTH"
)

if [[ -n "$NORM_PATH" ]]; then
    args+=(--norm_path "$NORM_PATH")
fi

if [[ "$USE_COMPILE" != "0" && "$USE_COMPILE" != "false" && "$USE_COMPILE" != "False" ]]; then
    args+=(--use_compile)
fi

if [[ "$PRESERVE_HEAD_FROM_STATE" != "0" && "$PRESERVE_HEAD_FROM_STATE" != "false" && "$PRESERVE_HEAD_FROM_STATE" != "False" ]]; then
    args+=(--preserve_head_from_state)
fi

echo "Starting LingBotVLA V2 S1 server"
echo "  checkpoint: $CHECKPOINT_DIR"
echo "  config:     $TRAINING_CONFIG"
echo "  robot cfg:  $ROBOT_CONFIG"
echo "  norm:       $NORM_PATH"
echo "  robo_name:  $ROBO_NAME"
echo "  listen:     ws://$HOST:$PORT"

cd "$RUNTIME_ROOT"
PYTHONPATH="$RUNTIME_ROOT:$REPO_ROOT:${PYTHONPATH:-}" python -m deploy.s1_websocket_server "${args[@]}"
