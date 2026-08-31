#!/usr/bin/env bash
# 正式训练启动脚本。
# 用法: bash run_train.sh [head|stereo|fridge|fridge_head] [额外的 --key value 覆盖参数...]
#
# 环境依赖（已在本机装好）：
#   - lerobot 0.3.2  (CODEBASE_VERSION v2.1，支持本数据集的 v2.0 布局；
#                     0.4.x 是 v3.0-only，会硬拒绝)
#   - flash-attn 2.8.3 (ViT 走 flash_attention_2，硬依赖)
#   - FFmpeg 运行时库 (torchcodec 解码 mp4 需要)
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VARIANT="${1:-head}"
shift || true

# 根据 variant 确定配置路径
case "$VARIANT" in
  head|stereo|head_v2|stereo_v2)
    CFG="configs/vla/s1_stationary/s1_stationary_${VARIANT}.yaml"
    ;;
  fridge|fridge_head)
    CFG="configs/vla/s1_fridge/s1_${VARIANT}.yaml"
    ;;
  fridge_qrot|fridge_head_qrot)
    CFG="configs/vla/s1_fridge_qrot/s1_${VARIANT}.yaml"
    ;;
  *)
    echo "unknown variant: $VARIANT (expected: head|stereo|head_v2|stereo_v2|fridge|fridge_head|fridge_qrot|fridge_head_qrot)" >&2
    exit 1
    ;;
esac

[[ -f "$CFG" ]] || { echo "config not found: $CFG" >&2; exit 1; }

LOG="logs/train_${VARIANT}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export WANDB_PROJECT="lingbotvla-s1"

# 出网必须走企业代理。不能依赖 ~/.bashrc —— 它开头有
# `[ -z "$PS1" ] && return`，非交互式 shell（ssh 跑训练就是）会直接返回，
# 后面的 proxy export 永远不执行。
# 大小写两套都设：wandb 0.21 的后端是 Go 写的 (wandb-core)，Go 的
# http.ProxyFromEnvironment 认 HTTP_PROXY/HTTPS_PROXY/NO_PROXY。
PROXY="${PROXY:-http://10.5.0.191:6666}"
NOPROXY="apt.ksyun.cn,10.0.0.0/8,127.0.0.1,localhost,pypi.ksyun.cn,198.18.0.0/15"
export http_proxy="$PROXY"  https_proxy="$PROXY"
export HTTP_PROXY="$PROXY"  HTTPS_PROXY="$PROXY"
export no_proxy="$NOPROXY"  NO_PROXY="$NOPROXY"

# 两个节点都出现过 wandb 建 run 超时（默认 90s），放宽一点。
export WANDB__SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=300

NPROC="${NPROC:-$(nvidia-smi -L | wc -l)}"
MASTER_PORT="${MASTER_PORT:-62620}"

echo "variant : $VARIANT"
echo "config  : $CFG"
echo "GPUs    : $NPROC"
echo "log     : $LOG"
echo "wandb   : $WANDB_PROJECT"
echo

.venv/bin/torchrun \
  --nnodes=1 --nproc-per-node "$NPROC" --master-port="$MASTER_PORT" \
  tasks/vla/train_lingbotvla.py "$CFG" "$@" \
  > "$LOG" 2>&1 &

PID=$!
echo "started pid=$PID"
echo "跟踪进度: tail -f $LOG"
echo "只看 loss: grep -a 'INFO - __main__ - Step' $LOG | tail"
wait $PID
echo "exit code: $?"
