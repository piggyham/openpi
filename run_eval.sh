#!/bin/bash
# 评测脚本:MuJoCo 仿真 + in-process policy
# 用法: run_eval.sh <checkpoint_step> <episodes> <start_seed> <gpu>
# 注:本机 EGL 初始化失败(nvidia/mesa 都不行),用 osmesa 软件渲染(已验证可用)
set -e
CKPT_STEP=$1
EPISODES=${2:-20}
START_SEED=${3:-0}
GPU=${4:-0}

cd /data/xuyuanxiang_proj/openpi
export VIRTUAL_ENV=/data/xuyuanxiang_proj/openpi/.venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PROJECT_DIR=/data/xuyuanxiang_proj/openpi
export OPENARM_DATA_DIR="$PROJECT_DIR/openarm_mission/artifacts/p10_new_camera_full"
export HF_LEROBOT_HOME="$OPENARM_DATA_DIR/lerobot"
export HF_HOME="$HF_LEROBOT_HOME/.hf_cache"
export OPENPI_DATA_HOME=/root/.cache/openpi
export CUDA_VISIBLE_DEVICES=$GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.30
export NCCL_SOCKET_IFNAME=lo
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

python -m openarm_mission.policy_eval \
  --policy-dir "checkpoints/pi05_openarm_paper_cup_relay/p10_new_camera_fullft_7xa100_bs56/${CKPT_STEP}" \
  --policy-config pi05_openarm_paper_cup_relay \
  --episodes "$EPISODES" \
  --start-seed "$START_SEED" \
  --video-out "openarm_mission/artifacts/p10_new_camera_full/eval/fullft_${CKPT_STEP}" \
  --no-npz
