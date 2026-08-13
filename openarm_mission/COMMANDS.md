# OpenArm 常用命令手册

本文记录当前 OpenArm 双臂接力项目中已经实际使用或验证过的命令。当前脚本专家
从自然下垂启动，第一步只移动 J1/J4：左 J1 `+0.55 rad`、右 J1 `-0.55 rad`、
J4 `π/2`，其他关节保持不动。

## 1. 环境与路径

所有命令默认在项目根目录执行：

```bash
cd /home/piggyham/aaaxuyuanxiang/openPi/openpi
source .venv/bin/activate
```

建议为当前全量数据设置变量，减少路径输错：

```bash
export OPENARM_RAW_DIR="$PWD/openarm_mission/artifacts/p10_j1_j4_055_full"
export OPENARM_LEROBOT_ROOT="$OPENARM_RAW_DIR/lerobot"
export HF_HOME="$OPENARM_LEROBOT_ROOT/.hf_cache"
export HF_LEROBOT_HOME="$OPENARM_LEROBOT_ROOT"
```

检查磁盘、GPU 和相关进程：

```bash
df -h .
nvidia-smi
ps -eo pid,etime,cmd | grep -E 'collect_dataset|convert_to_lerobot|compute_norm_stats|scripts/train.py|openarm_sim.server' | grep -v grep
```

## 2. 基础与专家冒烟测试

基础模型、IK 和物理测试：

```bash
MUJOCO_GL=egl python -m openarm_mission.smoke_test \
  --render-path openarm_mission/artifacts/smoke/render.png
```

运行一条纯摩擦脚本专家，不生成视频：

```bash
MUJOCO_GL=egl python -m openarm_mission.friction_expert \
  --seed 0 \
  --output-dir openarm_mission/artifacts/friction_smoke
```

生成专家演示视频：

```bash
MUJOCO_GL=egl python -m openarm_mission.friction_expert \
  --seed 0 \
  --output-dir openarm_mission/artifacts/friction_smoke_video \
  --video
```

运行 J1/J4 结构测试：

```bash
python -m unittest \
  openarm_mission.tests.test_expert.RelayScriptedExpertTest.test_first_unfold_waypoint_moves_only_j1_and_j4
```

## 3. 数据采集

### 3.1 单集冒烟采集

采集 1 集、20 Hz、三相机 640×480：

```bash
MUJOCO_GL=egl python -m openarm_mission.collect_dataset \
  --output-dir openarm_mission/artifacts/p10_smoke_j1_j4_055 \
  --episodes 1 \
  --start-seed 0 \
  --workers 1 \
  --fps 20 \
  --width 640 \
  --height 480 \
  --image-episodes 1
```

### 3.2 200 集全量采集

```bash
MUJOCO_GL=egl python -m openarm_mission.collect_dataset \
  --output-dir "$OPENARM_RAW_DIR" \
  --episodes 200 \
  --start-seed 0 \
  --workers 8 \
  --fps 20 \
  --width 640 \
  --height 480 \
  --image-episodes 200
```

中断后断点续采：

```bash
MUJOCO_GL=egl python -m openarm_mission.collect_dataset \
  --output-dir "$OPENARM_RAW_DIR" \
  --episodes 200 \
  --start-seed 0 \
  --workers 8 \
  --fps 20 \
  --width 640 \
  --height 480 \
  --image-episodes 200 \
  --resume
```

快速检查采集 manifest：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

p = Path(os.environ["OPENARM_RAW_DIR"]) / "manifest.json"
m = json.loads(p.read_text())
for key in (
    "version", "episodes", "successful_episodes", "all_valid", "fps",
    "image_size", "image_episodes", "total_frames", "total_bytes",
):
    print(key, m.get(key))
print("invalid", sum(not r["validation"]["valid"] for r in m["records"]))
PY
```

当前 `J1=±0.55` 全量数据的预期结果是 200/200、125200 帧、每集 626 帧。

## 4. 转换为 LeRobot

首次转换：

```bash
python -m openarm_mission.convert_to_lerobot \
  --source-dir "$OPENARM_RAW_DIR" \
  --output-dir "$OPENARM_LEROBOT_ROOT/openarm_paper_cup_relay" \
  --repo-id local/openarm_paper_cup_relay
```

如果上次转换中断，需要删除不完整输出并重建，可使用转换器的覆盖选项：

```bash
python -m openarm_mission.convert_to_lerobot \
  --source-dir "$OPENARM_RAW_DIR" \
  --output-dir "$OPENARM_LEROBOT_ROOT/openarm_paper_cup_relay" \
  --repo-id local/openarm_paper_cup_relay \
  --overwrite
```

检查转换报告：

```bash
python -m json.tool \
  "$OPENARM_LEROBOT_ROOT/openarm_paper_cup_relay/meta/openarm_conversion.json"
```

报告应包含 200 episodes、125200 frames，并且 `load_validation` 为 `true`。

## 5. 转换为 OpenArm v0.3.0

该格式供 OpenArmSim 直接发现 episode，也可用于厂商数据工具链：

```bash
python -m openarm_mission.convert_to_openarm_v03 \
  --source "$OPENARM_RAW_DIR" \
  --output "$OPENARM_RAW_DIR/openarm_paper_cup_relay_v03" \
  --overwrite
```

单集冒烟转换示例：

```bash
python -m openarm_mission.convert_to_openarm_v03 \
  --source openarm_mission/artifacts/p10_smoke_j1_j4_055 \
  --output openarm_mission/artifacts/p10_smoke_j1_j4_055/openarm_paper_cup_relay \
  --overwrite
```

## 6. OpenArmSim 回放

离线自检动态模式、运动学模式、seek 和四视角渲染：

```bash
MUJOCO_GL=egl python -m openarm_mission.openarm_sim.playback \
  --data-dir openarm_mission/artifacts/p10_smoke_j1_j4_055/openarm_paper_cup_relay
```

仅在本机启动网页服务，避免将采集图像暴露到局域网：

```bash
MUJOCO_GL=egl python -m openarm_mission.openarm_sim.server \
  --host 127.0.0.1 \
  --port 8081 \
  --data-dir openarm_mission/artifacts/p10_smoke_j1_j4_055/openarm_paper_cup_relay \
  --real-fps 20
```

浏览器打开 <http://127.0.0.1:8081/>。

同时加载冒烟和全量 v0.3.0 数据：

```bash
MUJOCO_GL=egl python -m openarm_mission.openarm_sim.server \
  --host 127.0.0.1 \
  --port 8081 \
  --data-dir openarm_mission/artifacts/p10_smoke_j1_j4_055/openarm_paper_cup_relay \
  --data-dir "$OPENARM_RAW_DIR/openarm_paper_cup_relay_v03" \
  --real-fps 20
```

查询和关闭服务：

```bash
ss -ltnp | grep ':8081'
kill -TERM <PID>
```

## 7. 重新计算归一化统计

每次改变轨迹脚本或重新采集数据后都必须重算。LoRA 配置共享 full config 的
统计文件，因此使用 `pi05_openarm_paper_cup_relay`：

```bash
HF_HOME="$HF_HOME" \
HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
python scripts/compute_norm_stats.py \
  --config-name pi05_openarm_paper_cup_relay
```

输出位置：

```text
assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json
```

计算前备份旧统计：

```bash
cp assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json \
  assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.before_j1_j4_055.json
```

检查夹爪统计和文件摘要：

```bash
sha256sum assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json
python -m json.tool assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json | less
```

## 8. LoRA 微调与 W&B

确认 W&B 登录：

```bash
wandb status
```

推荐在 RTX 4090 上使用 batch size 1。以下命令从官方 π0.5 base 开始一版新的
LoRA 微调，训练 30000 steps，每 50 steps 上传指标，每 5000 steps 保存：

```bash
export WANDB_DIR="$PWD/wandb"
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

python scripts/train.py pi05_openarm_paper_cup_relay_lora \
  --exp-name p10_j1_j4_055_state_lora_bs1 \
  --batch-size 1 \
  --num-train-steps 30000 \
  --log-interval 50 \
  --save-interval 5000 \
  --keep-period 5000
```

Checkpoint 输出位置：

```text
checkpoints/pi05_openarm_paper_cup_relay_lora/p10_j1_j4_055_state_lora_bs1/
```

若实验名目录已存在并且明确要覆盖，可在训练命令末尾加 `--overwrite`。不要在
未核对目录内容时使用该选项。

检查训练和 GPU：

```bash
nvidia-smi
ps -eo pid,etime,cmd | grep 'scripts/train.py' | grep -v grep
find checkpoints/pi05_openarm_paper_cup_relay_lora/p10_j1_j4_055_state_lora_bs1 \
  -maxdepth 1 -type d -printf '%f\n' | sort -n
```

## 9. 模型闭环评测

推荐用同进程推理，避免单张 GPU 上策略服务与 MuJoCo EGL 争抢上下文：

```bash
MUJOCO_GL=egl \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python -m openarm_mission.policy_eval \
  --episodes 5 \
  --start-seed 1000 \
  --max-steps 600 \
  --policy-dir checkpoints/pi05_openarm_paper_cup_relay_lora/p10_j1_j4_055_state_lora_bs1/29999 \
  --policy-config pi05_openarm_paper_cup_relay_lora \
  --video-out openarm_mission/artifacts/eval_p10_j1_j4_055_lora_29999_seeds1000_1004
```

测试 5000 steps checkpoint 时，只需把 `--policy-dir` 中的 `29999` 改为
`5000`，并使用新的 `--video-out`，避免覆盖结果。

评测目录会生成：

```text
rollout_seedXXXXXX_success.mp4 或 rollout_seedXXXXXX_failure.mp4
rollout_seedXXXXXX_success.npz 或 rollout_seedXXXXXX_failure.npz
summary.json
```

## 10. 可选：启动策略服务

如果需要从其他客户端通过 WebSocket 查询策略：

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_openarm_paper_cup_relay_lora \
  --policy.dir checkpoints/pi05_openarm_paper_cup_relay_lora/p10_j1_j4_055_state_lora_bs1/29999 \
  --port 8000
```

该服务默认绑定所有网卡。只在确实需要其他机器访问、且网络环境可信时使用；本机
评测优先使用上一节的 `--policy-dir` 同进程方式。

## 11. 代码检查

```bash
ruff check \
  openarm_mission/config.py \
  openarm_mission/p3_episode.py \
  openarm_mission/tests/test_expert.py

ruff format --check \
  openarm_mission/config.py \
  openarm_mission/p3_episode.py \
  openarm_mission/tests/test_expert.py
```

运行 OpenArm 测试：

```bash
python -m unittest discover -s openarm_mission/tests -p 'test_*.py'
```

## 12. 数据清理（谨慎）

先检查目标大小和绝对路径：

```bash
du -sh "$OPENARM_RAW_DIR"
realpath "$OPENARM_RAW_DIR"
```

确认目录无误后才删除。以下命令会永久删除当前全量原始数据及其所有转换结果：

```bash
rm -rf -- "$OPENARM_RAW_DIR"
```

不要删除以下目录，除非明确决定丢弃模型或训练配置：

```text
checkpoints/
assets/pi05_openarm_paper_cup_relay/
openarm_mission/artifacts/p10_smoke_j1_j4_055/
```

## 13. 推荐执行顺序

新版本标准流程：

```text
专家冒烟 → 单集图像采集 → OpenArmSim 回放 → 200 集全量采集
→ LeRobot 转换 → 转换报告校验 → 重算归一化 → LoRA 微调
→ W&B 检查 loss → 5-seed 闭环评测
```
