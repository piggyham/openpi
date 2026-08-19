# OpenArm pi0.5 LoRA：A100 八卡训练交接文档

> 本文档供 A100 服务器上的 AI/运维代理直接执行。用户使用 FileZilla（SFTP）传输源码、数据和基础模型；不要使用 GitHub，不要执行 `git clone`、`git pull` 或 `gh`。在任何删除、覆盖或终止进程操作前，必须先确认目标属于本次任务；不得终止其他用户的进程。

## 1. 目标与已知信息

- 项目目录：`/root/xuyuanxiang_proj/openpi`
- 数据集：OpenArm 双臂纸杯接力，新相机视角，全量 200 episodes
- 数据格式：本地 LeRobot v2
- 数据集 ID：`openarm_paper_cup_relay`
- 数据规模：200 episodes、125200 frames、20 FPS
- 数据划分：train `0:160`、validation `160:180`、test `180:200`
- 模型：pi0.5 base + LoRA
- 训练设备：8 张 A100
- W&B：必须启用，监控 `loss`、`grad_norm`、`param_norm` 和训练速度
- 正式训练：30000 steps，每 5000 steps 保留检查点

不要假定 Git 工作区是干净的。本项目包含训练所必需的未提交/新增文件，FileZilla 传来的实际文件内容才是本次训练的代码基线。

## 2. FileZilla 应已传到服务器的文件

### 2.1 源码

远端根目录：

```text
/root/xuyuanxiang_proj/openpi
```

必须存在：

```text
pyproject.toml
uv.lock
src/openpi/policies/openarm_policy.py
src/openpi/training/config.py
scripts/train.py
scripts/compute_norm_stats.py
openarm_mission/convert_to_lerobot.py
packages/
```

### 2.2 已转换训练数据

```text
/root/xuyuanxiang_proj/openpi/openarm_mission/artifacts/p10_new_camera_full/lerobot/openarm_paper_cup_relay
```

转换报告：

```text
/root/xuyuanxiang_proj/openpi/openarm_mission/artifacts/p10_new_camera_full/lerobot/openarm_paper_cup_relay/meta/openarm_conversion.json
```

### 2.3 pi0.5 基础模型

```text
/root/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

训练命令会显式使用这个本地路径，避免服务器尝试从 GCS/GitHub 下载。

## 3. 第一阶段：只读检查

先执行以下命令，不要立即开始训练：

```bash
cd /root/xuyuanxiang_proj/openpi

pwd
df -h
nvidia-smi -L

test -f pyproject.toml
test -f src/openpi/policies/openarm_policy.py
test -f src/openpi/training/config.py
test -f scripts/train.py
test -f scripts/compute_norm_stats.py
test -f openarm_mission/artifacts/p10_new_camera_full/lerobot/openarm_paper_cup_relay/meta/openarm_conversion.json
test -d /root/.cache/openpi/openpi-assets/checkpoints/pi05_base/params
```

检查转换报告：

```bash
python -m json.tool \
  openarm_mission/artifacts/p10_new_camera_full/lerobot/openarm_paper_cup_relay/meta/openarm_conversion.json
```

必须确认：

```text
episodes = 200
frames = 125200
fps = 20
load_validation = true
```

检查磁盘占用：

```bash
du -sh openarm_mission/artifacts/p10_new_camera_full/lerobot/openarm_paper_cup_relay
du -sh /root/.cache/openpi/openpi-assets/checkpoints/pi05_base
df -h /root/xuyuanxiang_proj/openpi
```

训练前建议至少保留 150 GB 可用空间；如检查点保留策略发生变化，应重新估算空间。

## 4. 训练前必须检查右腕相机 mask

打开：

```text
/root/xuyuanxiang_proj/openpi/src/openpi/policies/openarm_policy.py
```

在 `OpenArmInputs.__call__` 的 `image_mask` 中，三路真实相机都必须为 `True`：

```python
"image_mask": {
    "base_0_rgb": np.True_,
    "left_wrist_0_rgb": np.True_,
    "right_wrist_0_rgb": np.True_,
},
```

如果右腕仍是以下条件表达式：

```python
np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
```

必须将其改成 `np.True_` 后再训练。当前模型是 PI05，原条件会错误屏蔽真实的右腕相机。修改后运行语法检查：

```bash
python -m compileall -q src scripts openarm_mission \
  -x '(^|/)(artifacts|third_party)(/|$)'
```

## 5. 环境配置

进入已有虚拟环境：

```bash
cd /root/xuyuanxiang_proj/openpi
source .venv/bin/activate
```

如果 `.venv` 不存在，先检查服务器是否有已经配置好的 OpenPI/uv 环境。不要从 GitHub 克隆依赖。由于 `pyproject.toml` 中 LeRobot 可能是 GitHub 源，服务器无法访问 GitHub 时直接执行 `uv sync` 可能失败；此时停止并向用户报告缺失的依赖包，等待通过 FileZilla 提供离线依赖或配置可用的软件源，不要上传/复用另一台机器的普通 `.venv` 目录。

设置本次训练环境变量：

```bash
export PROJECT_DIR=/root/xuyuanxiang_proj/openpi
export OPENARM_DATA_DIR="$PROJECT_DIR/openarm_mission/artifacts/p10_new_camera_full"
export HF_LEROBOT_HOME="$OPENARM_DATA_DIR/lerobot"
export HF_HOME="$HF_LEROBOT_HOME/.hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"
export OPENPI_DATA_HOME=/root/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

mkdir -p "$HF_HOME" "$WANDB_DIR"
```

验证 Python、配置和八张 GPU：

```bash
python - <<'PY'
import jax
from openpi.training import config

cfg = config.get_config("pi05_openarm_paper_cup_relay_lora")
print("config:", cfg.name)
print("backend:", jax.default_backend())
print("device_count:", jax.device_count())
print("devices:", jax.devices())
assert jax.default_backend() == "gpu"
assert jax.device_count() == 8
PY
```

## 6. 归一化统计

即使服务器收到了旧的 `norm_stats.json`，也应针对 `p10_new_camera_full` 再计算一次，避免误用旧数据统计。

```bash
cd /root/xuyuanxiang_proj/openpi
source .venv/bin/activate

export HF_LEROBOT_HOME=/root/xuyuanxiang_proj/openpi/openarm_mission/artifacts/p10_new_camera_full/lerobot
export HF_HOME="$HF_LEROBOT_HOME/.hf_cache"

python scripts/compute_norm_stats.py \
  --config-name pi05_openarm_paper_cup_relay
```

预期输出：

```text
/root/xuyuanxiang_proj/openpi/assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json
```

必须检查文件存在且为合法 JSON：

```bash
test -f assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json
python -m json.tool \
  assets/pi05_openarm_paper_cup_relay/openarm_paper_cup_relay/norm_stats.json >/dev/null
```

## 7. W&B 登录与监控

```bash
wandb login
wandb status
```

如果用户已经登录，禁止重复覆盖其 W&B 配置。确认 `WANDB_MODE` 没有被设置成 `offline` 或 `disabled`：

```bash
env | grep '^WANDB_' || true
```

训练时 W&B 应看到：

- `loss`
- `grad_norm`
- `param_norm`
- step 和训练速度
- step 0 的三相机预览图

## 8. 八卡冒烟训练

首次运行用全局 batch size 8，即每张 GPU 1 个样本；`--fsdp-devices 1` 表示八路数据并行，适用于能够在单张 A100 上容纳的 LoRA 模型。

```bash
cd /root/xuyuanxiang_proj/openpi
source .venv/bin/activate

python scripts/train.py pi05_openarm_paper_cup_relay_lora \
  --exp-name p10_new_camera_lora_8xa100_smoke \
  --weight-loader.params-path /root/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --batch-size 8 \
  --fsdp-devices 1 \
  --num-workers 8 \
  --num-train-steps 20 \
  --log-interval 1 \
  --save-interval 20 \
  --keep-period 20 \
  --wandb-enabled \
  --wandb-log-images
```

并行监控：

```bash
watch -n 1 nvidia-smi
```

冒烟验收条件：

1. JAX 识别 8 张 GPU。
2. 八张 GPU 都有本次 Python 训练进程和显存占用。
3. 能取出首个 batch，无数据字段、图像 key、归一化或 shape 错误。
4. loss 是有限数，不是 `NaN`/`Inf`。
5. W&B 能收到 loss；step 0 三相机图像方向和内容正常。
6. 20 steps 完成并写出检查点。

若冒烟失败，保留完整错误日志并诊断，不得直接启动正式训练。

## 9. 正式 LoRA 训练

冒烟通过后，在 `tmux` 中启动：

```bash
tmux new -s openpi_a100
```

然后执行：

```bash
cd /root/xuyuanxiang_proj/openpi
source .venv/bin/activate

export PROJECT_DIR=/root/xuyuanxiang_proj/openpi
export OPENARM_DATA_DIR="$PROJECT_DIR/openarm_mission/artifacts/p10_new_camera_full"
export HF_LEROBOT_HOME="$OPENARM_DATA_DIR/lerobot"
export HF_HOME="$HF_LEROBOT_HOME/.hf_cache"
export WANDB_DIR="$PROJECT_DIR/wandb"
export OPENPI_DATA_HOME=/root/.cache/openpi
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

python scripts/train.py pi05_openarm_paper_cup_relay_lora \
  --exp-name p10_new_camera_state_lora_8xa100_bs64 \
  --weight-loader.params-path /root/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --batch-size 64 \
  --fsdp-devices 1 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --log-interval 50 \
  --save-interval 5000 \
  --keep-period 5000 \
  --wandb-enabled \
  --wandb-log-images \
  2>&1 | tee a100_p10_new_camera_lora.log
```

`batch-size 64` 是全局 batch，八卡时每卡 8。若出现显存不足，按照 `64 -> 32 -> 16 -> 8` 下调，始终保证 batch size 能被 8 整除。不要通过终止其他用户进程来释放显存；先确认 GPU 是否被占用并向用户报告。

检查点目录：

```text
/root/xuyuanxiang_proj/openpi/checkpoints/pi05_openarm_paper_cup_relay_lora/p10_new_camera_state_lora_8xa100_bs64/
```

## 10. 断点续训

只能使用同一个实验名称，并添加 `--resume`。禁止同时使用 `--resume` 和 `--overwrite`。

```bash
python scripts/train.py pi05_openarm_paper_cup_relay_lora \
  --exp-name p10_new_camera_state_lora_8xa100_bs64 \
  --weight-loader.params-path /root/.cache/openpi/openpi-assets/checkpoints/pi05_base/params \
  --batch-size 64 \
  --fsdp-devices 1 \
  --num-workers 8 \
  --num-train-steps 30000 \
  --log-interval 50 \
  --save-interval 5000 \
  --keep-period 5000 \
  --wandb-enabled \
  --resume
```

## 11. 训练期间报告要求

A100 AI 应向用户报告：

1. 代码、数据、基础模型和归一化统计是否齐全。
2. JAX 识别到的 GPU 数量。
3. 是否修正右腕相机 mask。
4. 冒烟训练的最终 loss、grad norm、W&B run 链接和检查点路径。
5. 正式训练的 PID/tmux 会话、W&B run 链接和日志路径。
6. 每个 5000-step 检查点的 loss 趋势和路径。
7. 如发生 OOM、NaN、数据加载错误或掉卡，提供完整错误末尾、发生 step 和已执行的只读诊断。

未经用户授权，不删除数据、不覆盖已有正式实验、不杀其他用户的进程。
