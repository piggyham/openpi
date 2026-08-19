# P9 命令大全

本文档记录 P9 阶段采集、转换、验证双臂纸杯接力仿真数据所用全部命令。

## 1. 数据采集

### 冒烟采集（2 条，用于快速验证）
```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.collect_dataset \
  --output-dir openarm_mission/artifacts/p9_smoke \
  --episodes 2 --start-seed 0 --workers 2 --fps 20 \
  --width 640 --height 480 --image-episodes 2
```
- **作用**：采集 2 条 640×480 三相机演示，验证采集链路正常
- **参数**：`--workers 2` 限制并发数避免 EGL 溢出；`--image-episodes 2` 为每集保存图像

### 全量采集（200 条）
```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.collect_dataset \
  --output-dir openarm_mission/artifacts/p9 \
  --episodes 200 --start-seed 0 --workers 16 --fps 20 \
  --width 640 --height 480 --image-episodes 200
```
- **作用**：全量采集 200 条 640×480 三相机演示
- **参数**：`--workers 16` 全速并行；`--image-episodes 200` 每集保存图像
- **预置条件**：`export HF_LEROBOT_HOME=openarm_mission/artifacts/p5/lerobot`
- **注意**：需要 stash 掉 config.py/model.py 的本地修改（专家 IK 对场景参数敏感）

## 2. 格式转换

### 冒烟转换
```bash
.venv/bin/python -m openarm_mission.convert_to_openarm_v03 \
  --source openarm_mission/artifacts/p9_smoke \
  --output openarm_mission/artifacts/p9_smoke/openarm_paper_cup_relay
```
- **作用**：将 P9 冒烟 npz 转换为 v0.3.0 格式输出

### 全量转换
```bash
.venv/bin/python -m openarm_mission.convert_to_openarm_v03 \
  --source openarm_mission/artifacts/p9 \
  --output openarm_mission/artifacts/p9/openarm_paper_cup_relay \
  --overwrite
```
- **作用**：将 200 条 npz 全部转换为 v0.3.0 格式
- **参数**：`--overwrite` 覆盖已存在的输出目录

## 3. 厂商工具链验证

### validate.py
```bash
PYTHONPATH=openarm_mission/artifacts/p5/lerobot/openarm_paper_cup_relay/data \
  .venv/bin/python \
  openarm_mission/artifacts/p5/lerobot/openarm_paper_cup_relay/data/openarm_dataset/validate.py \
  <输出目录>
```
- **作用**：运行厂商数据集校验器，静默退出 0 为通过

### Dataset 全量读取
```bash
PYTHONPATH=openarm_mission/artifacts/p5/lerobot/openarm_paper_cup_relay/data \
  .venv/bin/python -c "
from openarm_dataset import Dataset
ds = Dataset('<输出目录>')
print(ds.num_episodes, 'episodes')
for ep in ds.meta.episodes[:3]:
    obs = ds.load_obs(ep)
    act = ds.load_action(ep)
    print(f'ep {ep[\"id\"]}: obs {list(obs.keys())}, act {list(act.keys())}')
assert ds.validate(), 'validate failed'
"
```
- **作用**：用厂商 Dataset 类加载所有 episode，验证 obs/action schema 正确

## 4. 仓库读取器验证

```bash
.venv/bin/python -m openarm_mission.openarm_sim.real_data
```
- **作用**：用本仓库 `real_data.py` 读取器加载所有符合格式的 episode
- **注意**：该读取器默认查找 `data/real_data/` 子目录

## 5. 结构对比

```bash
.venv/bin/python -c "
import pyarrow.parquet as pq
real = pq.read_schema('openarm_mission/artifacts/p5/lerobot/.../real_data/.../state.parquet')
sim = pq.read_schema('openarm_mission/artifacts/p9/.../state.parquet')
print('Match:', real.equals(sim, check_metadata=False))
"
```
- **作用**：对比仿真输出与真机数据的 parquet schema 是否一致

## 6. 单测

```bash
.venv/bin/python -m pytest openarm_mission/tests/test_convert_to_openarm_v03.py -v
.venv/bin/python -m pytest openarm_mission/tests/test_dataset.py -v
```
- **作用**：运行转换器测试（4 项）和数据集测试（3 项）

## 7. 数据恢复

```bash
git stash pop
```
- **作用**：将采集前 stash 的 config.py/model.py 本地修改恢复

## 8. 关键文件位置

| 产出 | 路径 |
|---|---|
| 冒烟 npz | `openarm_mission/artifacts/p9_smoke/raw/` |
| 冒烟 v0.3.0 | `openarm_mission/artifacts/p9_smoke/openarm_paper_cup_relay/` |
| 全量 npz | `openarm_mission/artifacts/p9/raw/` |
| 全量 v0.3.0 | `openarm_mission/artifacts/p9/openarm_paper_cup_relay/` |
| 转换器 | `openarm_mission/convert_to_openarm_v03.py` |
| 转换器测试 | `openarm_mission/tests/test_convert_to_openarm_v03.py` |
| 厂商 Dataset | `openarm_mission/artifacts/p5/lerobot/openarm_paper_cup_relay/data/openarm_dataset/` |
| 阶段文档 | `openarm_mission/phase_docs/p9.md` |

## 9. 环境变量

| 变量 | 作用 |
|---|---|
| `MUJOCO_GL=egl` | 启用离屏 EGL 渲染（无显示器时必需） |
| `HF_LEROBOT_HOME=.../artifacts/p5/lerobot` | 定位 LeRobot 数据集（采集必需） |
| `PYTHONPATH=.../data` | 定位厂商 `openarm_dataset` 包 |