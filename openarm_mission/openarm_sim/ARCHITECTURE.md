# openarm_sim 架构文档

## 概述

`openarm_sim` 是一个基于 MuJoCo 的姿态回放与真机实时镜像系统。输入 16 维关节角序列（左臂 7 关节 + 夹爪开度 + 右臂 7 关节 + 夹爪开度），在浏览器中实时渲染机器人运动。支持 LeRobot、OpenArm v0.3.0 和 OpenArm Panel SSE/MJPEG 实时源。真机语义固定为 Target=commanded、Actual=pos，三维机器人由 Actual 直接驱动。

## 文件架构

```
openarm_mission/
├── openarm_sim/                    # 仿真回放子系统（本目录）
│   ├── __init__.py                 # 包文档
│   ├── playback.py                 # ★ 核心：parquet 加载 + MuJoCo 仿真引擎
│   ├── real_data.py                # ★ 真机 episode 加载器
│   ├── live_source.py              # ★ Panel SSE/MJPEG 实时客户端
│   ├── server.py                   # ★ WebSocket 服务端 + 仿真线程桥接
│   ├── web/
│   │   └── index.html              # 单文件前端（内联 CSS/JS，无构建依赖）
│   ├── README.md                   # 使用说明
│   └── ARCHITECTURE.md             # 本文档
│
├── config.py                       # 场景配置（MissionConfig, ControllerConfig 等）
├── model.py                        # MuJoCo 模型加载 + 场景组合
├── controller.py                   # 双臂 Cartesian 力矩 PD 控制器
├── dataset.py                      # 数据集 schema（CAMERAS, STATE_NAMES 等）
│
├── fetch_openarm_v1.sh             # 官方模型下载脚本
├── policy_eval.py                  # 策略评估（对接 openpi 推理）
├── collect_dataset.py              # 数据采集
├── convert_to_lerobot.py           # 数据格式转换
│
├── artifacts/                      # 数据 + 模型（忽略于 git）
│   └── p5/lerobot/openarm_paper_cup_relay/data/
│       ├── chunk-*/episode_*.parquet    # LeRobot 仿真数据集
│       └── real_data/episodes/<id>/     # 真机遥操作数据
│
└── third_party/openarm_mujoco/     # 官方 OpenArm MuJoCo 模型（fetch_openarm_v1.sh 下载）
```

## 数据流全景

```
┌──────────────────────────────────────────────────────────────┐
│  数据源                                                       │
│  ┌─────────────────────┐  ┌──────────────────────────────┐   │
│  │ LeRobot parquet      │  │ v0.3.0 数据集(双布局)         │   │
│  │ (chunk-*/episode_*.  │  │ ├ real_data/episodes/<id>/   │   │
│  │  parquet, 20 Hz)     │  │ │    (真机遥操作)             │   │
│  │                      │  │ └ <root>/episodes/<id>/      │   │
│  │                      │  │     (P9 转换集, metadata.yaml │   │
│  │                      │  │      + episodes/, 下拉框 sim_*)│   │
│  │                      │  │   obs/arms/*/state.parquet   │   │
│  │                      │  │   cameras/*/*.jpeg           │   │
│  └────────┬─────────────┘  └──────────────┬───────────────┘   │
│           │                               │                   │
│           ▼                               ▼                   │
│  ┌──────────────────────┐  ┌──────────────────────────────┐   │
│  │ playback.py          │  │ real_data.py                 │   │
│  │ load_episode()       │  │ load_real_episode()          │   │
│  │ → EpisodeData        │  │ →EpisodeData(按 --real-fps    │   │
│  │                      │  │   重采样, P9 用 --real-fps 20)│   │
│  │                      │  │ 夹爪映射: 0.044*clip(1+raw,0,1)│   │
│  └────────┬─────────────┘  └──────────────┬───────────────┘   │
│           │                               │                   │
│           └───────────┬───────────────────┘                   │
│                       ▼                                       │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ server.py: SimEngine                                 │     │
│  │  ┌──────────────────┐   ┌────────────────────────┐   │     │
│  │  │ 仿真线程          │   │ asyncio 事件循环         │   │     │
│  │  │ SimPlayback      │──▶│ latest-frame slot      │   │     │
│  │  │  ◄── 命令队列      │   │  ◄── WebSocket        │   │     │
│  │  └──────────────────┘   └────────┬───────────────┘   │     │
│  └──────────────────────────────────┼────────────────────┘     │
│                                     │                          │
│                                     ▼                          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ web/index.html (浏览器)                                │     │
│  │  自由视角 + 三相机 + 关节数值表 + 趋势曲线 + 对比录制  │     │
│  └──────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. `playback.py` — 仿真引擎

**职责**：parquet 加载、EpisodeData 管理、MuJoCo 仿真步进、EGL 离屏渲染。

**关键数据类**：

| 类/函数 | 说明 |
|---|---|
| `EpisodeData` | 一个 episode 的完整数据：states `(N,16)`, timestamps `(N,)`, fps, 录制图像（懒加载），可选的 `image_factory` 回调 |
| `ViewerMissionConfig` | 查看器场景配置：机器人 + 带桌腿的桌子 + 水瓶 + 红/蓝交接杯垫标记（与专家采集场景一致） |
| `SimPlayback` | 仿真引擎本体：持有 `OpenArmMission`、`BimanualCartesianController`、EGL 渲染器、自由相机 |

**两种回放模式**：

| 模式 | 原理 | 特点 |
|---|---|---|
| **dynamic**（默认） | 关节目标 → PD 控制器 `compute_ctrl()` → `mj_step()` 物理积分 | 展示机器人实际的物理跟踪行为。**夹爪手指直接写 qpos**（非伺服积分），避免指垫自锁（见下文） |
| **kinematic** | 逐帧直写 `data.qpos` → `mj_forward()` 运动学正解 | 精确复现录制姿态，actual ≡ target |

> **动态模式手指直写的原因**：v1 手指位置伺服从关节 0.044 闭合到 0 时欠阻尼过冲；
> 当真机录制数据的"闭合"姿态在手指间没有物体（真机抓取位与 viewer 场景纸杯
> 不对齐）时，过冲使两个椭球指垫接触并**自锁**（曲面近顶端接触，法向力机械增益
> 极大，摩擦自锁，8 N 甚至 50 N 都打不开），此后整集 actual 恒显示闭合。手指
> 直接写目标可彻底避免该陷阱，且因手指伺服远快于手臂，回放帧率下等同瞬时到位。

**关键方法**：

```
load_episode(episode)   → 加载新 episode，reset 到帧 0
step()                  → 前进一帧，返回是否结束
seek(frame_index)       → 跳转到指定帧（dynamic 下从 0 重仿真到目标帧）
render()                → 渲染三相机 + 自由视角，返回 RGB ndarray dict
move_camera(action,dx,dy) → 自由相机旋转/平移/缩放（mjv_moveCamera 语义）
reset_camera()          → 重置自由相机到默认视角
```

**线程约束**：所有 MuJoCo/EGL 对象必须在同一线程创建和使用（EGL context 是线程局部的）。

### 2. `real_data.py` — v0.3.0 数据加载器

**职责**：发现并加载 OpenArm 数据集格式 v0.3.0 的 episode（真机遥操作
`data/real_data/episodes/<id>/`，或 P9 转换后的 v0.3.0 数据集根——
含 `metadata.yaml` + `episodes/`），将其转换为与 LeRobot parquet 兼容的
`EpisodeData`。

**数据格式**（OpenArm 数据集 v0.3.0，两种布局共用）：

```
real_data/episodes/<id>/            # 真机布局
<root>/episodes/<id>/               # P9 数据集根布局(如 artifacts/p9/openarm_paper_cup_relay/)
├── obs/arms/{left,right}/state.parquet    # 含 timestamp + qpos(8维)
└── cameras/{head,wrist_left,wrist_right}/<ns>.jpeg  # 相机 JPEG 流
```

**布局发现**：`_episode_root(data_dir)` 优先认 `<data-dir>/real_data/episodes/`；
否则当 `<data-dir>` 自身含 `metadata.yaml` + `episodes/` 时把它当作数据集根。
`episode_prefix(data_dir)` 据此返回 `real`（真机树）或 `sim`（P9 数据集根），
下拉框分别命名为 `real_<id>` / `sim_<id>`。

**映射规则**：

| 真机原始数据 | 查看器 16 维状态 |
|---|---|
| 左臂 qpos[0:7]（7 关节） | states[0:7]，直通 |
| 左夹爪 raw（0=闭合，越负越开） | states[7] = `0.044 × clip(1+raw, 0, 1)`（闭合量:0.044≈闭合、0=张开，playback 驱动取反） |
| 右臂 qpos[0:7]（7 关节） | states[8:15]，直通 |
| 右夹爪 raw | states[15] = `0.044 × clip(1+raw, 0, 1)` |
| head 相机 | → front |
| wrist_left 相机 | → left_wrist |
| wrist_right 相机 | → right_wrist |

**采样策略**：在左右臂公共时间区间上按 `--real-fps`（默认 30 Hz）均匀网格重采样，每个流取时间最近邻样本，与 `openarm_dataset` 转换包的语义一致。P9 数据本身是 20 Hz，回放时用 `--real-fps 20` 可保持原生帧率。

**关键函数**：

```
_episode_root(data_dir)        → 解析 v0.3.0 episodes/ 目录(双布局)
episode_prefix(data_dir)       → real(真机树) 或 sim(P9 数据集根)
list_real_episodes(data_dir)   → 发现所有可加载 episode 目录
load_real_episode(ep_dir, fps) → 加载并重采样一个 episode，返回 EpisodeData
_sample_values(..., linear)    → Actual 线性插值
_sample_values(..., zoh)       → Commanded Target 零阶保持
_nearest_indices(times, grid)  → 相机最近邻索引
_gripper_opening(raw)          → 夹爪 raw → 开度(m) 映射
```

### 3. `server.py` — WebSocket 服务端

**职责**：单端口同时服务 HTML 前端（GET /）和 WebSocket 流，在 asyncio 事件循环和仿真线程之间桥接。

**线程架构**：

```
┌─ asyncio 事件循环 ─────────────────────┐
│  websockets.serve                      │
│  ws_handler() ───▶ 命令入队             │
│  broadcast_frames() ◀── frame slot     │
│  http_hook() → GET / 返回 index.html   │
└────────────┬───────────────────────────┘
             │ thread-safe queue + latest-frame slot
┌────────────┴───────────────────────────┐
│  仿真线程 (openarm-sim)                 │
│  _sim_loop()                           │
│    逐帧: _drain_commands() → step()    │
│    → render() → _publish_frame()       │
└────────────────────────────────────────┘
```

**关键类**：`SimEngine` — 持有 `SimPlayback`、episode 列表、播放状态（播放/暂停/速度/模式/对比开关）。

**WebSocket 协议**：

| 方向 | 消息类型 | 说明 |
|---|---|---|
| 服务端 → 客户端 | `config` | 初始化：episode 列表、相机、分辨率、模式 |
| 服务端 → 客户端 | `status` | 播放状态变更 |
| 服务端 → 客户端 | `frame` | 帧数据：base64 JPEG 图像 + 目标/实际关节角 |
| 服务端 → 客户端 | `error` | 错误消息 |
| 客户端 → 服务端 | `select_episode` | 切换 episode |
| 客户端 → 服务端 | `play/pause` | 播放/暂停 |
| 客户端 → 服务端 | `seek` | 跳转到指定帧 |
| 客户端 → 服务端 | `set_speed` | 设置播放速度 (0.25×–4×) |
| 客户端 → 服务端 | `set_mode` | 切换 dynamic/kinematic |
| 客户端 → 服务端 | `set_compare` | 开关对比录制行 |
| 客户端 → 服务端 | `camera_move/reset/follow` | 自由相机控制 |

**多标签页**：共享同一会话；任一标签页的控制对所有标签页生效。慢客户端只丢帧不阻塞。

### 4. `web/index.html` — 前端

**职责**：单文件 HTML 前端（371 行），内联 CSS/JS，无构建工具、无 CDN 依赖。

**UI 布局**：

```
┌─ 顶部工具栏 ──────────────────────────────────────────┐
│ episode ▼ | ▶ ⏸ | speed ▼ | mode ▼ | compare ☐ | …  │
├─ 自由视角（大面板）─────────────────────────────────────┤
│ 左键拖动旋转 | Shift/右键平移 | 滚轮缩放                │
├─ 三相机实时流 ─────────────────────────────────────────┤
│ front │ left_wrist │ right_wrist                      │
├─ 对比录制行（可选）─────────────────────────────────────┤
│ 原始录制 front │ left_wrist │ right_wrist              │
├─ 关节角数值表 ─────────────────────────────────────────┤
│ target / actual 16 通道，误差超限标红                   │
├─ 关节角趋势曲线 ───────────────────────────────────────┤
│ 16 通道滚动时序图                                      │
└────────────────────────────────────────────────────────┘
```

**WebSocket 通信**：连接后接收 `config` → 初始化 UI → 循环接收 `frame`/`status` → 更新渲染。

## 关键依赖

### `config.py` — 场景与控制器配置

| 配置类 | 说明 |
|---|---|
| `MissionConfig` | 冻结的 P0 场景参数：桌面尺寸、区域位置、纸杯物理属性、场景组合开关 |
| `FrictionMissionConfig` | 继承 MissionConfig，开启摩擦 pad + 纸杯阻尼 |
| `ControllerConfig` | 控制器参数：力矩限幅、超速制动阈值、夹爪开度、阻尼刚度 |
| `ViewerMissionConfig` (playback.py) | 继承 FrictionMissionConfig，关闭纸杯和区域标记，开启桌腿 |

### `model.py` — MuJoCo 模型

| 组件 | 说明 |
|---|---|
| `OpenArmMission` | 顶层模型：加载官方 v1 模型 → 组合任务场景（桌子、纸杯、区域标记）→ 添加软指垫 |
| `ArmModelHandles` | 单臂 MuJoCo 索引解析：关节 ID、qpos 索引、驱动 ID、手指关节、TCP site |
| 场景组合 | `_add_soft_finger_pads()` 添加绿色半球指垫，`_add_task_scene()` 添加桌面/纸杯/标记 |

### `controller.py` — 控制器

| 组件 | 说明 |
|---|---|
| `BimanualCartesianController` | 双臂 Cartesian 阻抗控制器：`compute_ctrl()` 计算力矩指令写入 `data.ctrl` |
| `IKResult` | 逆运动学结果：收敛状态、位置/姿态误差、迭代次数 |

核心安全机制：力矩限幅（`ctrl_clip`）、超速制动（`max_velocity`）、动作软限位（`joint_soft_limit`）。

### `dataset.py` — 数据 schema

| 导出 | 说明 |
|---|---|
| `CAMERAS` | `{"front": "mission_front_camera", "left_wrist": ..., "right_wrist": ...}` |
| `STATE_NAMES` | 16 维状态命名：`left_joint_1_rad` … `left_gripper_opening_m` … `right_gripper_opening_m` |
| `hide_collision_geomgroups()` | 隐藏碰撞/debug 几何组（group 3+），减少 EGL 渲染负载 |

## 辅助文件

| 文件 | 说明 |
|---|---|
| `fetch_openarm_v1.sh` | 下载官方 OpenArm v1 MuJoCo 模型到 `third_party/openarm_mujoco/` |
| `policy_eval.py` | 对接 openpi 推理的策略评估脚本 |
| `collect_dataset.py` | P5 数据采集主脚本 |
| `convert_to_lerobot.py` | 将采集数据转换为 LeRobot v2 格式 |
| `scripts/serve_policy.py` | 策略推理服务 |
| `src/openpi/policies/openarm_policy.py` | OpenArm 策略定义 |

## 启动流程

```
1. 下载模型（首次）
   bash openarm_mission/fetch_openarm_v1.sh

2. 启动服务
   MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server --port 8080

3. 浏览器打开
   http://127.0.0.1:8080/

4. 关闭进程
   lsof -ti :8080 | xargs -r kill
```

## 设计要点

1. **EGL 线程局部性**：所有 MuJoCo/EGL 对象（model, data, renderer, controller）必须在同一线程创建和使用。`SimEngine` 将仿真隔离在专用线程中，通过队列和 slot 与 asyncio 通信。

2. **EpisodeData 解耦**：数据加载（playback.py / real_data.py）和仿真播放（SimPlayback）通过 `EpisodeData` 解耦，真机数据通过 `image_factory` 回调无缝接入录制图像显示。

3. **关节限位保护**：`SimPlayback._apply_frame()` 在 dynamic 和 kinematic 模式下均 clamp 关节目标到 v1 模型限位，防止真机数据（可能略微超出仿真范围）导致 MuJoCo 异常。

4. **MuJoCo 2.3.7 兼容**：`MjvCamera.lookat` 和 `MjvOption.geomgroup` 存在 strides 缺陷，代码中通过 ctypes 直写绕过（`playback.py::_set_lookat`, `dataset.py::hide_collision_geomgroups`）。

5. **逐帧周期计算**：仿真循环每迭代重新计算 `period = 1.0 / (episode.fps * speed)`，确保 episode 切换和速度变化即时生效。
