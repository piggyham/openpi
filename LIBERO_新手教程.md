# π0.5-LIBERO 新手教程：从零跑通"看图→动作→机械臂动起来"的闭环

> 本教程记录的是**真实跑通过**的完整流程（不是抄官方文档）。每一步都附讲解，告诉你"这一步在干什么、为什么需要它"。
>
> 适合：第一次接触 openpi / 机器人策略 / LIBERO 仿真的同学。

---

## 0. 先搞懂我们在做什么（概念篇）

在看命令之前，先花 3 分钟理解几张图，否则后面全是"知其然不知其所以然"。

### 0.1 这几个名词分别是什么

| 名词 | 一句话解释 |
|---|---|
| **openpi** | Physical Intelligence 开源的机器人策略框架。它能"加载模型、接收图像、输出动作"。 |
| **π0.5 (pi05)** | 一个具体的视觉-语言-动作大模型（VLA）。看图 + 听指令 → 输出机械臂该怎么动。 |
| **LIBERO** | 一个机械臂仿真基准。里面有 Franka 机械臂、桌子、碗、盘子等，用 MuJoCo 物理引擎模拟。带语言任务（如"把碗放到盘子上"）和成功判断。 |
| **MuJoCo** | 物理仿真引擎，负责让机械臂"真的动"、物体"真的掉"。 |
| **checkpoint** | 训练好的模型参数文件（π0.5-LIBERO 这个约 11.6GB）。有了它模型才有"脑子"。 |

### 0.2 整体架构：为什么要分"服务器"和"客户端"

整个系统分成两个独立程序，**各开一个终端**：

```
┌─────────────────────┐         网络(websocket)        ┌──────────────────────────┐
│   策略服务器 server   │  <────────────────────────>  │   仿真客户端 client        │
│                     │                               │                          │
│  · 加载 π0.5 模型     │   ① client 发: 当前画面+指令     │  · 跑 LIBERO 仿真环境      │
│  · 接收图像+指令      │   ─────────────────────────>  │  · 拍下机械臂当前画面       │
│  · 输出机械臂动作     │   ② server 回: 一串动作(10步)   │  · 把动作喂给机械臂执行     │
│                     │   <─────────────────────────  │  · 判断是否成功、录视频     │
│  uv run serve_policy │                               │  python main.py           │
└─────────────────────┘                               └──────────────────────────┘
        (终端 1)                                              (终端 2)
```

**为什么分开？**
- server 跑模型推理，吃 GPU、需要新版本 Python(3.11) 和重依赖(jax/torch)。
- client 跑仿真环境，需要**老版本 Python(3.8)** 和 MuJoCo/robosuite 那套老依赖。
- 两套依赖**装不到同一个 Python 环境里**（版本冲突），所以拆成两个进程，用网络通信。
- 这也意味着：server 只要有一次，client 可以反复重启换任务，模型不用重新加载。

> 💡 类比：server 是"大脑"（看图想动作），client 是"身体+考场"（机械臂执行、打分）。两者用websocket对话。

### 0.3 一次"回合(episode)"里发生了什么

```
重置环境 → 等10步让物体落稳 → 循环(最多220步):
   ├─ 拍当前画面 (agentview + 手腕摄像头)
   ├─ 如果上一批动作用完了 → 把画面+指令发给server → 拿回10步动作
   ├─ 取出下一步动作 → 喂给机械臂执行 → 环境变化
   └─ 检查是否完成任务(done) → 是则记一次"成功"，跳出
最后 → 把这一回合的画面存成 MP4视频
```

理解了这个循环，你就懂了整个项目在干嘛。

---

## 1. 环境前提检查（开工前先确认）

打开一个终端，逐条粘贴运行，确认每项都 OK 再继续。

```bash
# 1) 确认在项目目录
cd ~/aaayuanxiang/openPi/openpi

# 2) 确认有 NVIDIA 显卡（π0.5 需要 GPU）
nvidia-smi
# 期望: 看到 RTX 4090 之类的卡，显存够用

# 3) 确认 git 子模块已初始化（LIBERO 代码在 third_party/libero）
git submodule status
# 期望: third_party/libero 和 third_party/aloha 前面没有减号"-"

# 4) 确认 openpi 主环境可用
.venv/bin/python -c "import openpi; print('openpi OK')"
# 期望: 打印 openpi OK

# 5) 如果你在国内: 确认代理通（下 checkpoint/依赖要用）
curl -s https://github.com -o /dev/null -w "%{http_code}\n"
# 期望: 200。如果显示 000 或超时，说明代理挂了，先修代理再继续。
```

**讲解：**
- 第2步：π0.5 是个大模型，没 GPU 跑不动（或极慢）。
- 第3步：LIBERO 的代码是作为 git 子模块引入的，必须 `git submodule update --init --recursive` 拉下来，否则后面 import libero 会报找不到。
- 第5步：checkpoint（11.6GB）要从谷歌云盘(GCS)下，依赖要从 GitHub/PyPI 下。国内网络通常需要代理。

---

## 2. 第一步：启动策略服务器（终端 1）

这一步加载 π0.5 模型，让它待命、准备接收图像。

```bash
cd ~/aaayuanxiang/openPi/openpi

# (国内) 让 server 走代理下载 checkpoint
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ALL_PROXY=socks://127.0.0.1:7897

# 启动！
uv run scripts/serve_policy.py --env LIBERO
```

**会发生什么 / 要等多久：**
- 首次运行会自动下载 π0.5-LIBERO 的 checkpoint（**约 11.6GB**，下到 `~/.cache/openpi/`）。网速 8MB/s 的话约 25 分钟。**之后再启动就不用下了**（已缓存）。
- 下载完后加载模型进 GPU。
- 看到 `server listening on 0.0.0.0:8000` 这行，就说明 **server 就绪了**。
- ⚠️ 这个终端要**一直开着**，别关。关了模型就没了。

**命令拆解：**
- `uv run` = 用 uv（一个超快的 Python 包管理器）在 openpi 的 `.venv` 里跑 Python 脚本。
- `scripts/serve_policy.py` = 启动策略服务器的入口脚本。
- `--env LIBERO` = 告诉它用 LIBERO 配置。**关键**：这个配置默认就是加载 `pi05_libero` 这个 finetune 好的 checkpoint，专门适合 LIBERO 任务。不用手动指定模型路径。

> 💡 怎么验证 server 好了？看终端最后有没有 `server listening on 0.0.0.0:8000`。也可以另开终端 `curl 0.0.0.0:8000` 看有没有响应。

---

## 3. 第二步：准备仿真客户端环境（终端 2）

client 需要一个**独立的 Python 3.8 环境**（和 server 的 3.11 隔离）。这一步一次性做好，以后都能复用。

### 3.1 装 Python 3.8 解释器

```bash
cd ~/aaayuanxiang/openPi/openpi
export HTTPS_PROXY=http://127.0.0.1:7897   # 国内走代理

# 让 uv 下载一个独立的 Python 3.8（不影响系统的 python）
uv python install 3.8
# 期望最后一行: + cpython-3.8.20 ... (python3.8)
```

**讲解：** uv 能给你装"绿色版"的 Python，不污染系统。LIBERO 这套老代码要求 Python 3.8，所以单独下一个。

### 3.2 创建虚拟环境 + 装依赖

```bash
# 用 3.8 建一个专属虚拟环境（放在 examples/libero/.venv）
uv venv --python 3.8 examples/libero/.venv

# 激活它（之后这个终端里的 python/pip 都指向这个环境）
source examples/libero/.venv/bin/activate

# 安装 LIBERO 需要的所有依赖（一次性，约几分钟）
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  packages/openpi-client/pyproject.toml \
  --extra-index-url https://download.pytorch.org/whl/cu113 \
  --index-strategy=unsafe-best-match
```

**命令拆解：**
- `uv venv` = 建虚拟环境（一个隔离的"小房间"，里面装的包不影响别人）。
- `source .../activate` = 走进这个小房间。激活后命令行最前面会出现 `(.venv)` 字样。
- `uv pip sync` = 按需求文件**精确**安装指定版本的包。
  - `examples/libero/requirements.txt` = openpi 给 LIBERO 客户端定的依赖清单。
  - `third_party/libero/requirements.txt` = LIBERO 自己的依赖（robosuite、mujoco 等）。
  - `packages/openpi-client/pyproject.toml` = client 要用的 openpi-client 库。
  - `--extra-index-url .../cu113` = 额外的包源（专门下 CUDA 11.3 版的 PyTorch，老版本需要）。
  - `--index-strategy=unsafe-best-match` = 让 uv 在多个源之间灵活匹配（老依赖常有版本陷阱）。

### 3.3 装两个"可编辑"的本地包

```bash
# openpi-client（client 用来和 server 通信的库）
uv pip install -e packages/openpi-client

# LIBERO 本体
uv pip install -e third_party/libero
```

**讲解：** `-e` 表示"可编辑安装"——直接链接到源代码目录，而不是复制一份。好处是你改了源码立刻生效。

### 3.4 写 LIBERO 的配置文件（避免交互卡住）

LIBERO 第一次 import 时会弹出一个 `input()` 交互提示问你数据路径，在脚本里会卡住报错。**预先写好配置文件就能绕过**：

```bash
# 用项目真实路径生成配置（直接整段粘贴运行）
LIBERO_ROOT="$PWD/third_party/libero/libero/libero"
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
benchmark_root: $LIBERO_ROOT
bddl_files: $LIBERO_ROOT/bddl_files
init_states: $LIBERO_ROOT/init_files
datasets: $PWD/third_party/libero/libero/datasets
assets: $LIBERO_ROOT/assets
EOF
```

**讲解：** 这个 `config.yaml` 告诉 LIBERO 去哪里找任务定义文件（bddl）、初始状态、模型资产等。`~/.libero/` 是 LIBERO 默认的配置目录。

### 3.5 验证客户端环境 OK

```bash
# 每次新开终端都要先做这三件事（设 PYTHONPATH）
export PYTHONPATH=$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PWD

# 验证关键库都能导入
python -c "from libero.libero import benchmark; print('libero OK, 套件:', list(benchmark.get_benchmark_dict().keys()))"
python -c "from openpi_client import websocket_client_policy; print('openpi-client OK')"
python -c "import mujoco, robosuite; print('mujoco', mujoco.__version__, 'OK')"
# 三个都 OK 就过关
```

**讲解：** `PYTHONPATH` 告诉 Python "除了 site-packages，也去这些目录找代码"。这里要包含 LIBERO、openpi-client、项目根目录的源码。

---

## 4. 第三步：运行闭环评估（终端 2，确保 server 已就绪）

先回到终端 1 确认看到 `server listening on 0.0.0.0:8000`，然后在终端 2 运行：

```bash
cd ~/aaayuanxiang/openPi/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PWD

# ⚠️ 关键: 用 glx 渲染！egl 会在跑多个环境时崩溃
export MUJOCO_GL=glx

# 先小规模验证: libero_spatial, 每个任务跑 2 回合
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 2
```

**会发生什么：**
- 终端会打印每个任务的语言指令（如 `"pick up the black bowl next to the plate and place it on the plate"`）。
- 每个 episode 结束打印 `Success: True/False` 和累计成功率。
- 大约 1.5 分钟跑完 10 个任务 × 2 回合 = 20 回合。
- 视频生成到 `data/libero/videos/`。

**命令拆解：**
- `examples/libero/main.py` = 评估脚本（核心循环就在这里）。
- `--args.task-suite-name libero_spatial` = 选哪个任务套件。
  - `libero_spatial`（最简单，官方~98.8%）
  - `libero_object` / `libero_goal`（中等）
  - `libero_10`（最难，官方~92.4%）
- `--args.num-trials-per-task 2` = 每个任务跑几次（官方口径是 50；这里先用 2 快速验证）。
- ⚠️ 注意参数前缀是 `--args.`！因为脚本用了 `tyro.cli(eval_libero)`，参数包在 `Args` 类里，CLI 里要带类名前缀。**这是最容易踩的坑**。

---

## 5. 看结果

```bash
# 看视频文件
ls data/libero/videos/
# 文件名格式: rollout_<任务描述>_success.mp4 或 _failure.mp4

# 看成功率统计（终端里已经打印了，也可以这样数）
echo "成功: $(ls data/libero/videos/*success.mp4 2>/dev/null | wc -l) 个"
echo "失败: $(ls data/libero/videos/*failure.mp4 2>/dev/null | wc -l) 个"
```

**看视频：** 把 `.mp4` 文件拷到本地或用播放器打开，能看到 Franka 机械臂根据语言指令去抓碗、放到盘子上。`success` 结尾的是成功的，`failure` 的是失败的（可以对比学习模型哪里没做好）。

---

## 6. 进阶玩法

### 6.1 跑官方完整口径（50 回合/任务）

```bash
python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 50
# 约 35 分钟，得到可与官方论文对比的成功率
```

### 6.2 换更难的任务套件

```bash
python examples/libero/main.py --args.task-suite-name libero_10 --args.num-trials-per-task 50
```

### 6.3 短程微调（用自己的数据训一个 π0.5）

```bash
# 1) 把 LIBERO 数据转成训练格式
uv run examples/libero/convert_libero_data_to_lerobot.py

# 2) 用 pi05_libero 配置训练（把 num_train_steps 调小做短程微调）
uv run scripts/train.py pi05_libero

# 3) 用自己训出来的 checkpoint 评估
uv run scripts/serve_policy.py --env LIBERO \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ./checkpoints/pi05_libero/<你的步数>
```

---

## 7. 常见问题（踩坑记录）

### Q1: `EOFError: EOF when reading a line`
LIBERO 的交互提示。→ 做了 **3.4**（写 `~/.libero/config.yaml`）就不会出现。

### Q2: `EGLError: EGL_NOT_INITIALIZED`
MuJoCo 用 egl 渲染跑多个环境会崩。→ **必须 `export MUJOCO_GL=glx`**（见第4步）。

### Q3: 参数报 `Unrecognized options`
参数没带 `args.` 前缀。→ 用 `--args.task-suite-name`、`--args.num-trials-per-task`。

### Q4: server 下载 checkpoint 卡住/失败
网络问题。→ 确认代理通（`curl -s https://github.com -o /dev/null -w "%{http_code}"` 返回 200），并设了 `HTTPS_PROXY`。

### Q5: `ModuleNotFoundError: No module named 'libero'` 或 `'openpi_client'`
PYTHONPATH 没设。→ 每次新开终端都要 `export PYTHONPATH=$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PWD`，且要先 `source examples/libero/.venv/bin/activate`。

### Q6: Docker 构建一直超时
国内 Docker Hub DNS 污染 + GitHub 阻断。→ 走本文档的**原生路线**（终端1 server + 终端2 client），不用 Docker。

---

## 8. 每日速查（环境已建好后，平时只要这两步）

```bash
# ===== 终端 1: 起 server（如果没起）=====
cd ~/aaayuanxiang/openPi/openpi
uv run scripts/serve_policy.py --env LIBERO

# ===== 终端 2: 跑评估 =====
cd ~/aaayuanxiang/openPi/openpi
source examples/libero/.venv/bin/activate
export PYTHONPATH=$PWD/third_party/libero:$PWD/packages/openpi-client/src:$PWD
export MUJOCO_GL=glx
python examples/libero/main.py --args.task-suite-name libero_spatial --args.num-trials-per-task 2
```

---

## 附：本项目已验证的实测结果

- **libero_spatial**，20 回合，**成功率 95%（19/20）**
- 与官方基准（π0.5 libero_spatial ~98.8%@50回合）一致 ✅
- 环境：RTX 4090 24GB / Ubuntu / Python 3.11(server) + 3.8(client)
```
