# openarm_sim — parquet 驱动的 OpenArm 姿态仿真前端

输入 16 维关节角序列(左臂 7 关节 + 左夹爪开度 + 右臂 7 关节 + 右夹爪开度),
在 MuJoCo 中驱动 OpenArm v1 仿真,并在一个浏览器页面上实时输出。数据源:

查看器及所有任务仿真在未加载 episode 或执行 reset 时，左右机械臂均以 7 关节
全零的自然下垂姿态开始。回放开始后则按 episode 中记录的状态驱动。

- **LeRobot v2 parquet**(`observation.state`,旧 P5 仿真数据集,20 Hz);
- **OpenArm 数据集格式 v0.3.0** 双布局:
  - 真机遥操作 episode(`<data-dir>/real_data/`,重采样到 30 Hz);
  - **P10 转换后的 v0.3.0 数据集根**(`--data-dir` 直接指向含
    `metadata.yaml` + `episodes/` 的目录,如下拉框以 `sim_<id>` 命名)。
- **OpenArm Panel 实时源**:订阅专用 30 Hz SSE (`Target=commanded`,
  `Actual=/joint_states pos`)并读取三路 MJPEG;OpenPI 主机不需要 ROS 2。

- **自由视角**(顶部大面板):左键拖动旋转、Shift/右键拖动平移、滚轮缩放
  (服务端 `mjv_moveCamera`,与 MuJoCo 原生查看器相同的语义),另含"重置视角"
  与"跟随纸杯"(锁定自由相机视线到水瓶)开关;
- 查看器场景与专家数据采集场景一致(见 `playback.ViewerMissionConfig`):
  **机器人 + 带桌腿的桌子 + 水瓶 + 红/蓝交接杯垫标记**。回放时水瓶随手臂
  物理互动(动态模式),"对比录制"行显示的仍是原始录制画面(含水瓶)。
- 仿真三相机画面(front / left_wrist / right_wrist,实时流);
- 全部 16 个关节角:实时数值表(target / actual,误差超限标红)+ 16 通道滚动趋势曲线;
- 默认显示“录制相机”行：与仿真画面同帧展示 parquet 内嵌的
  front / left_wrist / right_wrist 三路采集图像；可用顶部复选框隐藏。

## 运行

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server --port 8080
```

浏览器打开 `http://127.0.0.1:8080/`。默认读取 P10 刚采集的 200 条仿真数据
(转换后的 v0.3.0 数据集根 `artifacts/p10/openarm_paper_cup_relay/`,
下拉框以 `sim_0` … `sim_199` 命名,20 Hz;另纳入 `--data-dir` 下
`real_data/` 的真机 episode,以 `real_<id>` 命名)。每个下拉选项在名字后附上
该 episode 的存放地址(如 `sim_0 — <...>/openarm_paper_cup_relay/episodes/0`):

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server \
    --data-dir openarm_mission/artifacts/p10/openarm_paper_cup_relay
```

页面顶部的“数据源”区域可以在服务运行期间切换：

- **录制文件**：填写运行 OpenArmSim 的服务器上的路径，可指向单个
  `episode_*.parquet`、LeRobot 数据集目录或 OpenArm v0.3.0 数据集目录，点击
  “加载路径”后 episode 下拉框会立即更新，无需重启服务。
- **实时实物关节**：填写 OpenArm Panel SSE 地址（例如
  `http://127.0.0.1:9000/sse/sim`），点击“连接实物”。如需鉴权，启动
  OpenArmSim 前设置 `OPENARM_SIM_STREAM_TOKEN`。

浏览器的安全模型不会把本机文件选择框中的绝对路径暴露给服务器，因此录制
路径输入框表示的是**服务器文件系统路径**；远程访问页面时也应填写远程服务器
上的路径。

## ROS episode 选择接口

OpenArmSim 提供独立 ROS bridge，订阅 `std_msgs/Int32`。消息值是录制数据列表的
**零基序号**：`0` 播放第 0 个 episode，`5` 播放第 5 个；超出当前 episode
数量的值会被拒绝。ROS 节点通过本机 `GET /api/episode?index=N` 控制 OpenArmSim，
因此 ROS 与 MuJoCo/OpenPI 可以使用各自的 Python 环境。

ROS1：

```bash
source /opt/ros/noetic/setup.bash
python3 -m openarm_mission.openarm_sim.ros_episode_bridge \
  --ros-version 1 \
  --topic /openarm_sim/episode \
  --server-url http://127.0.0.1:8080

rostopic pub --once /openarm_sim/episode std_msgs/Int32 "data: 3"
```

ROS2：

```bash
source /opt/ros/humble/setup.bash
python3 -m openarm_mission.openarm_sim.ros_episode_bridge \
  --ros-version 2 \
  --topic /openarm_sim/episode \
  --server-url http://127.0.0.1:8080

ros2 topic pub --once /openarm_sim/episode std_msgs/msg/Int32 "{data: 3}"
```

也可省略 `--ros-version` 使用自动检测。建议 ROS bridge 与 OpenArmSim 运行在同一
台主机；这样控制 API 可以继续只绑定 `127.0.0.1`，不会暴露到局域网。

## 带鉴权的局域网 REST 网关

不使用 ROS 时，推荐让 OpenArmSim 继续只监听 `127.0.0.1:8080`，另启最小权限
网关监听局域网。网关只暴露 `POST /episode`，不会代理页面、WebSocket 或文件
路径加载接口。

```bash
cd /home/piggyham/aaaxuyuanxiang/openPi/openpi

.venv/bin/python -m openarm_mission.openarm_sim.lan_episode_gateway \
  --host 0.0.0.0 \
  --port 8090 \
  --openarm-url http://127.0.0.1:8080
```

首次启动会自动生成权限为 `0600` 的 `.openarm_sim_gateway_token`（已加入
`.gitignore`）。在服务器本机查看并安全地分发给获准设备：

```bash
cat .openarm_sim_gateway_token
```

局域网设备调用（`SERVER_LAN_IP` 替换为 OpenArmSim 主机的局域网 IP）：

```bash
curl -X POST "http://SERVER_LAN_IP:8090/episode" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"index": 3}'
```

消息索引从 0 开始。默认仅允许回环地址和 RFC1918 私网
（`10/8`、`172.16/12`、`192.168/16`）；可重复传入 `--allow-cidr` 收紧范围，
例如只允许实验室子网：

```bash
--allow-cidr 192.168.10.0/24
```

也可通过环境变量提供至少 32 字符的固定 Token：

```bash
export OPENARM_SIM_GATEWAY_TOKEN='replace-with-a-long-random-secret'
```

Bearer Token 在普通 HTTP 上不加密；此模式只适用于可信、隔离的实验室局域网。
跨不可信网络时应在网关前增加 HTTPS 反向代理或使用 VPN。

真机 Reality-to-Sim（共享数据集目录 + 实时镜像）:

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server \
    --scene reality \
    --data-dir /mnt/openarm_data \
    --live-url http://127.0.0.1:9000/sse/sim \
    --live-fps 30
```

`reality` 场景只包含机器人、相机和配置中的桌子；桌面及四条桌腿的
`contype/conaffinity` 均为 0，不参与碰撞。真机/实时源的三维机器人由
Actual 直接写 qpos，并叠加青色半透明 Target 虚影。虚影使用独立 `MjData`
做正运动学，只追加到渲染场景，不参与碰撞、接触或动力学；Target 无效时
自动隐藏。页面上的“taget虚影”开关可以只关闭该渲染层，Target 数值和
趋势仍会保留。
如 Panel 设置了 `OPENARM_SIM_STREAM_TOKEN`，OpenArmSim 进程使用同名环境
变量发送 Bearer token。

可**多次指定 `--data-dir`** 同时导入多个数据集(每个 v0.3.0 根 / real_data
树 / LeRobot 目录一个 `--data-dir`)。下拉框按目录加前缀以避免重名
(单目录保持 `sim_0`… 原名;多目录时每集前缀源目录标签,如
`p10_smoke_newlayout__sim_0`、`p10/openarm_paper_cup_relay__sim_0`):

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server \
    --data-dir openarm_mission/artifacts/p10_smoke_newlayout/openarm_paper_cup_relay \
    --data-dir openarm_mission/artifacts/p10/openarm_paper_cup_relay
```

单文件模式(任意符合 16 维状态列约定的 parquet):

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server \
    --parquet path/to/file.parquet --state-col observation.state --fps 20
```

离线自检(不启服务,验证物理/渲染/seek/两种模式):

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.playback
```

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--host/--port` | `0.0.0.0/8080` | 单端口同时服务 HTML(GET /)与 WebSocket |
| `--data-dir` | P10 `data/` | episode 列表来源(递归 `chunk-*` + `real_data/`;或直接指向 v0.3.0 数据集根)。**可重复指定**以同时加载多个数据集 |
| `--parquet` | 无 | 指定单个 parquet(覆盖 data-dir) |
| `--state-col` | `observation.state` | 关节角列名,必须为 16 维 |
| `--fps` | 20 | LeRobot episode 控制帧率 |
| `--real-fps` | 30 | v0.3.0 episode 重采样帧率(回放 P9 建议 20) |
| `--live-url` | 无 | Panel `/sse/sim` 地址；设置后下拉框增加 `LIVE — OpenArm Panel` |
| `--live-fps` | 30 | 实时源最大渲染频率 |
| `--scene` | `mission` | `reality` 使用无任务物体、无碰撞桌子的真机镜像场景 |
| `--width/--height` | 320/240 | 三相机渲染分辨率 |
| `--free-width/--free-height` | 640/480 | 自由视角渲染分辨率 |
| `--jpeg-quality` | 80 | 串流 JPEG 质量 |

## 两种回放模式(页面右上切换)

- **Dynamic**(默认):关节目标经既有力矩 PD 控制器(`controller.compute_ctrl`,
  含力矩限幅与超速制动)物理跟踪,展示机器人实际会怎么动;seek 时从 episode
  起点快速重仿真(不渲染,最坏约 1 s)。**夹爪手指除外**:动态模式下手指直接
  写到目标 qpos(而非经手指伺服积分),避免 pad-pad 自锁——当录制数据的
  "闭合"姿态在手指间没有物体(真机抓取位与 viewer 场景纸杯不对齐)时,
  欠阻尼的手指伺服会过冲越过闭合限位,两个椭球指垫自锁后永远无法再张开;
  手指伺服比手臂快得多,直接写目标即可在回放帧率下瞬间到位,手臂仍按 PD 跟踪。
- **Kinematic**:逐帧直写 qpos/夹爪开度,精确复现录制姿态(actual ≡ target),
  不做积分。注意:水瓶不在 16 维状态中,两种模式下水瓶均不受该序列驱动
  (dynamic 下水瓶会因手臂接触而物理移动)。

多标签页共享同一会话;任一标签页的控制(播放/暂停/seek/切换)对所有标签页生效。

## 真机 episode(real_data)

`<data-dir>/real_data/episodes/<id>/` 下的 OpenArm 数据集格式 v0.3.0
(左右臂 `obs/arms/*/state.parquet` @100 Hz + `cameras/{head,wrist_left,
wrist_right}` JPEG 流)。加载语义与随数据附带的 `openarm_dataset` 转换包一致:
公共时间区间上按 `--real-fps` 均匀重采样;Actual 线性插值,Commanded Target
使用零阶保持(不会读取未来命令),相机取时间最近样本。状态顺序调整为 viewer
的左臂在前。映射:

- 7 个关节逐关节直通(真机与 v1 模型关节零位约定一致),播放时再 clamp 到
  v1 模型限位;
- 新 Panel 数据由 metadata 声明 `gripper_encoding: opening_m`,直接使用
  `0=闭合、0.044=张开`;旧的负值归一化格式按 `0.044*clip(-raw,0,1)`
  转为相同的物理开度单位;
- 相机映射 head→front、wrist_left→left_wrist、wrist_right→right_wrist,
  "对比录制"行按时间最近邻显示真机画面。

注意:真机场景(橙色桌面、瓶装物)与 viewer 的仿真场景(木桌 + 水瓶)不同,
仿真行复现的是 MuJoCo 中的手臂与水瓶运动,真机桌面/物体以真机录像为准。

## 结构

- `playback.py` — parquet 校验加载(`EpisodeData`,录制图像懒加载)+
  `SimPlayback`(仿真/渲染/seek/JPEG)。所有 MuJoCo/EGL 对象必须在同一线程创建使用。
- `real_data.py` — 真机 episode 发现与加载(30 Hz 重采样、夹爪映射、真机
  JPEG 懒加载),产出同一个 `EpisodeData`。
- `server.py` — `websockets` 单端口;仿真线程与 asyncio 之间用命令队列 +
  latest-frame slot 衔接;慢客户端只丢帧不阻塞。
- `web/index.html` — 单文件前端(内联 CSS/JS,无构建、无 CDN)。

## 故障排查

- 启动报 EGL/OpenGL 错误:确认 `MUJOCO_GL=egl` 且有可用 GPU 驱动;服务会在
  绑定端口前快速失败并提示。
- 缺官方模型:先 `bash openarm_mission/fetch_openarm_v1.sh`。
- parquet 校验失败(缺列/维度≠16/非有限值)会在页面报错并保留上一个 episode。
- 退出时 `GLContext.__del__` 的 EGL 报错、`MUJOCO_LOG.TXT` 的 0x502 警告为已知
  无害噪声,可忽略。
- MuJoCo 2.3.7 的 `MjvCamera.lookat` 绑定存在 strides=(0,) 缺陷,代码内已用
  ctypes 直写绕过(`playback.py::_set_lookat`),无需处理。
