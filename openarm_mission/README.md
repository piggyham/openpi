# openarm_mission

OpenArm v1 双臂纸杯接力任务的独立实现目录，当前完成 P0～P5：

1. 右臂从机器人右前方的红色 A 区夹起无把手一次性纸杯。
2. 右臂把杯子直立放到桌面中央交接位并退出。
3. 左臂从中央重新夹起纸杯。
4. 左臂把杯子直立放到机器人左前方的蓝色 B 区。

P0～P2 包含需求、官方模型、任务场景、14 维双臂控制接口、IK、力矩控制和
安全限制。P3 增加双指接触门控、动态 weld 抓取约束、物理释放、任务状态机、
随机化、成功/失败判定、LIBERO 风格 BDDL 以及带指标的视频记录。P4 增加
双臂脚本专家、备选预抓取、后撤重试、双臂互锁、碰撞监控和 100 次正式评测。
P4.5 进一步提供不使用 weld 的软指垫纯摩擦抓取、力限位阻抗、滑移检测和
独立 100 次物理评测。P5 增加同步轨迹采集、视觉域随机化、确定性数据划分、
LeRobot v2 转换和数据回放。

## 目录结构

```text
openarm_mission/
├── config.py                   # 场景尺寸、区域位置和控制参数
├── model.py                    # OpenArm v1 MJCF 组合、纸杯、桌面和相机
├── controller.py               # 双臂 DLS IK、力矩 PD 和安全限制
├── demo.py                     # 右手→中央→左手→蓝区动态演示
├── task.py                     # P3 接触门控、weld、状态机和成功判定
├── p3_episode.py               # P3 success/failure episode 与视频导出
├── expert.py                   # P4 可恢复双臂脚本专家
├── p4_benchmark.py             # P4 多进程评测、CSV/JSON 和可视化面板
├── friction_task.py            # P4.5 无 weld 接触力/滑移状态机
├── friction_expert.py          # P4.5 软指垫纯摩擦双臂专家
├── p45_benchmark.py            # P4.5 纯摩擦多进程评测和面板
├── dataset.py                  # P5 数据定义、同步记录和域随机化
├── collect_dataset.py          # P5 批量采集与对齐报告
├── convert_to_lerobot.py       # P5 LeRobot v2 转换
├── replay_dataset.py           # P5 三视图轨迹回放
├── bddl/                       # LIBERO 风格任务定义
├── smoke_test.py               # 模型、IK、物理和离屏渲染检查
├── tests/                      # P0～P5 自动测试
├── dependencies/              # 官方模型 revision 锁定信息
├── fetch_openarm_v1.sh         # 幂等依赖下载脚本
├── SPEC.md                     # 冻结需求和验收口径
├── TODO.md                     # 完整 Todo List
└── artifacts/                  # 生成成果；默认不纳入 Git
```

## 准备官方模型

```bash
bash openarm_mission/fetch_openarm_v1.sh
```

官方 `openarm_mujoco` 依赖固定在 revision：

```text
8955afb54e4adfb59a236e2b4d15192b7a02865c
```

## 生成动态展示

Linux 无窗口环境使用 EGL：

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.demo
```

默认生成：

```text
openarm_mission/artifacts/openarm_paper_cup_relay.mp4
openarm_mission/artifacts/openarm_paper_cup_relay.gif
openarm_mission/artifacts/openarm_paper_cup_relay_storyboard.png
openarm_mission/artifacts/openarm_paper_cup_relay.json
```

快速预览：

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.demo \
  --duration-scale 0.35 --width 720 --height 480 --no-gif
```

这段 P0～P2 展示中，机械臂使用真实 MuJoCo 动力学与力矩控制；纸杯使用确定性
抓取锁存跟随 TCP，以保证跨 MuJoCo 版本稳定复现。它是任务流程与控制基础设施
演示，不代表已经训练好的物理抓取策略。P3 已提供接触门控物理任务；自动失败
恢复和 100 次成功率评测列在 `TODO.md` 的 P4。

## 运行 P3 物理任务

同时生成一条成功 episode 和一条抓取丢失失败 episode：

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.p3_episode \
  --mode both --seed 7
```

输出目录：

```text
openarm_mission/artifacts/p3/p3_success_seed007.mp4
openarm_mission/artifacts/p3/p3_success_seed007.json
openarm_mission/artifacts/p3/p3_failure_seed007.mp4
openarm_mission/artifacts/p3/p3_failure_seed007.json
```

不渲染视频，只快速执行状态机：

```bash
.venv/bin/python -m openarm_mission.p3_episode \
  --mode success --seed 7 --no-video
```

P3 与 P2 展示不同：P3 只有在两侧手指都实际接触纸杯、夹爪闭合且杯子进入
捕获空间时，才会激活对应手的 MuJoCo weld。激活后不再逐帧覆盖纸杯 free-joint
位姿；放置时关闭 weld，由桌面接触承载纸杯。

## 运行 P4 可恢复脚本专家

运行普通 P4 episode：

```bash
.venv/bin/python -m openarm_mission.expert --seed 7
```

生成一条左右手首次抓取都被拒绝、随后分别后撤重试成功的动态演示：

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.expert \
  --seed 7 --video \
  --inject-right-grasp-failure \
  --inject-left-grasp-failure
```

P4 专家对左右手各提供 3 个预抓取变体和最多 3 次局部抓取尝试。接触门控拒绝
后，当前手会张开夹爪、抬升、稳定并改用下一组偏移；局部恢复耗尽或发生运动
异常时，最多允许 2 次整局执行。状态机互锁要求右手完成中央释放并退出后，
左手才可进入交接区域。意外的机器人—桌面、双臂互撞和非手指杯体碰撞均会
被检测和记录。

恢复演示输出：

```text
openarm_mission/artifacts/p4/p4_expert_seed007.mp4
openarm_mission/artifacts/p4/p4_expert_seed007.json
openarm_mission/artifacts/p4/p4_expert_recovery_storyboard.png
```

## 运行 P4 正式评测

```bash
.venv/bin/python -m openarm_mission.p4_benchmark \
  --episodes 100 --workers 4
```

种子 0～99 的正式结果为 `100/100` 成功，超过 `≥95%` 验收线；其中自动后撤
重试 5 次、意外碰撞 0 次。平均终点 XY 误差 `9.7 mm`，最大终点倾角
`0.034°`。评测会输出逐 episode CSV、完整 JSON 和 PNG 可视化面板：

```text
openarm_mission/artifacts/p4/p4_benchmark_100.csv
openarm_mission/artifacts/p4/p4_benchmark_100.json
openarm_mission/artifacts/p4/p4_benchmark_100.png
```

## 运行 P4.5 纯摩擦专家

P4.5 使用独立场景配置，原始 P3/P4 模型和结果不变。它禁用官方刚性指面碰撞，
改用绿色软指垫；夹爪保持位置阻抗形式，但将双指执行器力限制随机化为
`8～12 N`。抓取确认要求双指接触和最小夹持力成立，随后分别执行 `0.3 s`
静态保持和抬升保持，并持续检查接触丢失及杯子相对夹爪的位姿滑移。

```bash
.venv/bin/python -m openarm_mission.friction_expert --seed 7
```

生成动态视频：

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.friction_expert \
  --seed 7 --video --width 720 --height 480
```

OpenArm v1 在原桌面高度无法让左手到达中央的低位侧壁抓取位，因此 P4.5 专用
配置将桌面上表面从 `0.28 m` 抬高至 `0.33 m`。放置时杯底降至桌面上方约
`50 mm`，随后张开夹爪，由重力、摩擦和桌面碰撞完成落桌；状态机继续验证
直立、桌面接触、双手退出和连续 `0.5 s` 稳定保持。

运行 100 次纯摩擦评测：

```bash
.venv/bin/python -m openarm_mission.p45_benchmark \
  --episodes 100 --workers 4
```

种子 0～99 的结果为 `100/100` 成功、weld 违规 `0`、意外碰撞 `0`。平均终点
XY 误差 `28.3 mm`，最大最终倾角 `8.13°`。输出包括：

```text
openarm_mission/artifacts/p45/p45_friction_seed007.mp4
openarm_mission/artifacts/p45/p45_friction_seed007.json
openarm_mission/artifacts/p45/p45_friction_storyboard.png
openarm_mission/artifacts/p45/p45_friction_benchmark_100.csv
openarm_mission/artifacts/p45/p45_friction_benchmark_100.json
openarm_mission/artifacts/p45/p45_friction_benchmark_100.png
```

## 测试与静态渲染

```bash
.venv/bin/python -m unittest discover -s openarm_mission/tests -v
examples/libero/.venv/bin/python -m unittest discover -s openarm_mission/tests -v
```

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.smoke_test \
  --render-path openarm_mission/artifacts/paper_cup_scene.png
```

当前自动验证覆盖 MuJoCo 2.3.7 和 3.2.3，包括 25 项测试、40 个随机近邻
全位姿 IK、任务区域位置 IK、接触门控、跨版本 weld、0.5 秒成功保持、双臂
互锁、抓取恢复、软指垫和无 weld 完整接力。P3 完整物理流程已验证随机种子
0～9，结果为 10/10 成功；P4 与 P4.5 均已验证随机种子 0～99，结果分别为
100/100 成功。

## P5 轨迹数据采集与 LeRobot 转换

P5 以 `20 Hz` 记录纯摩擦脚本专家。每个控制帧包含 16 维双臂关节/夹爪状态、
14 维双臂笛卡尔增量动作、纸杯位姿、MuJoCo 仿真时间和任务阶段。动作顺序为：

```text
[left dx dy dz dRx dRy dRz gripper,
 right dx dy dz dRx dRy dRz gripper]
```

平移单位为米，旋转采用旋转向量且单位为弧度；夹爪 `-1` 表示张开、`+1`
表示闭合。对齐语义为 `observation[t] -> action[t] -> target[t+1]`。
采集会随机化杯子位置/偏航、质量、摩擦、指垫摩擦、夹持力、光照、材质亮度
以及相机位置和视场角。

正式采集的 200 条成功轨迹均包含严格同帧的前视、左腕和右腕 RGB；其中前
20 条作为图像—状态—动作—时间戳的显式对齐验收集。

```bash
MUJOCO_GL=egl .venv/bin/python -m openarm_mission.collect_dataset \
  --episodes 200 --workers 16 --image-episodes 200
```

长时间采集可增加 `--resume --max-new-episodes 32` 分批恢复。

种子按 `seed % 10` 确定性划分：余数 `0` 为 test、`1` 为 validation、其余为
train，200 条数据固定得到 `160/20/20`。原始数据和报告位于：

```text
openarm_mission/artifacts/p5/raw/
openarm_mission/artifacts/p5/schema.json
openarm_mission/artifacts/p5/manifest.json
openarm_mission/artifacts/p5/alignment_20.json
openarm_mission/artifacts/p5/splits.json
```

转换全部 200 条严格同步三相机轨迹：

```bash
.venv/bin/python -m openarm_mission.convert_to_lerobot
```

转换器保留 train/validation/test 边界，并在
`meta/openarm_source_map.json` 记录 LeRobot episode、源种子和 SHA-256
校验和的对应关系。回放命令：

```bash
.venv/bin/python -m openarm_mission.replay_dataset \
  openarm_mission/artifacts/p5/raw/episode_seed000000.npz \
  --output openarm_mission/artifacts/p5/replay_seed000000.mp4
```

正式结果：种子 `0～199` 为 `200/200` 成功，共 `86,400` 个三相机控制帧；
train/validation/test 为 `160/20/20`。转换后得到
`200 episodes / 86,400 frames` 的 LeRobot v2 数据集，重新加载检查通过。
