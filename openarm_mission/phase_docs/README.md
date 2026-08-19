# OpenArm v1 双臂纸杯接力任务 · 阶段讲解文档总索引

本目录把 OpenArm 双臂纸杯接力任务（`openarm_mission/`）从 P0 到 P8 的每一步
拆成独立文档，逐一讲解 **目标、关键文件、详细步骤、参数与验收口径**。读完后
你应当能够：复现场景与控制流程、理解接触物理任务的状态机、读懂脚本专家与纯摩
擦专家的差异、清楚数据采集与 LeRobot 转换的语义，并掌握后续接入 openpi 训练、
闭环评测和真机准备的计划。

## 任务一句话定义

> 右臂从机器人右前方红色 A 区夹起无把手一次性纸杯，直立放到桌面中央交接位并退
> 出；左臂从中央重新夹起纸杯，再直立放到机器人左前方蓝色 B 区。

动作顺序固定为：**右手 A 区取杯 → 中央直立放杯 → 右手退出 → 左手中央取杯 →
B 区直立放杯 → 左手退出**。

## 阶段总览

| 阶段 | 主题 | 状态 | 文档 |
|------|------|------|------|
| P0 | 冻结需求、坐标、14 维动作接口、安全约束 | ✅ 已完成 | [p0.md](p0.md) |
| P1 | 导入 OpenArm v1 模型与任务场景 | ✅ 已完成 | [p1.md](p1.md) |
| P2 | 双臂控制器、安全限制与动态展示 | ✅ 已完成 | [p2.md](p2.md) |
| P3 | 真实接触任务与成功判定（weld 接触门控） | ✅ 已完成 | [p3.md](p3.md) |
| P4 | 双臂可恢复脚本专家 + 100 次评测 | ✅ 已完成 | [p4.md](p4.md) |
| P4.5 | 无 weld 纯摩擦抓取 + 100 次评测 | ✅ 已完成 | [p4_5.md](p4_5.md) |
| P5 | 轨迹采集与 LeRobot v2 转换 | ✅ 已完成 | [p5.md](p5.md) |
| P6 | 接入 openpi（Inputs/Outputs、DataConfig、训练配置） | ⏳ 计划 | [p6.md](p6.md) |
| P7 | π0.5 LoRA 微调与闭环评测 | ⏳ 计划 | [p7.md](p7.md) |
| P8 | 可复现性与真机准备 | ⏳ 计划 | [p8.md](p8.md) |

> P4.5 是 P4 的“物理抓取替代路线”，不改变 P3/P4 的场景与结果，故单列一篇。

## 阅读建议

- **想快速跑起来**：P0 → P1 → P2，然后按 `openarm_mission/README.md` 的命令生成
  动态展示视频。
- **想理解物理任务**：P3 是核心，P4 / P4.5 是两套可评测的抓取实现，P5 在此之上
  做数据采集。
- **想训练策略**：P5 产出数据，P6 把数据和环境接入 openpi，P7 微调并评测，P8
  面向真机落地。

## 配套文件速查

- 需求与口径：`openarm_mission/SPEC.md`、`openarm_mission/TODO.md`、`openarm_mission/README.md`
- 配置：`openarm_mission/config.py`
- 模型/控制器/任务/专家/数据：`openarm_mission/{model,controller,task,expert,friction_expert,dataset,collect_dataset,convert_to_lerobot,replay_dataset}.py`
- 单元测试：`openarm_mission/tests/`
- 生成成果（默认不入 Git）：`openarm_mission/artifacts/`

## 环境约定

- 主环境 MuJoCo 2.3.7，LIBERO 客户端环境 MuJoCo 3.2.3，两者均通过 25 项自动测试。
- 无窗口环境用 EGL 离屏渲染：`MUJOCO_GL=egl ...`。
- 官方 `openarm_mujoco` 依赖固定在 revision
  `8955afb54e4adfb59a236e2b4d15192b7a02865c`，用
  `bash openarm_mission/fetch_openarm_v1.sh` 幂等下载。
