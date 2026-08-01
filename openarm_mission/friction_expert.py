"""P4.5 weld-free scripted expert with force-limited soft-pad grasping."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from openarm_mission.config import FrictionExpertConfig
from openarm_mission.config import FrictionMissionConfig
from openarm_mission.config import FrictionTaskConfig
from openarm_mission.expert import RecoverableExpertError
from openarm_mission.expert import RelayScriptedExpert
from openarm_mission.friction_task import FrictionRelayTask
from openarm_mission.model import OpenArmMission
from openarm_mission.p3_episode import _yaw_from_quaternion
from openarm_mission.task import RelayStage


class FrictionScriptedExpert(RelayScriptedExpert):
    """Slow bimanual expert that never activates a cup weld."""

    def __init__(
        self,
        *,
        seed: int,
        output_dir: Path,
        fps: int = 20,
        width: int = 960,
        height: int = 640,
        write_video: bool = False,
        mission_config: FrictionMissionConfig | None = None,
        task_config: FrictionTaskConfig | None = None,
        expert_config: FrictionExpertConfig | None = None,
        fault_injections: dict[str, int] | None = None,
    ):
        mission = OpenArmMission(mission_config or FrictionMissionConfig())
        task = FrictionRelayTask(
            mission,
            task_config or FrictionTaskConfig(),
        )
        self.friction_expert_config = expert_config or FrictionExpertConfig()
        super().__init__(
            seed=seed,
            output_dir=output_dir,
            fps=fps,
            width=width,
            height=height,
            write_video=write_video,
            config=self.friction_expert_config,
            fault_injections=fault_injections,
            mission=mission,
            task=task,
            artifact_prefix=f"p45_friction_seed{seed:03d}",
            visual_title="OpenArm v1 · P4.5 纯摩擦双臂接力",
            visual_subtitle=("柔性高摩擦指垫 → 力限位阻抗 → 静置/抬升滑移检测 → 无 weld"),
        )
        self._apply_randomized_gripper_force()

    @property
    def friction_task(self) -> FrictionRelayTask:
        return self.task  # type: ignore[return-value]

    def _apply_randomized_gripper_force(self) -> None:
        for side in ("left", "right"):
            self.controller.set_gripper_force_limit(
                side,
                self.friction_task.gripper_force_limit_n,
            )

    def _move_tcp_linear(
        self,
        side: str,
        target_position: np.ndarray,
        seconds: float,
        *,
        segments: int = 7,
    ) -> None:
        """Follow short Cartesian waypoints near the cup."""
        start_position, _ = self.mission.tcp_pose(side)
        target_position = np.asarray(target_position, dtype=np.float64)
        for segment in range(1, segments + 1):
            fraction = segment / segments
            waypoint = (1.0 - fraction) * start_position + fraction * target_position
            self._move_tcp(
                side,
                waypoint,
                seconds / segments,
            )

    def _reset_for_episode_attempt(self) -> None:
        super()._reset_for_episode_attempt()
        self._apply_randomized_gripper_force()

    def _approach_and_attach(self, side: str) -> float:
        offsets = self.expert_config.grasp_xy_offsets
        z_offsets = self.expert_config.grasp_z_offsets
        last_reason = "unknown_friction_grasp_failure"
        for grasp_attempt in range(
            1,
            self.expert_config.max_grasp_attempts_per_arm + 1,
        ):
            variant_index = (grasp_attempt - 1 + self._episode_attempt) % min(len(offsets), len(z_offsets))
            xy_offset = np.asarray(offsets[variant_index])
            z_offset = z_offsets[variant_index]
            self._grasp_side = side
            try:
                cup_position = self.mission.cup_position()
                pregrasp = np.array(
                    [
                        cup_position[0] + xy_offset[0],
                        cup_position[1] + xy_offset[1],
                        self.expert_config.pregrasp_height,
                    ]
                )
                self._move_tcp(side, pregrasp, 0.8)

                cup_position = self.mission.cup_position()
                grasp_tcp = np.array(
                    [
                        cup_position[0] + xy_offset[0],
                        cup_position[1] + xy_offset[1],
                        cup_position[2] + self.task.config.physical_grasp_tcp_offset_z + z_offset,
                    ]
                )
                self._move_tcp_linear(side, grasp_tcp, 0.7)
                self._set_gripper(
                    side,
                    self.expert_config.closed_gripper_target,
                    0.65,
                )
                self._hold(self.friction_expert_config.contact_settle_seconds)

                fault_key = f"{side}_grasp"
                if self._consume_fault(fault_key):
                    raise RecoverableExpertError(f"injected_{side}_friction_grasp_rejection")
                attached, details = self.friction_task.try_attach(side)
                if not attached:
                    raise RecoverableExpertError(f"friction_gate_rejected: {details}")
                self._set_phase(
                    f"{side.upper()} 静态夹持验证",
                    "保持夹持力并确认双指接触与相对位姿稳定",
                    (91, 214, 147),
                )
                self._hold(self.friction_task.friction_config.static_grasp_hold_seconds)
                if self.task.done:
                    raise RuntimeError(f"{side} static friction grasp failed: {self.task.failure_reason}")
                self.recovery_events.append(
                    {
                        "time": round(self.task.elapsed, 6),
                        "episode_attempt": self._episode_attempt,
                        "side": side,
                        "grasp_attempt": grasp_attempt,
                        "stage": self.task.stage.value,
                        "reason": None,
                        "recovery_action": "friction_grasp_succeeded",
                    }
                )
                return _yaw_from_quaternion(self.mission.cup_quaternion())
            except (RecoverableExpertError, RuntimeError) as error:
                last_reason = str(error)
                if self.task.done:
                    raise
                self._recover_grasp(
                    side,
                    grasp_attempt,
                    last_reason,
                )
            finally:
                self._grasp_side = None
        raise RuntimeError(
            f"{side} friction grasp exhausted {self.expert_config.max_grasp_attempts_per_arm} attempts: {last_reason}"
        )

    def _lift_welded(
        self,
        side: str,
        cup_xy: tuple[float, float],
        cup_yaw: float,
    ) -> float:
        cup_target = np.array(
            [
                cup_xy[0],
                cup_xy[1],
                self.mission.config.table_top_z + 0.20,
            ]
        )
        qpos, yaw = self._find_upright_target(side, cup_target, cup_yaw)
        self._move_joints(
            side,
            qpos,
            self.friction_expert_config.lift_seconds,
        )
        self._hold(self.friction_task.friction_config.lifted_grasp_hold_seconds)
        expected = {
            "right": RelayStage.RIGHT_LIFTED,
            "left": RelayStage.LEFT_LIFTED,
        }[side]
        if self.task.stage is not expected:
            raise RuntimeError(f"{side} friction lift not stable: {self.task.summary()}")
        return yaw

    def _place_welded(
        self,
        side: str,
        target_xy: tuple[float, float],
        cup_yaw: float,
    ) -> float:
        high_target = np.array(
            [
                target_xy[0],
                target_xy[1],
                self.mission.config.table_top_z + 0.20,
            ]
        )
        high_qpos, high_yaw = self._find_upright_target(
            side,
            high_target,
            cup_yaw,
        )
        self._move_joints(
            side,
            high_qpos,
            self.friction_expert_config.transport_seconds,
        )

        place_target = np.array(
            [
                target_xy[0],
                target_xy[1],
                self.mission.config.table_top_z
                + self.mission.config.cup_half_height
                + self.friction_task.friction_config.release_hover_clearance,
            ]
        )
        current_tcp_position, _ = self.mission.tcp_pose(side)
        current_cup_position = self.mission.cup_position()
        place_tcp_position = current_tcp_position + place_target - current_cup_position
        self._move_tcp_linear(
            side,
            place_tcp_position,
            self.friction_expert_config.descend_seconds,
        )
        self._hold(self.friction_expert_config.place_settle_seconds)
        return high_yaw

    def _release_and_retreat(
        self,
        side: str,
        retreat_xy: tuple[float, float],
    ) -> None:
        prepared, details = self.friction_task.prepare_release(side)
        if not prepared:
            raise RuntimeError(f"{side} friction release rejected: {details}")
        self._set_gripper(
            side,
            self.controller.config.gripper_open_position,
            0.5,
        )
        self._hold(0.45)
        self._move_tcp(
            side,
            np.array(
                [
                    retreat_xy[0],
                    retreat_xy[1],
                    self.expert_config.retreat_height,
                ]
            ),
            0.85,
        )
        self._move_joints(side, self.home_qpos[side], 0.75)
        self.recovery_events.append(
            {
                "time": round(self.task.elapsed, 6),
                "episode_attempt": self._episode_attempt,
                "side": side,
                "grasp_attempt": None,
                "stage": self.task.stage.value,
                "reason": None,
                "recovery_action": "friction_release_verified_and_retreated",
            }
        )

    def _run_success(self) -> None:
        red = self.mission.config.region_a_center
        center = self.mission.config.handoff_center
        blue = self.mission.config.region_b_center

        self._set_phase(
            "纯摩擦任务初始化",
            "随机化纸杯、指垫摩擦与夹持力上限",
            (242, 83, 89),
        )
        self._hold(0.55)

        self._set_phase(
            "右手力控夹取",
            "软指垫双侧接触 · 不启用任何 weld",
            (242, 83, 89),
        )
        right_yaw = self._approach_and_attach("right")
        right_yaw = self._lift_welded("right", red, right_yaw)

        self._set_phase(
            "纯摩擦搬运至中央",
            "低加速度搬运 · 低空张开后由重力落桌",
            (232, 190, 72),
        )
        self._place_welded("right", center, right_yaw)
        self._release_and_retreat("right", center)
        self._hold(0.45)
        if self.task.stage is not RelayStage.WAIT_LEFT_GRASP:
            raise RuntimeError(f"Friction handoff did not stabilize: {self.task.summary()}")

        self._set_phase(
            "左手力控接力",
            "中央再次验证双指接触、夹持力与静态滑移",
            (99, 183, 255),
        )
        left_yaw = self._approach_and_attach("left")
        left_yaw = self._lift_welded("left", center, left_yaw)

        self._set_phase(
            "纯摩擦搬运至蓝区",
            "保持软指垫夹持 · 低空张开后物理落桌",
            (71, 111, 255),
        )
        self._place_welded("left", blue, left_yaw)
        self._release_and_retreat("left", blue)

        self._set_phase(
            "无约束成功判定",
            "蓝区、直立、桌面接触、双手退出、weld 始终关闭",
            (91, 214, 147),
        )
        self._hold(0.85)
        if not self.task.success:
            raise RuntimeError(f"P4.5 friction episode did not succeed: {self.task.summary()}")

    def run_expert(self, *, write_summary: bool = True) -> dict[str, Any]:
        summary = super().run_expert(write_summary=False)
        summary.update(
            {
                "p45_grasp_mode": "pure_friction_no_weld",
                "friction_mission_config": asdict(self.mission.config),
                "friction_task_config": asdict(self.friction_task.friction_config),
            }
        )
        if write_summary:
            self.summary_path.write_text(
                json.dumps(
                    summary,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P4.5 weld-free OpenArm friction expert.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p45"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--inject-right-grasp-failure", action="store_true")
    parser.add_argument("--inject-left-grasp-failure", action="store_true")
    args = parser.parse_args()

    expert = FrictionScriptedExpert(
        seed=args.seed,
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        write_video=args.video,
        fault_injections={
            "right_grasp": int(args.inject_right_grasp_failure),
            "left_grasp": int(args.inject_left_grasp_failure),
        },
    )
    summary = expert.run_expert()
    print(
        f"P4.5 friction expert: success={summary['expert_success']}, "
        f"attempts={summary['episode_attempts']}, "
        f"recoveries={summary['recovery_count']}, "
        f"failure={summary['failure_reason']}"
    )
    print(f"summary: {expert.summary_path}")
    if args.video:
        print(f"video: {expert.video_path}")
    if not summary["expert_success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
