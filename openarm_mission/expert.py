"""P4 recovery-capable scripted expert for the OpenArm relay task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from openarm_mission.config import ScriptedExpertConfig
from openarm_mission.model import OpenArmMission
from openarm_mission.p3_episode import P3EpisodeRunner
from openarm_mission.p3_episode import _yaw_from_quaternion
from openarm_mission.task import OpenArmRelayTask


class RecoverableExpertError(RuntimeError):
    """A local motion or grasp failure that permits a retract-and-retry."""


class RelayScriptedExpert(P3EpisodeRunner):
    """P4 expert with grasp variants, contact monitoring and recovery."""

    def __init__(
        self,
        *,
        seed: int,
        output_dir: Path,
        fps: int = 20,
        width: int = 960,
        height: int = 640,
        write_video: bool = False,
        config: ScriptedExpertConfig | None = None,
        fault_injections: dict[str, int] | None = None,
        mission: OpenArmMission | None = None,
        task: OpenArmRelayTask | None = None,
        artifact_prefix: str | None = None,
        visual_title: str = "OpenArm v1 · P4 可恢复双臂脚本专家",
        visual_subtitle: str = ("接触门控 → 失败后撤重试 → 双臂互锁 → 蓝区稳定放置"),
    ):
        super().__init__(
            mode="success",
            seed=seed,
            output_dir=output_dir,
            fps=fps,
            width=width,
            height=height,
            write_video=write_video,
            artifact_prefix=artifact_prefix or f"p4_expert_seed{seed:03d}",
            visual_title=visual_title,
            visual_subtitle=visual_subtitle,
            mission=mission,
            task=task,
        )
        self.expert_config = config or ScriptedExpertConfig()
        self.fault_injections = dict(fault_injections or {})
        self.recovery_events: list[dict[str, Any]] = []
        self.collision_events: list[dict[str, Any]] = []
        self._episode_attempt = 0
        self._grasp_side: str | None = None
        self._suppress_collision_check = False

    def _geom_name(self, geom_id: int) -> str:
        return (
            mujoco.mj_id2name(
                self.mission.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            or f"geom_{geom_id}"
        )

    def unexpected_contacts(self) -> list[dict[str, Any]]:
        """Return robot-table, cross-arm and non-finger cup collisions."""
        unexpected: list[dict[str, Any]] = []
        for index in range(self.mission.data.ncon):
            contact = self.mission.data.contact[index]
            if contact.dist > 0.001:
                continue
            names = (
                self._geom_name(int(contact.geom1)),
                self._geom_name(int(contact.geom2)),
            )
            joined = " ".join(names)
            has_cup = "mission_cup_" in joined
            has_table = "mission_table_top" in joined
            has_robot = "openarm_" in joined
            has_finger = "finger_collision" in joined or "finger_pad" in joined

            allowed = (
                (has_cup and has_table)
                or (has_cup and has_finger)
                or (
                    all("openarm_" in name for name in names)
                    and not (
                        any("openarm_left_" in name for name in names)
                        and any("openarm_right_" in name for name in names)
                    )
                )
            )
            hazardous = (
                (has_robot and has_table)
                or (has_cup and has_robot and not has_finger)
                or (any("openarm_left_" in name for name in names) and any("openarm_right_" in name for name in names))
            )
            if hazardous and not allowed:
                unexpected.append(
                    {
                        "geom1": names[0],
                        "geom2": names[1],
                        "distance_m": round(float(contact.dist), 6),
                    }
                )
        return unexpected

    def _step_frame(self) -> None:
        super()._step_frame()
        if self._suppress_collision_check or self.task.done:
            return
        contacts = self.unexpected_contacts()
        if contacts:
            event = {
                "time": round(self.task.elapsed, 6),
                "episode_attempt": self._episode_attempt,
                "stage": self.task.stage.value,
                "contacts": contacts,
            }
            self.collision_events.append(event)
            raise RecoverableExpertError(f"unexpected_collision: {contacts}")

    def _record_recovery(
        self,
        *,
        side: str,
        grasp_attempt: int,
        reason: str,
        action: str,
    ) -> None:
        self.recovery_events.append(
            {
                "time": round(self.task.elapsed, 6),
                "episode_attempt": self._episode_attempt,
                "side": side,
                "grasp_attempt": grasp_attempt,
                "stage": self.task.stage.value,
                "reason": reason,
                "recovery_action": action,
            }
        )

    def _consume_fault(self, key: str) -> bool:
        remaining = self.fault_injections.get(key, 0)
        if remaining <= 0:
            return False
        self.fault_injections[key] = remaining - 1
        return True

    def _recover_grasp(self, side: str, grasp_attempt: int, reason: str) -> None:
        self._record_recovery(
            side=side,
            grasp_attempt=grasp_attempt,
            reason=reason,
            action="open_gripper_retract_and_retry",
        )
        self._set_phase(
            f"{side.upper()} 抓取恢复",
            "张开夹爪、抬升后使用备选预抓取位姿",
            (245, 156, 66),
        )
        self._suppress_collision_check = True
        try:
            self._set_gripper(
                side,
                self.controller.config.gripper_open_position,
                0.30,
            )
            cup_position = self.mission.cup_position()
            try:
                self._move_tcp(
                    side,
                    np.array(
                        [
                            cup_position[0],
                            cup_position[1],
                            self.expert_config.retreat_height,
                        ]
                    ),
                    self.expert_config.recovery_retract_seconds,
                )
            except RuntimeError:
                self._move_joints(
                    side,
                    self.home_qpos[side],
                    self.expert_config.recovery_retract_seconds,
                )
            self._hold(self.expert_config.recovery_settle_seconds)
        finally:
            self._suppress_collision_check = False

    def _approach_and_attach(self, side: str) -> float:
        offsets = self.expert_config.grasp_xy_offsets
        z_offsets = self.expert_config.grasp_z_offsets
        last_reason = "unknown_grasp_failure"
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
                self._move_tcp(side, grasp_tcp, 0.7)
                self._set_gripper(
                    side,
                    self.expert_config.closed_gripper_target,
                    0.65,
                )

                fault_key = f"{side}_grasp"
                if self._consume_fault(fault_key):
                    raise RecoverableExpertError(f"injected_{side}_grasp_rejection")
                attached, details = self.task.try_attach(side)
                if not attached:
                    raise RecoverableExpertError(f"contact_gate_rejected: {details}")
                self.recovery_events.append(
                    {
                        "time": round(self.task.elapsed, 6),
                        "episode_attempt": self._episode_attempt,
                        "side": side,
                        "grasp_attempt": grasp_attempt,
                        "stage": self.task.stage.value,
                        "reason": None,
                        "recovery_action": "grasp_succeeded",
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
            f"{side} grasp exhausted {self.expert_config.max_grasp_attempts_per_arm} attempts: {last_reason}"
        )

    def _release_and_retreat(
        self,
        side: str,
        retreat_xy: tuple[float, float],
    ) -> None:
        super()._release_and_retreat(side, retreat_xy)
        self.recovery_events.append(
            {
                "time": round(self.task.elapsed, 6),
                "episode_attempt": self._episode_attempt,
                "side": side,
                "grasp_attempt": None,
                "stage": self.task.stage.value,
                "reason": None,
                "recovery_action": "release_verified_and_retreated",
            }
        )

    def _reset_for_episode_attempt(self) -> None:
        self.task.reset(self.seed)
        self.controller.reset_targets()
        self._grasp_side = None
        self._suppress_collision_check = False

    def _close_media(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.renderer is not None:
            close = getattr(self.renderer, "close", None)
            if close is not None:
                close()
            else:
                gl_context = getattr(self.renderer, "_gl_context", None)
                if gl_context is not None:
                    gl_context.free()
            self.renderer = None

    def run_expert(self, *, write_summary: bool = True) -> dict[str, Any]:
        """Run with whole-episode restart as the final recovery layer."""
        attempt_summaries: list[dict[str, Any]] = []
        final_exception: str | None = None
        try:
            for episode_attempt in range(self.expert_config.max_episode_attempts):
                self._episode_attempt = episode_attempt
                if episode_attempt > 0:
                    self._reset_for_episode_attempt()
                try:
                    self._run_success()
                except Exception as error:
                    final_exception = str(error)
                    if not self.task.done:
                        self.task.fail(
                            "expert_execution_error",
                            error=final_exception,
                        )
                    attempt = self.task.summary()
                    attempt["expert_exception"] = final_exception
                    attempt_summaries.append(attempt)
                    self.recovery_events.append(
                        {
                            "time": round(self.task.elapsed, 6),
                            "episode_attempt": episode_attempt,
                            "side": None,
                            "grasp_attempt": None,
                            "stage": self.task.stage.value,
                            "reason": final_exception,
                            "recovery_action": (
                                "reset_episode"
                                if episode_attempt + 1 < self.expert_config.max_episode_attempts
                                else "terminate"
                            ),
                        }
                    )
                    continue
                attempt_summaries.append(self.task.summary())
                final_exception = None
                break
        finally:
            self._close_media()

        summary = self.task.summary()
        blue = np.asarray(self.mission.config.region_b_center)
        summary.update(
            {
                "expert_success": self.task.success,
                "expert_seed": self.seed,
                "episode_attempts": len(attempt_summaries),
                "grasp_attempts": {
                    side: sum(
                        event["side"] == side
                        and event["grasp_attempt"] is not None
                        and event["recovery_action"]
                        in {
                            "open_gripper_retract_and_retry",
                            "grasp_succeeded",
                            "friction_grasp_succeeded",
                        }
                        for event in self.recovery_events
                    )
                    for side in ("left", "right")
                },
                "recovery_count": sum(
                    event["recovery_action"]
                    not in {
                        "grasp_succeeded",
                        "friction_grasp_succeeded",
                        "release_verified_and_retreated",
                        "friction_release_verified_and_retreated",
                    }
                    for event in self.recovery_events
                ),
                "collision_count": len(self.collision_events),
                "final_xy_error_m": round(
                    float(np.linalg.norm(self.mission.cup_position()[:2] - blue)),
                    6,
                ),
                "expert_exception": final_exception,
                "expert_config": asdict(self.expert_config),
                "recovery_events": self.recovery_events,
                "collision_events": self.collision_events,
                "attempt_summaries": attempt_summaries,
                "frames": self._frame_index,
                "video": str(self.video_path) if self.write_video else None,
            }
        )
        if write_summary:
            self.summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the P4 recovery-capable OpenArm scripted expert.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p4"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--inject-right-grasp-failure", action="store_true")
    parser.add_argument("--inject-left-grasp-failure", action="store_true")
    args = parser.parse_args()

    fault_injections = {
        "right_grasp": int(args.inject_right_grasp_failure),
        "left_grasp": int(args.inject_left_grasp_failure),
    }
    expert = RelayScriptedExpert(
        seed=args.seed,
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        write_video=args.video,
        fault_injections=fault_injections,
    )
    summary = expert.run_expert()
    print(
        f"expert: success={summary['expert_success']}, "
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
