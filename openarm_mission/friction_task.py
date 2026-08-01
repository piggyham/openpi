"""P4.5 weld-free relay task with contact-force and slip validation."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from openarm_mission.config import FrictionTaskConfig
from openarm_mission.model import OpenArmMission
from openarm_mission.task import OpenArmRelayTask
from openarm_mission.task import RelayStage
from openarm_mission.task import _quaternion_angle
from openarm_mission.task import _quaternion_inverse


class FrictionRelayTask(OpenArmRelayTask):
    """Relay state machine whose grasp is maintained only by pad friction."""

    def __init__(
        self,
        mission: OpenArmMission,
        config: FrictionTaskConfig | None = None,
    ):
        if not mission.config.soft_finger_pads:
            raise ValueError("FrictionRelayTask requires soft finger pads")
        self._grasp_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._contact_loss_started_at: float | None = None
        self._gripper_force_limit_n = 0.0
        super().__init__(mission, config or FrictionTaskConfig())

    @property
    def friction_config(self) -> FrictionTaskConfig:
        return self.config  # type: ignore[return-value]

    @property
    def gripper_force_limit_n(self) -> float:
        return self._gripper_force_limit_n

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset base randomization and independently randomize pad physics."""
        super().reset(seed)
        self._grasp_reference.clear()
        self._contact_loss_started_at = None
        rng = np.random.default_rng(self._seed + 45_000)
        pad_friction = rng.uniform(*self.friction_config.pad_slide_friction_range)
        self._gripper_force_limit_n = float(rng.uniform(*self.friction_config.gripper_force_limit_range_n))
        for per_side in self.mission.finger_pad_geom_ids.values():
            for geom_id in per_side.values():
                if geom_id < 0:
                    raise RuntimeError("Soft finger pad geom is missing")
                self.model.geom_friction[geom_id, 0] = pad_friction
        self._randomization.update(
            {
                "pad_slide_friction": round(float(pad_friction), 6),
                "gripper_force_limit_n": round(
                    self._gripper_force_limit_n,
                    6,
                ),
            }
        )
        if self.events:
            self.events[0] = type(self.events[0])(
                time=self.events[0].time,
                stage=self.events[0].stage,
                event=self.events[0].event,
                details=dict(self._randomization),
            )
        return dict(self._randomization)

    def gripper_forces(self, side: str) -> np.ndarray:
        """Return absolute force from both finger impedance actuators."""
        actuator_ids = self.mission.arms[side].finger_actuator_ids
        return np.abs(np.asarray(self.data.actuator_force[actuator_ids])).copy()

    def grasp_gate(self, side: str) -> tuple[bool, dict[str, Any]]:
        allowed, details = super().grasp_gate(side)
        forces = self.gripper_forces(side)
        details.update(
            {
                "finger_forces_n": forces.round(6).tolist(),
                "minimum_finger_force_n": round(float(np.min(forces)), 6),
            }
        )
        allowed = allowed and float(np.min(forces)) >= self.friction_config.min_finger_force_n
        return allowed, details

    def try_attach(self, side: str) -> tuple[bool, dict[str, Any]]:
        """Confirm a force-bearing bilateral grasp without activating weld."""
        mujoco.mj_forward(self.model, self.data)
        allowed, details = self.grasp_gate(side)
        if not allowed:
            self._record("friction_grasp_rejected", side=side, **details)
            return False, details
        if any(self.weld_active(item) for item in ("left", "right")):
            self.fail("weld_used_in_friction_mode")
            return False, details

        relative_position, relative_quaternion = self._relative_cup_pose(side)
        self._grasp_reference[side] = (
            np.array(relative_position, copy=True),
            np.array(relative_quaternion, copy=True),
        )
        self._contact_loss_started_at = None
        next_stage = RelayStage.RIGHT_ATTACHED if side == "right" else RelayStage.LEFT_ATTACHED
        self._transition(
            next_stage,
            "friction_grasp_confirmed",
            side=side,
            relative_position_m=relative_position.round(6).tolist(),
            relative_quaternion=relative_quaternion.round(6).tolist(),
            **details,
        )
        return True, details

    def upright_tcp_pose(
        self,
        side: str,
        cup_position: np.ndarray,
        cup_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an upright target from the measured friction-grasp pose."""
        if side not in self._grasp_reference:
            raise RuntimeError(f"{side} has no confirmed friction grasp")
        relative_position, relative_quaternion = self._grasp_reference[side]
        desired_cup_quaternion = np.array([np.cos(cup_yaw / 2.0), 0.0, 0.0, np.sin(cup_yaw / 2.0)])
        desired_tcp_quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            desired_tcp_quaternion,
            desired_cup_quaternion,
            _quaternion_inverse(relative_quaternion),
        )
        tcp_rotation = np.empty(9)
        mujoco.mju_quat2Mat(tcp_rotation, desired_tcp_quaternion)
        desired_tcp_position = np.asarray(cup_position) - tcp_rotation.reshape(3, 3) @ relative_position
        return desired_tcp_position, desired_tcp_quaternion

    def _friction_slip(self, side: str) -> tuple[float, float]:
        reference_position, reference_quaternion = self._grasp_reference[side]
        relative_position, relative_quaternion = self._relative_cup_pose(side)
        position_error = float(np.linalg.norm(relative_position - reference_position))
        quaternion_error = np.empty(4)
        mujoco.mju_mulQuat(
            quaternion_error,
            _quaternion_inverse(reference_quaternion),
            relative_quaternion,
        )
        return position_error, float(np.degrees(_quaternion_angle(quaternion_error)))

    def prepare_release(self, side: str) -> tuple[bool, dict[str, Any]]:
        """Validate table support, then end grasp monitoring before opening."""
        expected_stages = {
            "right": {RelayStage.RIGHT_ATTACHED, RelayStage.RIGHT_LIFTED},
            "left": {RelayStage.LEFT_ATTACHED, RelayStage.LEFT_LIFTED},
        }[side]
        target_xy = self.mission.config.handoff_center if side == "right" else self.mission.config.region_b_center
        target_radius = self.config.handoff_radius if side == "right" else self.config.goal_radius
        details = {
            "side": side,
            "inside_target": self._inside_region(target_xy, target_radius),
            "upright_angle_deg": round(self.cup_upright_angle_deg(), 6),
            "cup_stable": self._stable(),
            "table_contact": self.table_contact(),
            "cup_bottom_clearance_m": round(
                float(
                    self.mission.cup_position()[2]
                    - self.mission.config.cup_half_height
                    - self.mission.config.table_top_z
                ),
                6,
            ),
            "finger_forces_n": self.gripper_forces(side).round(6).tolist(),
        }
        allowed = (
            self.stage in expected_stages
            and side in self._grasp_reference
            and details["inside_target"]
            and details["upright_angle_deg"] <= self.config.upright_limit_deg
            and details["cup_stable"]
            and (
                details["table_contact"]
                or details["cup_bottom_clearance_m"] <= self.friction_config.max_release_drop_height
            )
            and not any(self.weld_active(item) for item in ("left", "right"))
        )
        if not allowed:
            self._record("friction_release_rejected", **details)
            return False, details
        del self._grasp_reference[side]
        self._contact_loss_started_at = None
        next_stage = RelayStage.WAIT_CENTER_STABLE if side == "right" else RelayStage.WAIT_GOAL_STABLE
        self._transition(next_stage, "friction_release_started", **details)
        return True, details

    def update(self) -> RelayStage:
        """Advance weld-free grasp, slip, handoff and goal predicates."""
        if self.done:
            return self.stage
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            self.fail("non_finite_state")
            return self.stage
        if self.elapsed >= self.config.timeout_seconds:
            self.fail("timeout", timeout_seconds=self.config.timeout_seconds)
            return self.stage
        if any(self.weld_active(side) for side in ("left", "right")):
            self.fail("weld_used_in_friction_mode")
            return self.stage
        if self.mission.cup_position()[2] < self.mission.config.table_top_z - self.config.drop_below_table_margin:
            self.fail("cup_dropped_below_table")
            return self.stage

        attached_side = {
            RelayStage.RIGHT_ATTACHED: "right",
            RelayStage.RIGHT_LIFTED: "right",
            RelayStage.LEFT_ATTACHED: "left",
            RelayStage.LEFT_LIFTED: "left",
        }.get(self.stage)
        if attached_side is None and self.cup_upright_angle_deg() > self.config.max_ungrasped_tilt_deg:
            self.fail(
                "cup_excessively_tilted",
                upright_angle_deg=round(self.cup_upright_angle_deg(), 6),
            )
            return self.stage

        if attached_side is not None:
            contacts = self.finger_contacts(attached_side)
            forces = self.gripper_forces(attached_side)
            contact_ok = len(contacts) == 2 and float(np.min(forces)) >= 0.35 * self.friction_config.min_finger_force_n
            if contact_ok:
                self._contact_loss_started_at = None
            elif self._contact_loss_started_at is None:
                self._contact_loss_started_at = self.elapsed
            elif self.elapsed - self._contact_loss_started_at >= self.friction_config.contact_loss_grace_seconds:
                self.fail(
                    f"{attached_side}_friction_contact_lost",
                    finger_contacts=sorted(contacts),
                    finger_forces_n=forces.round(6).tolist(),
                )
                return self.stage

            position_error, angle_error = self._friction_slip(attached_side)
            if (
                position_error > self.friction_config.grasp_position_slip_tolerance
                or angle_error > self.friction_config.grasp_angle_slip_tolerance_deg
            ):
                self.fail(
                    f"{attached_side}_friction_slip",
                    relative_position_error_m=round(position_error, 6),
                    relative_angle_error_deg=round(angle_error, 6),
                )
                return self.stage

            cup_bottom_z = self.mission.cup_position()[2] - self.mission.config.cup_half_height
            if (
                self.stage is RelayStage.RIGHT_ATTACHED
                and cup_bottom_z >= self.mission.config.table_top_z + self.config.lift_clearance
            ):
                self._transition(
                    RelayStage.RIGHT_LIFTED,
                    "right_friction_lift_confirmed",
                )
            elif (
                self.stage is RelayStage.LEFT_ATTACHED
                and cup_bottom_z >= self.mission.config.table_top_z + self.config.lift_clearance
            ):
                self._transition(
                    RelayStage.LEFT_LIFTED,
                    "left_friction_lift_confirmed",
                )

        elif self.stage is RelayStage.WAIT_CENTER_STABLE:
            condition = (
                self._inside_region(
                    self.mission.config.handoff_center,
                    self.config.handoff_radius,
                )
                and self.cup_upright_angle_deg() <= self.config.upright_limit_deg
                and self._stable()
                and self.table_contact()
                and not self.finger_contacts("right")
            )
            if self._hold_condition(
                condition=condition,
                seconds=self.config.handoff_hold_seconds,
            ):
                self._transition(
                    RelayStage.WAIT_LEFT_GRASP,
                    "center_handoff_stable",
                )

        elif self.stage is RelayStage.WAIT_GOAL_STABLE:
            condition = (
                self._inside_region(
                    self.mission.config.region_b_center,
                    self.config.goal_radius,
                )
                and self.cup_upright_angle_deg() <= self.config.upright_limit_deg
                and self._stable()
                and self.table_contact()
                and not self.finger_contacts("left")
                and not self.finger_contacts("right")
            )
            if self._hold_condition(
                condition=condition,
                seconds=self.config.goal_hold_seconds,
            ):
                self._transition(RelayStage.SUCCESS, "goal_hold_complete")
        return self.stage

    def summary(self) -> dict[str, Any]:
        result = super().summary()
        result.update(
            {
                "grasp_mode": "pure_friction",
                "weld_permitted": False,
                "gripper_force_limit_n": round(
                    self._gripper_force_limit_n,
                    6,
                ),
                "finger_forces_n": {side: self.gripper_forces(side).round(6).tolist() for side in ("left", "right")},
            }
        )
        return result
