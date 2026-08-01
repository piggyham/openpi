"""P3 task state, contact-gated weld grasping and terminal predicates."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import mujoco
import numpy as np

from openarm_mission.config import RelayTaskConfig
from openarm_mission.model import OpenArmMission


class RelayStage(str, Enum):
    """Ordered states of the right-to-left paper-cup relay."""

    WAIT_RIGHT_GRASP = "wait_right_grasp"
    RIGHT_ATTACHED = "right_attached"
    RIGHT_LIFTED = "right_lifted"
    WAIT_CENTER_STABLE = "wait_center_stable"
    WAIT_LEFT_GRASP = "wait_left_grasp"
    LEFT_ATTACHED = "left_attached"
    LEFT_LIFTED = "left_lifted"
    WAIT_GOAL_STABLE = "wait_goal_stable"
    SUCCESS = "success"
    FAILURE = "failure"


@dataclass(frozen=True)
class TaskEvent:
    """One timestamped task transition or terminal event."""

    time: float
    stage: str
    event: str
    details: dict[str, Any]


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    return np.array(
        [
            quaternion[0],
            -quaternion[1],
            -quaternion[2],
            -quaternion[3],
        ],
        dtype=np.float64,
    )


def _quaternion_angle(quaternion: np.ndarray) -> float:
    quaternion = quaternion / max(np.linalg.norm(quaternion), 1e-12)
    return 2.0 * float(np.arccos(np.clip(abs(quaternion[0]), 0.0, 1.0)))


class OpenArmRelayTask:
    """P3 evaluator and grasp manager around an :class:`OpenArmMission`.

    Grasping is hybrid: bilateral finger contact and a closed-gripper capture
    volume gate activation of a MuJoCo weld equality. Once active, the cup is
    entirely advanced by MuJoCo constraints; its free-joint pose is never
    overwritten frame by frame.
    """

    TERMINAL_STAGES: ClassVar[set[RelayStage]] = {
        RelayStage.SUCCESS,
        RelayStage.FAILURE,
    }

    def __init__(
        self,
        mission: OpenArmMission,
        config: RelayTaskConfig | None = None,
    ):
        self.mission = mission
        self.model = mission.model
        self.data = mission.data
        self.config = config or RelayTaskConfig()
        self._finger_geom_ids = {
            side: {
                finger: {
                    geom_id
                    for geom_id in (
                        self._id(
                            mujoco.mjtObj.mjOBJ_GEOM,
                            f"openarm_{side}_{finger}_finger_collision",
                        ),
                        int(self.mission.finger_pad_geom_ids[side][finger]),
                    )
                    if geom_id >= 0
                }
                for finger in ("left", "right")
            }
            for side in ("left", "right")
        }
        self._table_geom_id = self._id(
            mujoco.mjtObj.mjOBJ_GEOM,
            "mission_table_top",
        )
        self._cup_geom_ids = {
            geom_id
            for geom_id in range(self.model.ngeom)
            if (
                (
                    mujoco.mj_id2name(
                        self.model,
                        mujoco.mjtObj.mjOBJ_GEOM,
                        geom_id,
                    )
                    or ""
                ).startswith("mission_cup_")
            )
        }
        self._default_cup_mass = float(self.model.body_mass[self.mission.cup_body_id])
        self._default_cup_inertia = np.array(
            self.model.body_inertia[self.mission.cup_body_id],
            copy=True,
        )
        self._default_cup_friction = np.array(
            self.model.geom_friction[self.mission.cup_geom_id],
            copy=True,
        )

        self.stage = RelayStage.WAIT_RIGHT_GRASP
        self.failure_reason: str | None = None
        self.events: list[TaskEvent] = []
        self._stage_started_at = 0.0
        self._stage_durations: dict[str, float] = {}
        self._condition_started_at: float | None = None
        self._seed = 0
        self._randomization: dict[str, Any] = {}
        self._start_time = 0.0
        self.reset(seed=0)

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(object_id)

    @property
    def elapsed(self) -> float:
        return float(self.data.time - self._start_time)

    @property
    def done(self) -> bool:
        return self.stage in self.TERMINAL_STAGES

    @property
    def success(self) -> bool:
        return self.stage is RelayStage.SUCCESS

    def _active_array(self):
        if hasattr(self.data, "eq_active"):
            return self.data.eq_active
        return self.model.eq_active

    def weld_active(self, side: str) -> bool:
        return bool(self._active_array()[self.mission.cup_weld_ids[side]])

    def _set_weld_active(self, side: str, *, active: bool) -> None:
        self._active_array()[self.mission.cup_weld_ids[side]] = int(active)

    def _record(self, event: str, **details: Any) -> None:
        self.events.append(
            TaskEvent(
                time=round(self.elapsed, 6),
                stage=self.stage.value,
                event=event,
                details=details,
            )
        )

    def _transition(self, stage: RelayStage, event: str, **details: Any) -> None:
        now = self.elapsed
        self._stage_durations[self.stage.value] = (
            self._stage_durations.get(self.stage.value, 0.0) + now - self._stage_started_at
        )
        previous = self.stage
        self.stage = stage
        self._stage_started_at = now
        self._condition_started_at = None
        self._record(event, previous_stage=previous.value, **details)

    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Reset and randomize cup pose, yaw, mass and friction."""
        self._seed = 0 if seed is None else int(seed)
        rng = np.random.default_rng(self._seed)
        self.mission.reset()
        for side in ("left", "right"):
            self._set_weld_active(side, active=False)

        radius = self.config.initial_xy_randomization_radius * np.sqrt(rng.random())
        angle = rng.uniform(-np.pi, np.pi)
        xy_offset = radius * np.array([np.cos(angle), np.sin(angle)])
        yaw = rng.uniform(*self.config.initial_yaw_range)
        mass_scale = rng.uniform(*self.config.cup_mass_scale_range)
        friction = rng.uniform(*self.config.cup_friction_range)

        cup_position = np.asarray(self.mission.config.cup_initial_position).copy()
        cup_position[:2] += xy_offset
        cup_quaternion = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        address = self.mission.cup_qpos_address
        self.data.qpos[address : address + 7] = np.concatenate([cup_position, cup_quaternion])
        self.data.qvel[self.mission.cup_dof_address : self.mission.cup_dof_address + 6] = 0.0

        self.model.body_mass[self.mission.cup_body_id] = self._default_cup_mass * mass_scale
        self.model.body_inertia[self.mission.cup_body_id] = self._default_cup_inertia * mass_scale
        self.model.geom_friction[self.mission.cup_geom_id] = self._default_cup_friction
        self.model.geom_friction[self.mission.cup_geom_id, 0] = friction
        mujoco.mj_forward(self.model, self.data)

        self.stage = RelayStage.WAIT_RIGHT_GRASP
        self.failure_reason = None
        self.events = []
        self._stage_started_at = 0.0
        self._stage_durations = {}
        self._condition_started_at = None
        self._start_time = float(self.data.time)
        self._randomization = {
            "seed": self._seed,
            "cup_xy_offset_m": xy_offset.round(6).tolist(),
            "cup_yaw_rad": round(float(yaw), 6),
            "cup_mass_kg": round(
                float(self.model.body_mass[self.mission.cup_body_id]),
                6,
            ),
            "cup_slide_friction": round(float(friction), 6),
        }
        self._record("reset", **self._randomization)
        return dict(self._randomization)

    def _contact_pairs(self) -> list[tuple[int, int, float]]:
        return [
            (
                int(self.data.contact[index].geom1),
                int(self.data.contact[index].geom2),
                float(self.data.contact[index].dist),
            )
            for index in range(self.data.ncon)
        ]

    def finger_contacts(self, side: str) -> set[str]:
        """Return logical finger names currently touching the cup."""
        contacts: set[str] = set()
        for geom1, geom2, _ in self._contact_pairs():
            pair = {geom1, geom2}
            if not pair.intersection(self._cup_geom_ids):
                continue
            for finger, finger_geom_ids in self._finger_geom_ids[side].items():
                if pair.intersection(finger_geom_ids):
                    contacts.add(finger)
        return contacts

    def table_contact(self) -> bool:
        for geom1, geom2, _ in self._contact_pairs():
            pair = {geom1, geom2}
            if self._table_geom_id in pair and pair.intersection(self._cup_geom_ids):
                return True
        return False

    def gripper_opening(self, side: str) -> float:
        arm = self.mission.arms[side]
        return float(np.mean(self.data.qpos[arm.finger_qpos_indices]))

    def cup_upright_angle_deg(self) -> float:
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, self.mission.cup_quaternion())
        local_z = rotation.reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(local_z[2], -1.0, 1.0))))

    def _relative_cup_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        hand_body_id = self._id(
            mujoco.mjtObj.mjOBJ_BODY,
            f"openarm_{side}_hand_tcp",
        )
        hand_position = np.asarray(self.data.xpos[hand_body_id])
        hand_rotation = np.asarray(self.data.xmat[hand_body_id]).reshape(3, 3)
        hand_quaternion = np.asarray(self.data.xquat[hand_body_id])
        cup_position = np.asarray(self.data.xpos[self.mission.cup_body_id])
        cup_quaternion = np.asarray(self.data.xquat[self.mission.cup_body_id])
        relative_position = hand_rotation.T @ (cup_position - hand_position)
        relative_quaternion = np.empty(4)
        mujoco.mju_mulQuat(
            relative_quaternion,
            _quaternion_inverse(hand_quaternion),
            cup_quaternion,
        )
        return relative_position, relative_quaternion

    def grasp_gate(self, side: str) -> tuple[bool, dict[str, Any]]:
        expected_stage = {
            "right": RelayStage.WAIT_RIGHT_GRASP,
            "left": RelayStage.WAIT_LEFT_GRASP,
        }[side]
        tcp_position, _ = self.mission.tcp_pose(side)
        cup_position = self.mission.cup_position()
        horizontal_distance = float(np.linalg.norm(tcp_position[:2] - cup_position[:2]))
        vertical_offset = float(tcp_position[2] - cup_position[2])
        contacts = self.finger_contacts(side)
        opening = self.gripper_opening(side)
        details = {
            "expected_stage": expected_stage.value,
            "actual_stage": self.stage.value,
            "finger_contacts": sorted(contacts),
            "gripper_opening_m": round(opening, 6),
            "horizontal_distance_m": round(horizontal_distance, 6),
            "vertical_offset_m": round(vertical_offset, 6),
        }
        allowed = (
            self.stage is expected_stage
            and not self.done
            and len(contacts) == 2
            and opening <= self.config.grasp_opening_max
            and horizontal_distance <= self.config.grasp_horizontal_tolerance
            and self.config.grasp_vertical_offset_range[0]
            <= vertical_offset
            <= self.config.grasp_vertical_offset_range[1]
            and self.cup_upright_angle_deg() <= self.config.upright_limit_deg
        )
        return allowed, details

    def try_attach(self, side: str) -> tuple[bool, dict[str, Any]]:
        """Activate a weld only after the bilateral contact gate passes."""
        mujoco.mj_forward(self.model, self.data)
        allowed, details = self.grasp_gate(side)
        if not allowed:
            self._record("grasp_rejected", side=side, **details)
            return False, details

        relative_position, relative_quaternion = self._relative_cup_pose(side)
        equality_id = self.mission.cup_weld_ids[side]
        self.model.eq_data[equality_id, 3:6] = relative_position
        self.model.eq_data[equality_id, 6:10] = relative_quaternion
        for other_side in ("left", "right"):
            self._set_weld_active(
                other_side,
                active=other_side == side,
            )
        mujoco.mj_forward(self.model, self.data)

        next_stage = RelayStage.RIGHT_ATTACHED if side == "right" else RelayStage.LEFT_ATTACHED
        self._transition(
            next_stage,
            "grasp_attached",
            side=side,
            relative_position_m=relative_position.round(6).tolist(),
            relative_quaternion=relative_quaternion.round(6).tolist(),
            **details,
        )
        return True, details

    def _inside_region(self, xy: tuple[float, float], radius: float) -> bool:
        return bool(np.linalg.norm(self.mission.cup_position()[:2] - np.asarray(xy)) <= radius)

    def _stable(self) -> bool:
        linear_velocity, angular_velocity = self.mission.cup_velocity()
        return bool(
            np.linalg.norm(linear_velocity) <= self.config.stable_linear_speed
            and np.linalg.norm(angular_velocity) <= self.config.stable_angular_speed
        )

    def release(self, side: str) -> tuple[bool, dict[str, Any]]:
        """Release only in the correct region, upright and with open fingers."""
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
            "gripper_opening_m": round(self.gripper_opening(side), 6),
            "cup_stable": self._stable(),
        }
        allowed = (
            self.stage in expected_stages
            and self.weld_active(side)
            and details["inside_target"]
            and details["upright_angle_deg"] <= self.config.upright_limit_deg
            and details["gripper_opening_m"] >= self.config.release_opening_min
            and details["cup_stable"]
        )
        if not allowed:
            self._record("release_rejected", **details)
            return False, details

        self._set_weld_active(side, active=False)
        mujoco.mj_forward(self.model, self.data)
        next_stage = RelayStage.WAIT_CENTER_STABLE if side == "right" else RelayStage.WAIT_GOAL_STABLE
        self._transition(next_stage, "released", **details)
        return True, details

    def _weld_error(self, side: str) -> tuple[float, float]:
        relative_position, relative_quaternion = self._relative_cup_pose(side)
        equality_data = self.model.eq_data[self.mission.cup_weld_ids[side]]
        position_error = float(np.linalg.norm(relative_position - equality_data[3:6]))
        quaternion_error = np.empty(4)
        mujoco.mju_mulQuat(
            quaternion_error,
            _quaternion_inverse(equality_data[6:10]),
            relative_quaternion,
        )
        return position_error, float(np.degrees(_quaternion_angle(quaternion_error)))

    def upright_tcp_pose(
        self,
        side: str,
        cup_position: np.ndarray,
        cup_yaw: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the TCP pose that makes a welded cup upright at a target."""
        if not self.weld_active(side):
            raise RuntimeError(f"{side} cup weld must be active")
        equality_data = self.model.eq_data[self.mission.cup_weld_ids[side]]
        relative_position = np.asarray(equality_data[3:6])
        relative_quaternion = np.asarray(equality_data[6:10])
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

    def fail(self, reason: str, **details: Any) -> None:
        if self.done:
            return
        self.failure_reason = reason
        for side in ("left", "right"):
            self._set_weld_active(side, active=False)
        self._transition(RelayStage.FAILURE, "failure", reason=reason, **details)

    def inject_grasp_loss(self, side: str) -> None:
        """Fault-injection hook used to produce a deterministic failure video."""
        if self.weld_active(side):
            self._set_weld_active(side, active=False)
            self._record("injected_grasp_loss", side=side)

    def _hold_condition(self, *, condition: bool, seconds: float) -> bool:
        if not condition:
            self._condition_started_at = None
            return False
        if self._condition_started_at is None:
            self._condition_started_at = self.elapsed
        return self.elapsed - self._condition_started_at >= seconds

    def update(self) -> RelayStage:
        """Advance predicates after physics stepping and return the new stage."""
        if self.done:
            return self.stage
        if not (np.all(np.isfinite(self.data.qpos)) and np.all(np.isfinite(self.data.qvel))):
            self.fail("non_finite_state")
            return self.stage
        if self.elapsed >= self.config.timeout_seconds:
            self.fail("timeout", timeout_seconds=self.config.timeout_seconds)
            return self.stage
        active_welds = [side for side in ("left", "right") if self.weld_active(side)]
        if len(active_welds) > 1:
            self.fail("both_welds_active")
            return self.stage
        if self.mission.cup_position()[2] < self.mission.config.table_top_z - self.config.drop_below_table_margin:
            self.fail("cup_dropped_below_table")
            return self.stage
        if not active_welds and self.cup_upright_angle_deg() > self.config.max_ungrasped_tilt_deg:
            self.fail(
                "cup_excessively_tilted",
                upright_angle_deg=round(self.cup_upright_angle_deg(), 6),
            )
            return self.stage

        attached_side = {
            RelayStage.RIGHT_ATTACHED: "right",
            RelayStage.RIGHT_LIFTED: "right",
            RelayStage.LEFT_ATTACHED: "left",
            RelayStage.LEFT_LIFTED: "left",
        }.get(self.stage)
        if attached_side is not None:
            if not self.weld_active(attached_side):
                self.fail(f"{attached_side}_grasp_lost")
                return self.stage
            position_error, angle_error = self._weld_error(attached_side)
            if (
                position_error > self.config.slip_position_tolerance
                or angle_error > self.config.slip_angle_tolerance_deg
            ):
                self.fail(
                    f"{attached_side}_grasp_slipped",
                    weld_position_error_m=round(position_error, 6),
                    weld_angle_error_deg=round(angle_error, 6),
                )
                return self.stage
            cup_bottom_z = self.mission.cup_position()[2] - self.mission.config.cup_half_height
            if (
                self.stage is RelayStage.RIGHT_ATTACHED
                and cup_bottom_z >= self.mission.config.table_top_z + self.config.lift_clearance
            ):
                self._transition(
                    RelayStage.RIGHT_LIFTED,
                    "right_lift_confirmed",
                )
            elif (
                self.stage is RelayStage.LEFT_ATTACHED
                and cup_bottom_z >= self.mission.config.table_top_z + self.config.lift_clearance
            ):
                self._transition(
                    RelayStage.LEFT_LIFTED,
                    "left_lift_confirmed",
                )

        elif self.stage is RelayStage.WAIT_CENTER_STABLE:
            condition = (
                not active_welds
                and self._inside_region(
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
                not active_welds
                and self._inside_region(
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
        durations = dict(self._stage_durations)
        durations[self.stage.value] = durations.get(self.stage.value, 0.0) + self.elapsed - self._stage_started_at
        linear_velocity, angular_velocity = self.mission.cup_velocity()
        return {
            "success": self.success,
            "done": self.done,
            "stage": self.stage.value,
            "failure_reason": self.failure_reason,
            "elapsed_seconds": round(self.elapsed, 6),
            "randomization": dict(self._randomization),
            "cup_position_m": self.mission.cup_position().round(6).tolist(),
            "cup_upright_angle_deg": round(self.cup_upright_angle_deg(), 6),
            "cup_linear_speed_mps": round(float(np.linalg.norm(linear_velocity)), 6),
            "cup_angular_speed_radps": round(
                float(np.linalg.norm(angular_velocity)),
                6,
            ),
            "table_contact": self.table_contact(),
            "active_weld": next(
                (side for side in ("left", "right") if self.weld_active(side)),
                None,
            ),
            "stage_durations_seconds": {key: round(value, 6) for key, value in durations.items()},
            "events": [asdict(event) for event in self.events],
        }
