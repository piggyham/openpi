"""Safe bimanual Cartesian controller for the OpenArm v1 mission."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from openarm_mission.config import ControllerConfig
from openarm_mission.model import ArmModelHandles
from openarm_mission.model import OpenArmMission


class InvalidActionError(ValueError):
    """Raised when an action cannot safely enter the controller."""


@dataclass(frozen=True)
class IKResult:
    qpos: np.ndarray
    position_error: float
    orientation_error: float
    converged: bool
    iterations: int


def _normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero quaternion")
    result = quaternion / norm
    if result[0] < 0:
        result = -result
    return result


def _rotation_vector_to_quaternion(rotation_vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotation_vector))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rotation_vector / angle
    half_angle = 0.5 * angle
    return np.concatenate([[np.cos(half_angle)], axis * np.sin(half_angle)])


def _rotation_error(target_quaternion: np.ndarray, current_matrix: np.ndarray) -> np.ndarray:
    target_matrix = np.empty(9)
    mujoco.mju_quat2Mat(target_matrix, target_quaternion)
    error_matrix = target_matrix.reshape(3, 3) @ current_matrix.T
    error_quaternion = np.empty(4)
    mujoco.mju_mat2Quat(error_quaternion, error_matrix.reshape(-1))
    error_quaternion = _normalize_quaternion(error_quaternion)
    vector_norm = float(np.linalg.norm(error_quaternion[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(error_quaternion[0], -1.0, 1.0))
    return error_quaternion[1:] / vector_norm * angle


class BimanualCartesianController:
    """14D Cartesian action interface backed by DLS IK and torque PD."""

    ACTION_DIMENSION = 14

    def __init__(
        self,
        mission: OpenArmMission,
        config: ControllerConfig | None = None,
    ):
        self.mission = mission
        self.model = mission.model
        self.data = mission.data
        self.config = config or ControllerConfig()
        self._kp = np.asarray(self.config.joint_kp)
        self._kd = np.asarray(self.config.joint_kd)
        self._velocity_limit = np.asarray(self.config.max_joint_velocity)
        if self._kp.shape != (7,) or self._kd.shape != (7,) or self._velocity_limit.shape != (7,):
            raise ValueError("Joint controller arrays must contain exactly seven values")

        self.target_position: dict[str, np.ndarray] = {}
        self.target_quaternion: dict[str, np.ndarray] = {}
        self.target_joint_qpos: dict[str, np.ndarray] = {}
        self.target_gripper: dict[str, float] = {}
        self.last_ik: dict[str, IKResult] = {}
        self.reset_targets()

    def reset_targets(self) -> None:
        """Synchronize all controller targets to the current simulation state."""
        for side, arm in self.mission.arms.items():
            position, quaternion = self.mission.tcp_pose(side)
            self.target_position[side] = position
            self.target_quaternion[side] = quaternion
            self.target_joint_qpos[side] = np.array(self.data.qpos[arm.qpos_indices], copy=True)
            opening = float(np.mean(self.data.qpos[arm.finger_qpos_indices]))
            self.target_gripper[side] = opening
        self.last_ik.clear()

    def _validate_action(self, action) -> np.ndarray:
        array = np.asarray(action, dtype=np.float64)
        if array.shape != (self.ACTION_DIMENSION,):
            raise InvalidActionError(f"Expected action shape ({self.ACTION_DIMENSION},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise InvalidActionError("Action contains NaN or Inf")
        return array

    def _integrate_target(self, side: str, arm_action: np.ndarray) -> None:
        translation = np.clip(
            arm_action[:3],
            -self.config.max_translation_delta,
            self.config.max_translation_delta,
        )
        rotation = np.asarray(arm_action[3:6], dtype=np.float64)
        rotation_norm = float(np.linalg.norm(rotation))
        if rotation_norm > self.config.max_rotation_delta:
            rotation *= self.config.max_rotation_delta / rotation_norm

        self.target_position[side] = np.clip(
            self.target_position[side] + translation,
            self.config.workspace_min_array(),
            self.config.workspace_max_array(),
        )

        delta_quaternion = _rotation_vector_to_quaternion(rotation)
        composed = np.empty(4)
        mujoco.mju_mulQuat(composed, delta_quaternion, self.target_quaternion[side])
        self.target_quaternion[side] = _normalize_quaternion(composed)

        gripper_command = float(np.clip(arm_action[6], -1.0, 1.0))
        interpolation = 0.5 * (gripper_command + 1.0)
        self.target_gripper[side] = (
            1.0 - interpolation
        ) * self.config.gripper_open_position + interpolation * self.config.gripper_closed_position

    def apply_action(self, action) -> dict[str, IKResult]:
        """Apply one synchronized left/right Cartesian action."""
        action = self._validate_action(action)
        arm_actions = {"left": action[:7], "right": action[7:]}

        for side, arm_action in arm_actions.items():
            self._integrate_target(side, arm_action)

        results: dict[str, IKResult] = {}
        for side in ("left", "right"):
            result = self.solve_ik(
                side,
                self.target_position[side],
                self.target_quaternion[side],
            )
            current = np.array(
                self.data.qpos[self.mission.arms[side].qpos_indices],
                copy=True,
            )
            joint_delta = np.clip(
                result.qpos - current,
                -self.config.max_joint_target_delta,
                self.config.max_joint_target_delta,
            )
            self.target_joint_qpos[side] = current + joint_delta
            results[side] = result
        self.last_ik = results
        return results

    def _joint_bounds(self, arm: ArmModelHandles) -> tuple[np.ndarray, np.ndarray]:
        limits = np.asarray(self.model.jnt_range[arm.joint_ids])
        lower = limits[:, 0] + self.config.joint_limit_margin
        upper = limits[:, 1] - self.config.joint_limit_margin
        if np.any(lower >= upper):
            raise ValueError(f"Invalid joint bounds for {arm.side} arm")
        return lower, upper

    def solve_ik(
        self,
        side: str,
        target_position: np.ndarray,
        target_quaternion: np.ndarray | None = None,
    ) -> IKResult:
        """Solve one arm while holding the other arm at its current state.

        Passing ``None`` for ``target_quaternion`` enables position-only IK.
        This is useful for long, presentation-oriented motions where allowing
        the wrist to turn produces a much larger reachable workspace.
        """
        arm = self.mission.arms[side]
        target_position = np.asarray(target_position, dtype=np.float64)
        position_only = target_quaternion is None
        if position_only:
            target_quaternion_array = None
        else:
            target_quaternion_array = _normalize_quaternion(np.asarray(target_quaternion, dtype=np.float64))
        if target_position.shape != (3,) or (
            target_quaternion_array is not None and target_quaternion_array.shape != (4,)
        ):
            raise ValueError("IK target must be a 3D position and an optional scalar-first quaternion")
        if not np.all(np.isfinite(target_position)) or (
            target_quaternion_array is not None and not np.all(np.isfinite(target_quaternion_array))
        ):
            raise ValueError("IK target contains NaN or Inf")

        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos
        ik_data.qvel[:] = 0.0
        lower, upper = self._joint_bounds(arm)
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))

        position_error_norm = float("inf")
        orientation_error_norm = float("inf")
        iterations = 0
        for iteration in range(1, self.config.ik_iterations + 1):
            iterations = iteration
            mujoco.mj_forward(self.model, ik_data)
            current_position = np.array(ik_data.site_xpos[arm.tcp_site_id], copy=True)
            current_matrix = np.array(ik_data.site_xmat[arm.tcp_site_id], copy=True).reshape(3, 3)
            position_error = target_position - current_position
            orientation_error = (
                np.zeros(3) if position_only else _rotation_error(target_quaternion_array, current_matrix)
            )
            position_error_norm = float(np.linalg.norm(position_error))
            orientation_error_norm = float(np.linalg.norm(orientation_error))
            if position_error_norm <= self.config.ik_position_tolerance and (
                position_only or orientation_error_norm <= self.config.ik_orientation_tolerance
            ):
                break

            mujoco.mj_jacSite(
                self.model,
                ik_data,
                jacobian_position,
                jacobian_rotation,
                arm.tcp_site_id,
            )
            if position_only:
                jacobian = jacobian_position[:, arm.dof_indices]
                error = position_error
            else:
                jacobian = np.vstack(
                    [
                        jacobian_position[:, arm.dof_indices],
                        jacobian_rotation[:, arm.dof_indices],
                    ]
                )
                error = np.concatenate([position_error, orientation_error])
            regularized = (
                jacobian @ jacobian.T + np.eye(jacobian.shape[0], dtype=np.float64) * self.config.ik_damping**2
            )
            delta = jacobian.T @ np.linalg.solve(regularized, error)
            delta *= self.config.ik_step_size
            max_delta = float(np.max(np.abs(delta)))
            if max_delta > self.config.ik_max_joint_step:
                delta *= self.config.ik_max_joint_step / max_delta

            qpos = ik_data.qpos[arm.qpos_indices] + delta
            ik_data.qpos[arm.qpos_indices] = np.clip(qpos, lower, upper)

        return IKResult(
            qpos=np.array(ik_data.qpos[arm.qpos_indices], copy=True),
            position_error=position_error_norm,
            orientation_error=orientation_error_norm,
            converged=(
                position_error_norm <= self.config.ik_position_tolerance
                and (position_only or orientation_error_norm <= self.config.ik_orientation_tolerance)
            ),
            iterations=iterations,
        )

    def _torque_limits(self, arm: ArmModelHandles) -> tuple[np.ndarray, np.ndarray]:
        ranges = np.asarray(self.model.actuator_forcerange[arm.actuator_ids])
        return ranges[:, 0], ranges[:, 1]

    def compute_ctrl(self) -> np.ndarray:
        """Compute and write safe arm torques and gripper position targets."""
        mujoco.mj_forward(self.model, self.data)
        for side, arm in self.mission.arms.items():
            qpos = np.asarray(self.data.qpos[arm.qpos_indices])
            qvel = np.asarray(self.data.qvel[arm.dof_indices])
            error = self.target_joint_qpos[side] - qpos
            error = np.arctan2(np.sin(error), np.cos(error))
            torque = self._kp * error - self._kd * qvel
            torque += self.data.qfrc_bias[arm.dof_indices]

            over_speed = np.abs(qvel) > self._velocity_limit
            if np.any(over_speed):
                torque[over_speed] = (
                    self.data.qfrc_bias[arm.dof_indices][over_speed] - self.config.brake_gain * qvel[over_speed]
                )

            lower, upper = self._torque_limits(arm)
            torque = np.clip(torque, lower, upper)
            self.data.ctrl[arm.actuator_ids] = torque
            self.data.ctrl[arm.finger_actuator_ids] = self.target_gripper[side]

        if not np.all(np.isfinite(self.data.ctrl)):
            raise RuntimeError("Controller produced NaN or Inf")
        return np.array(self.data.ctrl, copy=True)

    def set_gripper_force_limit(self, side: str, force_n: float) -> None:
        """Set a symmetric force cap for one position-impedance gripper."""
        if side not in self.mission.arms:
            raise ValueError(f"Unknown arm side: {side}")
        if not np.isfinite(force_n) or force_n <= 0.0:
            raise ValueError("Gripper force limit must be finite and positive")
        actuator_ids = self.mission.arms[side].finger_actuator_ids
        self.model.actuator_forcerange[actuator_ids, 0] = -float(force_n)
        self.model.actuator_forcerange[actuator_ids, 1] = float(force_n)

    def gripper_actuator_forces(self, side: str) -> np.ndarray:
        """Return absolute force produced by both finger actuators."""
        if side not in self.mission.arms:
            raise ValueError(f"Unknown arm side: {side}")
        actuator_ids = self.mission.arms[side].finger_actuator_ids
        return np.abs(np.asarray(self.data.actuator_force[actuator_ids])).copy()

    def step(self, substeps: int = 1) -> None:
        if substeps <= 0:
            raise ValueError("substeps must be positive")
        for _ in range(substeps):
            self.compute_ctrl()
            mujoco.mj_step(self.model, self.data)
