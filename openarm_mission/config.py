"""Configuration shared by the OpenArm mission model and controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSION_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MissionConfig:
    """Frozen P0 scene configuration."""

    dependency_root: Path = MISSION_ROOT / "third_party" / "openarm_mujoco"
    official_revision: str = "8955afb54e4adfb59a236e2b4d15192b7a02865c"

    table_center: tuple[float, float, float] = (0.50, 0.0, 0.24)
    table_half_size: tuple[float, float, float] = (0.38, 0.36, 0.04)
    region_a_center: tuple[float, float] = (0.46, -0.20)
    handoff_center: tuple[float, float] = (0.50, 0.0)
    region_b_center: tuple[float, float] = (0.46, 0.20)
    region_radius: float = 0.085

    cup_bottom_radius: float = 0.026
    cup_top_radius: float = 0.035
    cup_half_height: float = 0.060
    cup_wall_thickness: float = 0.002
    cup_mass: float = 0.035
    cup_grasp_clearance: float = 0.030
    cup_linear_damping: float = 0.08
    cup_angular_damping: float = 0.004
    soft_finger_pads: bool = False
    finger_servo_kp: float = 100.0
    finger_force_limit_n: float = 333.0
    finger_pad_half_size: tuple[float, float, float] = (
        0.018,
        0.0015,
        0.012,
    )
    finger_pad_center_x: float = -0.020
    finger_pad_center_y: float = 0.004
    finger_pad_center_z: float = 0.060
    finger_pad_friction: tuple[float, float, float] = (2.5, 0.08, 0.02)
    finger_pad_solref: tuple[float, float] = (0.006, 1.0)
    finger_pad_solimp: tuple[float, float, float] = (0.97, 0.995, 0.001)

    home_arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 1.570796, 0.0, 0.0, 0.0)
    open_finger_qpos: float = 0.044
    timestep: float = 0.002

    @property
    def official_model_path(self) -> Path:
        return self.dependency_root / "v1" / "openarm_bimanual.xml"

    @property
    def mesh_dir(self) -> Path:
        return self.dependency_root / "v1" / "meshes"

    @property
    def table_top_z(self) -> float:
        return self.table_center[2] + self.table_half_size[2]

    @property
    def cup_initial_position(self) -> tuple[float, float, float]:
        return (
            self.region_a_center[0],
            self.region_a_center[1],
            self.table_top_z + self.cup_half_height + 0.002,
        )

    @property
    def cup_tcp_offset_z(self) -> float:
        """Vertical cup-center offset from the TCP during a scripted grasp."""
        return -(self.cup_half_height + self.cup_grasp_clearance)

    @property
    def cup_grasp_tcp_z(self) -> float:
        """TCP height that preserves the cup's resting center during closure."""
        return self.cup_initial_position[2] - self.cup_tcp_offset_z


@dataclass(frozen=True)
class ControllerConfig:
    """P2 controller gains and hard safety limits."""

    max_translation_delta: float = 0.025
    max_rotation_delta: float = 0.12
    max_joint_target_delta: float = 0.20

    workspace_min: tuple[float, float, float] = (0.18, -0.46, 0.30)
    workspace_max: tuple[float, float, float] = (0.82, 0.46, 0.78)
    joint_limit_margin: float = 0.015

    ik_damping: float = 0.035
    ik_step_size: float = 0.65
    ik_max_joint_step: float = 0.10
    ik_iterations: int = 100
    ik_position_tolerance: float = 0.003
    ik_orientation_tolerance: float = 0.025

    joint_kp: tuple[float, ...] = (120.0, 120.0, 85.0, 75.0, 30.0, 30.0, 30.0)
    joint_kd: tuple[float, ...] = (2.7, 2.7, 2.2, 2.2, 1.5, 1.5, 1.5)
    max_joint_velocity: tuple[float, ...] = (2.5, 2.5, 2.5, 2.5, 3.0, 3.0, 3.0)
    brake_gain: float = 10.0

    gripper_open_position: float = 0.044
    gripper_closed_position: float = 0.0

    def workspace_min_array(self) -> np.ndarray:
        return np.asarray(self.workspace_min, dtype=np.float64)

    def workspace_max_array(self) -> np.ndarray:
        return np.asarray(self.workspace_max, dtype=np.float64)


@dataclass(frozen=True)
class RelayTaskConfig:
    """P3 task randomization, grasp gating and terminal thresholds."""

    initial_xy_randomization_radius: float = 0.015
    initial_yaw_range: tuple[float, float] = (-np.pi, np.pi)
    cup_mass_scale_range: tuple[float, float] = (0.85, 1.15)
    cup_friction_range: tuple[float, float] = (1.5, 2.2)

    grasp_opening_max: float = 0.039
    release_opening_min: float = 0.034
    grasp_horizontal_tolerance: float = 0.045
    grasp_vertical_offset_range: tuple[float, float] = (0.045, 0.105)
    physical_grasp_tcp_offset_z: float = 0.060
    lift_clearance: float = 0.035
    slip_position_tolerance: float = 0.025
    slip_angle_tolerance_deg: float = 25.0

    handoff_radius: float = 0.033
    goal_radius: float = 0.048
    upright_limit_deg: float = 20.0
    max_ungrasped_tilt_deg: float = 75.0
    stable_linear_speed: float = 0.04
    stable_angular_speed: float = 0.75
    handoff_hold_seconds: float = 0.25
    goal_hold_seconds: float = 0.50
    timeout_seconds: float = 35.0
    drop_below_table_margin: float = 0.035


@dataclass(frozen=True)
class ScriptedExpertConfig:
    """P4 scripted-expert approach variants and recovery limits."""

    pregrasp_height: float = 0.50
    retreat_height: float = 0.52
    closed_gripper_target: float = 0.008
    max_grasp_attempts_per_arm: int = 3
    max_episode_attempts: int = 2
    grasp_xy_offsets: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (0.004, 0.0),
        (-0.004, 0.0),
    )
    grasp_z_offsets: tuple[float, ...] = (0.0, -0.004, 0.004)
    recovery_retract_seconds: float = 0.55
    recovery_settle_seconds: float = 0.25
    benchmark_success_threshold: float = 0.95


@dataclass(frozen=True)
class FrictionMissionConfig(MissionConfig):
    """P4.5 model settings for soft-pad, weld-free grasping."""

    table_center: tuple[float, float, float] = (0.50, 0.0, 0.29)
    soft_finger_pads: bool = True
    finger_servo_kp: float = 240.0
    finger_force_limit_n: float = 8.0
    cup_linear_damping: float = 0.035
    cup_angular_damping: float = 0.002


@dataclass(frozen=True)
class FrictionTaskConfig(RelayTaskConfig):
    """P4.5 contact-force, slip and physical-release thresholds."""

    physical_grasp_tcp_offset_z: float = 0.020
    grasp_vertical_offset_range: tuple[float, float] = (0.005, 0.055)
    pad_slide_friction_range: tuple[float, float] = (2.1, 2.9)
    gripper_force_limit_range_n: tuple[float, float] = (8.0, 12.0)
    min_finger_force_n: float = 0.8
    contact_loss_grace_seconds: float = 0.12
    grasp_position_slip_tolerance: float = 0.014
    grasp_angle_slip_tolerance_deg: float = 18.0
    static_grasp_hold_seconds: float = 0.30
    lifted_grasp_hold_seconds: float = 0.30
    release_hover_clearance: float = 0.050
    max_release_drop_height: float = 0.065


@dataclass(frozen=True)
class FrictionExpertConfig(ScriptedExpertConfig):
    """P4.5 slow transport and force-limited grasp parameters."""

    retreat_height: float = 0.64
    closed_gripper_target: float = 0.0
    contact_settle_seconds: float = 0.30
    lift_seconds: float = 1.20
    transport_seconds: float = 1.55
    descend_seconds: float = 1.10
    place_settle_seconds: float = 0.45
