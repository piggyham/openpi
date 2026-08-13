"""Configuration shared by the OpenArm mission model and controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSION_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MissionConfig:
    """Frozen P0 scene configuration.

    Real-world spec layout (2026-08-08): a 26cm-high table with a 20cm-deep top
    surface (x = +-0.10), the near edge 24cm in front of the robot base's
    front-most surface (base column front at x = 0.03, so the near edge sits at
    x = 0.27 and the table center at x = 0.37), and a 4.5cm-diameter / 9.5cm-tall
    plastic water bottle instead of the paper cup. The bottle rests on the two
    region markers, each 25cm from the central axis (region y = +-0.25) and
    37cm in front of the base front (x = 0.37); the handoff pad sits at
    x = 0.39. All three pads were moved 5cm toward the robot (from x = 0.42 /
    0.44) on 2026-08-08. Reachability was verified end-to-end: 6/6
    scripted-expert seeds converge with zero recoveries under this layout.
    """

    dependency_root: Path = MISSION_ROOT / "third_party" / "openarm_mujoco"
    official_revision: str = "8955afb54e4adfb59a236e2b4d15192b7a02865c"

    table_center: tuple[float, float, float] = (0.37, 0.0, 0.24)
    table_half_size: tuple[float, float, float] = (0.10, 0.46, 0.02)
    region_a_center: tuple[float, float] = (0.37, -0.25)
    handoff_center: tuple[float, float] = (0.39, 0.0)
    region_b_center: tuple[float, float] = (0.37, 0.25)
    region_radius: float = 0.085

    cup_bottom_radius: float = 0.0225
    cup_top_radius: float = 0.0225
    cup_half_height: float = 0.0475
    cup_wall_thickness: float = 0.002
    cup_mass: float = 0.12
    # Vertical gap between the scripted-demo TCP and the object's top surface.
    # Zero so the demo's grasp TCP sits exactly at the bottle top, matching the
    # relay expert's physical_grasp_tcp_offset_z (= cup_half_height). The old
    # 0.030 paper-cup value left the fingers hovering ~2cm above this shorter
    # 9.5cm bottle. (Measured: at 0.030 the finger meshes bottom out at
    # z ~ 0.375 vs bottle top 0.357; at 0.0 they reach z ~ 0.346 and wrap it.)
    cup_grasp_clearance: float = 0.0
    cup_linear_damping: float = 0.08
    cup_angular_damping: float = 0.004
    # Scene composition switches (the openarm_sim viewer uses a robot+table
    # only scene; data collection / evals keep the full task scene).
    include_cup: bool = True
    include_region_markers: bool = True
    table_legs: bool = True
    soft_finger_pads: bool = False
    finger_servo_kp: float = 100.0
    finger_force_limit_n: float = 333.0
    finger_pad_half_size: tuple[float, float, float] = (
        0.008,
        0.0015,
        0.010,
    )
    finger_pad_center_x: float = -0.010
    finger_pad_center_y: float = 0.004
    finger_pad_center_z: float = 0.068
    finger_pad_friction: tuple[float, float, float] = (2.5, 0.08, 0.02)
    finger_pad_solref: tuple[float, float] = (0.006, 1.0)
    finger_pad_solimp: tuple[float, float, float] = (0.97, 0.995, 0.001)

    # All simulations reset with both arms hanging naturally beside the
    # pedestal.  The old reset pose (joint 4 at pi/2) is retained separately
    # as the collision-free task-ready pose used by the scripted experts.
    home_arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ready_arm_qpos: tuple[float, ...] = (0.0, 0.0, 0.0, 1.570796, 0.0, 0.0, 0.0)
    # The first scripted-expert motion is a deliberately sparse shoulder
    # retraction: only J1 and J4 move.  J1 is mirrored between arms so both
    # grippers rise just behind the table edge without involving the wrist.
    # 0.55 rad keeps the gripper well behind the near table edge throughout
    # the synchronized J1/J4 unfold. Values <=0.35 let the finger meshes graze
    # the tabletop; 0.55 leaves roughly 47 mm horizontal clearance at the end.
    arm_unfold_joint1: float = 0.55
    arm_unfold_joint4: float = 1.570796
    arm_unfold_seconds: float = 1.80
    # Legacy Cartesian clearance point retained for recovery/stowing paths.
    arm_stow_clearance_x: float = 0.20
    arm_stow_clearance_z: float = 0.52
    # Head camera sits just above the two shoulder pivots and looks down over
    # the task workspace. Wrist cameras pitch 30 degrees forward from their
    # former straight-down view, retaining enough downward component to see
    # the gripper and contact region.
    head_camera_position: tuple[float, float, float] = (0.08, 0.0, 0.82)
    head_camera_target: tuple[float, float, float] = (0.39, 0.0, 0.26)
    head_camera_fovy: float = 72.0
    wrist_camera_forward_tilt_deg: float = 30.0
    wrist_camera_fovy: float = 75.0
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
    # The grasp TCP sits at the object's top (cup_half_height = 0.0475 for the
    # plastic bottle), same relationship the relay chain had with the old paper
    # cup (half_height = 0.060 -> offset 0.060). The expert's grasp_z_offsets
    # reach 0.004 below that, so the vertical window must start below
    # 0.0475 - 0.004 = 0.0435.
    grasp_vertical_offset_range: tuple[float, float] = (0.040, 0.105)
    physical_grasp_tcp_offset_z: float = 0.0475
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
    """P4.5 model settings for soft-pad, weld-free grasping.

    Inherits the reachable layout, finger pads, table legs and table footprint
    from ``MissionConfig`` (the "maximally synced" viewer compromise); this
    subclass only tunes the grasp physics — soft pads, stiffer finger servo, a
    low force limit, and heavier cup damping.
    """

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
