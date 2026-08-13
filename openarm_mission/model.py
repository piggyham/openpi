"""OpenArm v1 model loading and mission-scene composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from openarm_mission.config import MissionConfig


class MissingOpenArmAssetsError(FileNotFoundError):
    """Raised when the pinned OpenArm model has not been fetched."""


@dataclass(frozen=True)
class ArmModelHandles:
    """Resolved MuJoCo indexes for one seven-axis arm and its gripper."""

    side: str
    joint_ids: np.ndarray
    qpos_indices: np.ndarray
    dof_indices: np.ndarray
    actuator_ids: np.ndarray
    finger_joint_ids: np.ndarray
    finger_qpos_indices: np.ndarray
    finger_actuator_ids: np.ndarray
    tcp_site_id: int


def _format_vector(values) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)


def _camera_xyaxes(position: np.ndarray, target: np.ndarray) -> str:
    """Return MuJoCo camera x/y axes for a camera looking at target."""
    forward = target - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(forward, world_up)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    z_axis = -forward
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return _format_vector(np.concatenate([x_axis, y_axis]))


def _add_mission_assets(asset: ET.Element, config: MissionConfig) -> None:
    materials = {
        "mission_table_material": "0.36 0.25 0.16 1",
        "mission_red_material": "0.9 0.05 0.05 1",
        "mission_blue_material": "0.05 0.18 0.95 1",
        "mission_handoff_material": "0.82 0.70 0.32 1",
        "mission_cup_material": "0.96 0.95 0.90 1",
        "mission_cup_inside_material": "0.12 0.12 0.12 1",
        "mission_finger_pad_material": "0.12 0.62 0.34 1",
        "mission_camera_material": "0.18 0.18 0.20 1",
        "mission_lens_material": "0.02 0.02 0.02 1",
    }
    for name, rgba in materials.items():
        ET.SubElement(asset, "material", name=name, rgba=rgba)

    vertices, faces = _paper_cup_frustum_mesh(
        bottom_radius=config.cup_bottom_radius,
        top_radius=config.cup_top_radius,
        half_height=config.cup_half_height,
        wall_thickness=config.cup_wall_thickness,
    )
    ET.SubElement(
        asset,
        "mesh",
        name="mission_paper_cup_mesh",
        vertex=_format_vector(vertices),
        face=" ".join(str(index) for index in faces),
    )


def _paper_cup_frustum_mesh(
    *,
    bottom_radius: float,
    top_radius: float,
    half_height: float,
    wall_thickness: float,
    segments: int = 32,
) -> tuple[np.ndarray, list[int]]:
    """Create a hollow, open-top paper cup with finite walls and bottom."""
    if not 0.0 < wall_thickness < min(bottom_radius, top_radius):
        raise ValueError("Paper-cup wall thickness must fit inside both radii")
    vertices: list[tuple[float, float, float]] = []
    rings = (
        (bottom_radius, -half_height),
        (top_radius, half_height),
        (bottom_radius - wall_thickness, -half_height + wall_thickness),
        (top_radius - wall_thickness, half_height),
    )
    for radius, z in rings:
        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            vertices.append((radius * np.cos(angle), radius * np.sin(angle), z))
    outer_bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height))
    inner_bottom_center = len(vertices)
    vertices.append((0.0, 0.0, -half_height + wall_thickness))

    faces: list[int] = []
    for index in range(segments):
        next_index = (index + 1) % segments
        outer_bottom = index
        outer_bottom_next = next_index
        outer_top = segments + index
        outer_top_next = segments + next_index
        inner_bottom = 2 * segments + index
        inner_bottom_next = 2 * segments + next_index
        inner_top = 3 * segments + index
        inner_top_next = 3 * segments + next_index

        faces.extend((outer_bottom, outer_bottom_next, outer_top_next))
        faces.extend((outer_bottom, outer_top_next, outer_top))
        faces.extend((inner_bottom, inner_top_next, inner_bottom_next))
        faces.extend((inner_bottom, inner_top, inner_top_next))
        faces.extend((outer_top, outer_top_next, inner_top_next))
        faces.extend((outer_top, inner_top_next, inner_top))
        faces.extend((outer_bottom_center, outer_bottom_next, outer_bottom))
        faces.extend((inner_bottom_center, inner_bottom, inner_bottom_next))
    return np.asarray(vertices, dtype=np.float64).reshape(-1), faces


def _add_tcp_sites(root: ET.Element, config: MissionConfig) -> None:
    for side, rgba in (("left", "1 0.3 0.1 1"), ("right", "0.1 0.5 1 1")):
        body = root.find(f".//body[@name='openarm_{side}_hand_tcp']")
        if body is None:
            raise ValueError(f"Official model is missing openarm_{side}_hand_tcp")
        ET.SubElement(
            body,
            "site",
            name=f"mission_{side}_tcp",
            pos="0 0 0",
            size="0.009",
            rgba=rgba,
            group="4",
        )
        # Camera bracket mounted on the gripper-motor housing (J8 / `hand`
        # body) at the J7-J8 junction. In the hand frame +x is world-up and +z
        # is forward (toward the cup). The bracket is an L: a short vertical
        # post along +x then a horizontal arm along +z, carrying the camera
        # housing at its end. The render sensor keeps the 90deg-about-y quat so
        # it still looks straight world-down from the same vantage point
        # (hand + 0.03*up + 0.105*forward == the previous raised position),
        # which leaves the existing dataset viewpoint unchanged.
        hand = root.find(f".//body[@name='openarm_{side}_hand']")
        if hand is None:
            raise ValueError(f"Official model is missing openarm_{side}_hand")
        bracket = ET.SubElement(
            hand,
            "body",
            name=f"mission_{side}_camera_bracket",
            pos="0 0 0",
        )
        ET.SubElement(
            bracket,
            "inertial",
            pos="0.04 0 0.025",
            mass="0.05",
            diaginertia="1e-5 1e-5 3e-6",
        )
        # Vertical post (rises along hand +x = world up).
        ET.SubElement(
            bracket,
            "geom",
            name=f"mission_{side}_camera_bracket_post",
            type="box",
            pos="0.04 0 0",
            size="0.04 0.004 0.004",
            material="mission_camera_material",
            group="0",
            contype="0",
            conaffinity="0",
        )
        # Horizontal arm (runs forward along hand +z toward the grip point).
        ET.SubElement(
            bracket,
            "geom",
            name=f"mission_{side}_camera_bracket_arm",
            type="box",
            pos="0.08 0 0.025",
            size="0.004 0.004 0.025",
            material="mission_camera_material",
            group="0",
            contype="0",
            conaffinity="0",
        )
        # Collision volumes for the bracket.
        ET.SubElement(
            bracket,
            "geom",
            name=f"mission_{side}_camera_bracket_post_collision",
            type="box",
            pos="0.04 0 0",
            size="0.04 0.004 0.004",
            rgba="0.18 0.18 0.20 0.25",
            group="3",
            contype="1",
            conaffinity="1",
        )
        ET.SubElement(
            bracket,
            "geom",
            name=f"mission_{side}_camera_bracket_arm_collision",
            type="box",
            pos="0.08 0 0.025",
            size="0.004 0.004 0.025",
            rgba="0.18 0.18 0.20 0.25",
            group="3",
            contype="1",
            conaffinity="1",
        )
        # Camera housing at the bracket end. Its +z is world-up (behind the
        # lens), so the housing stays out of the downward view.
        mount = ET.SubElement(
            bracket,
            "body",
            name=f"mission_{side}_wrist_camera_mount",
            pos="0.08 0 0.05",
            quat="0.5 0.5 0.5 0.5",
        )
        ET.SubElement(
            mount,
            "inertial",
            pos="0 0 0.006",
            mass="0.03",
            diaginertia="3e-6 3e-6 2e-6",
        )
        ET.SubElement(
            mount,
            "geom",
            name=f"mission_{side}_wrist_camera_housing",
            type="box",
            pos="0 0 0.006",
            size="0.014 0.014 0.008",
            material="mission_camera_material",
            group="0",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            mount,
            "geom",
            name=f"mission_{side}_wrist_camera_lens",
            type="cylinder",
            pos="0 0 0.001",
            size="0.006 0.002",
            material="mission_lens_material",
            group="0",
            contype="0",
            conaffinity="0",
        )
        ET.SubElement(
            mount,
            "geom",
            name=f"mission_{side}_wrist_camera_collision",
            type="box",
            pos="0 0 0.006",
            size="0.014 0.014 0.008",
            rgba="0.18 0.18 0.20 0.25",
            group="3",
            contype="1",
            conaffinity="1",
        )
        tilt = np.deg2rad(config.wrist_camera_forward_tilt_deg)
        camera_quat = np.array([np.cos(tilt / 2.0), np.sin(tilt / 2.0), 0.0, 0.0])
        ET.SubElement(
            mount,
            "camera",
            name=f"mission_{side}_wrist_camera",
            pos="0 0 0",
            quat=_format_vector(camera_quat),
            fovy=str(config.wrist_camera_fovy),
        )


def _make_grippers_position_controlled(
    root: ET.Element,
    config: MissionConfig,
) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        raise ValueError("Official model does not define actuators")
    for side in ("left", "right"):
        for finger in (1, 2):
            name = f"{side}_finger{finger}_ctrl"
            element = actuator.find(f"*[@name='{name}']")
            if element is None:
                raise ValueError(f"Official model is missing actuator {name}")
            element.tag = "position"
            element.set("kp", str(config.finger_servo_kp))
            element.set("ctrllimited", "true")
            element.set("ctrlrange", "0 0.044")
            element.set("forcelimited", "true")
            element.set(
                "forcerange",
                _format_vector(
                    (
                        -config.finger_force_limit_n,
                        config.finger_force_limit_n,
                    )
                ),
            )


def _add_soft_finger_pads(
    root: ET.Element,
    config: MissionConfig,
) -> None:
    """Replace rigid mesh contacts with compliant inner-finger pads."""
    if not config.soft_finger_pads:
        return
    for side in ("left", "right"):
        for finger, y_sign in (("left", -1.0), ("right", 1.0)):
            body = root.find(f".//body[@name='openarm_{side}_{finger}_finger']")
            if body is None:
                raise ValueError(f"Missing {side} {finger} finger body")
            collision = body.find(f"geom[@name='openarm_{side}_{finger}_finger_collision']")
            if collision is None:
                raise ValueError(f"Missing {side} {finger} finger collision")
            collision.set("contype", "0")
            collision.set("conaffinity", "0")
            ET.SubElement(
                body,
                "geom",
                name=f"mission_{side}_{finger}_finger_pad",
                type="box",
                pos=_format_vector(
                    (
                        config.finger_pad_center_x,
                        y_sign * config.finger_pad_center_y,
                        config.finger_pad_center_z,
                    )
                ),
                size=_format_vector(config.finger_pad_half_size),
                material="mission_finger_pad_material",
                mass="0",
                friction=_format_vector(config.finger_pad_friction),
                condim="6",
                solref=_format_vector(config.finger_pad_solref),
                solimp=_format_vector(config.finger_pad_solimp),
                margin="0.0005",
            )


def _stabilize_arm_dynamics(root: ET.Element) -> None:
    """Add reflected motor inertia used by the maintained OpenArm models."""
    dynamics = {
        1: ("0.0081", "1.0"),
        2: ("0.0081", "1.0"),
        3: ("0.1600", "0.9"),
        4: ("0.1600", "0.9"),
        5: ("0.0100", "0.9"),
        6: ("0.0100", "0.9"),
        7: ("0.0100", "0.9"),
    }
    for side in ("left", "right"):
        for joint_index, (armature, damping) in dynamics.items():
            joint = root.find(f".//joint[@name='openarm_{side}_joint{joint_index}']")
            if joint is None:
                raise ValueError(f"Official model is missing openarm_{side}_joint{joint_index}")
            joint.set("armature", armature)
            joint.set("damping", damping)


def _add_mission_cameras(worldbody: ET.Element, config: MissionConfig) -> None:
    target = np.array([config.table_center[0], 0.0, config.table_top_z])
    front_pos = np.asarray(config.head_camera_position, dtype=np.float64)
    front_target = np.asarray(config.head_camera_target, dtype=np.float64)
    overhead_pos = np.array([config.table_center[0], 0.0, 1.25])
    ET.SubElement(
        worldbody,
        "camera",
        name="mission_front_camera",
        mode="fixed",
        pos=_format_vector(front_pos),
        xyaxes=_camera_xyaxes(front_pos, front_target),
        fovy=str(config.head_camera_fovy),
    )
    ET.SubElement(
        worldbody,
        "camera",
        name="mission_overhead_camera",
        mode="fixed",
        pos=_format_vector(overhead_pos),
        xyaxes=_camera_xyaxes(overhead_pos, target),
        fovy="52",
    )


def _add_task_scene(root: ET.Element, config: MissionConfig) -> None:
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError("Official model is missing asset or worldbody")
    _add_mission_assets(asset, config)

    ET.SubElement(
        worldbody,
        "light",
        name="mission_key_light",
        pos="0.5 0 1.4",
        dir="0 0 -1",
        directional="true",
        diffuse="0.8 0.8 0.8",
    )
    ET.SubElement(
        worldbody,
        "geom",
        name="mission_floor",
        type="plane",
        pos="0 0 -0.005",
        size="2 2 0.02",
        rgba="0.16 0.18 0.20 1",
        friction="1 0.01 0.001",
    )

    table = ET.SubElement(
        worldbody,
        "body",
        name="mission_table",
        pos=_format_vector(config.table_center),
    )
    ET.SubElement(
        table,
        "geom",
        name="mission_table_top",
        type="box",
        size=_format_vector(config.table_half_size),
        material="mission_table_material",
        friction="1.2 0.02 0.002",
    )

    if config.table_legs:
        leg_top = config.table_center[2] - config.table_half_size[2]
        leg_half_z = leg_top / 2.0
        # Place the square legs at the table corners. Inset by the leg half-width
        # so the outer leg faces sit flush with the table edge; the older
        # hard-coded 0.08 inset put the legs outside a narrow (10cm-deep) table.
        leg_x = config.table_half_size[0] - 0.03
        leg_y = config.table_half_size[1] - 0.03
        for sx in (-1, 1):
            for sy in (-1, 1):
                ET.SubElement(
                    table,
                    "geom",
                    name=f"mission_table_leg_{'p' if sx > 0 else 'n'}{'p' if sy > 0 else 'n'}",
                    type="box",
                    pos=_format_vector(
                        (
                            sx * leg_x,
                            sy * leg_y,
                            leg_half_z - config.table_center[2],
                        )
                    ),
                    size=_format_vector((0.03, 0.03, leg_half_z)),
                    material="mission_table_material",
                )

    if config.include_region_markers:
        region_z = config.table_half_size[2] + 0.003
        for name, center, material, rgba in (
            ("mission_region_a", config.region_a_center, "mission_red_material", "0.9 0.05 0.05 1"),
            ("mission_region_b", config.region_b_center, "mission_blue_material", "0.05 0.18 0.95 1"),
        ):
            ET.SubElement(
                table,
                "geom",
                name=name,
                type="cylinder",
                pos=_format_vector(
                    (
                        center[0] - config.table_center[0],
                        center[1] - config.table_center[1],
                        region_z,
                    )
                ),
                size=_format_vector((config.region_radius, 0.0025)),
                material=material,
                rgba=rgba,
                contype="0",
                conaffinity="0",
                group="2",
            )
        ET.SubElement(
            table,
            "geom",
            name="mission_handoff_marker",
            type="cylinder",
            pos=_format_vector(
                (
                    config.handoff_center[0] - config.table_center[0],
                    config.handoff_center[1] - config.table_center[1],
                    region_z,
                )
            ),
            size="0.07 0.0018",
            material="mission_handoff_material",
            rgba="0.82 0.70 0.32 0.55",
            contype="0",
            conaffinity="0",
            group="2",
        )

    if not config.include_cup:
        _add_mission_cameras(worldbody, config)
        return
    cup = ET.SubElement(
        worldbody,
        "body",
        name="mission_cup",
        pos=_format_vector(config.cup_initial_position),
    )
    ET.SubElement(cup, "freejoint", name="mission_cup_freejoint")
    ET.SubElement(
        cup,
        "geom",
        name="mission_cup_body",
        type="mesh",
        mesh="mission_paper_cup_mesh",
        material="mission_cup_material",
        mass=str(config.cup_mass),
        friction="1.8 0.05 0.005",
        condim="4",
        solref="0.008 1",
        solimp="0.95 0.99 0.001",
    )
    ET.SubElement(
        cup,
        "geom",
        name="mission_cup_inside",
        type="cylinder",
        pos=_format_vector((0.0, 0.0, config.cup_half_height - 0.003)),
        size=_format_vector((config.cup_top_radius * 0.88, 0.0015)),
        material="mission_cup_inside_material",
        mass="0",
        contype="0",
        conaffinity="0",
        group="2",
    )
    for index in range(32):
        angle0 = 2.0 * np.pi * index / 32
        angle1 = 2.0 * np.pi * (index + 1) / 32
        start = (
            config.cup_top_radius * np.cos(angle0),
            config.cup_top_radius * np.sin(angle0),
            config.cup_half_height,
        )
        end = (
            config.cup_top_radius * np.cos(angle1),
            config.cup_top_radius * np.sin(angle1),
            config.cup_half_height,
        )
        ET.SubElement(
            cup,
            "geom",
            name=f"mission_cup_rim_{index}",
            type="capsule",
            fromto=_format_vector((*start, *end)),
            size="0.003",
            material="mission_cup_material",
            mass="0",
            contype="0",
            conaffinity="0",
            group="2",
        )
    ET.SubElement(
        cup,
        "site",
        name="mission_cup_center",
        pos="0 0 0",
        size="0.006",
        rgba="0 1 0 1",
        group="4",
    )

    _add_mission_cameras(worldbody, config)

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for side in ("left", "right"):
        ET.SubElement(
            equality,
            "weld",
            name=f"mission_{side}_cup_weld",
            body1=f"openarm_{side}_hand_tcp",
            body2="mission_cup",
            active="false",
            relpose="0 0 0 1 0 0 0",
            solref="0.008 1",
            solimp="0.95 0.99 0.001",
            torquescale="0.05",
        )


def build_mission_xml(config: MissionConfig | None = None) -> str:
    """Build a self-contained XML string around the pinned OpenArm v1 model."""
    config = config or MissionConfig()
    model_path = config.official_model_path
    if not model_path.is_file():
        raise MissingOpenArmAssetsError(f"Missing {model_path}. Run: bash openarm_mission/fetch_openarm_v1.sh")
    if not config.mesh_dir.is_dir():
        raise MissingOpenArmAssetsError(f"Missing OpenArm mesh directory: {config.mesh_dir}")

    root = ET.parse(model_path).getroot()
    root.set("model", "openarm_v1_bimanual_cup_mission")

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(config.mesh_dir.resolve()))
    compiler.set("angle", "radian")

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", str(config.timestep))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("iterations", "100")
    option.set("noslip_iterations", "8")
    option.set("tolerance", "1e-10")

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(2, visual)
    visual_global = visual.find("global")
    if visual_global is None:
        visual_global = ET.SubElement(visual, "global")
    visual_global.set("offwidth", "1280")
    visual_global.set("offheight", "720")

    _make_grippers_position_controlled(root, config)
    _stabilize_arm_dynamics(root)
    _add_tcp_sites(root, config)
    _add_task_scene(root, config)
    _add_soft_finger_pads(root, config)
    for side in ("left", "right"):
        for finger in ("left", "right"):
            geom = root.find(f".//geom[@name='openarm_{side}_{finger}_finger_collision']")
            if geom is None:
                raise ValueError(f"Missing {side} {finger} finger collision geom")
            if not config.soft_finger_pads:
                geom.set("condim", "4")
                geom.set("friction", "2.0 0.05 0.005")
                geom.set("solref", "0.006 1")
                geom.set("solimp", "0.95 0.99 0.001")
    return ET.tostring(root, encoding="unicode")


class OpenArmMission:
    """Compiled OpenArm v1 bimanual mission model and its resolved handles."""

    def __init__(self, config: MissionConfig | None = None):
        self.config = config or MissionConfig()
        self.xml = build_mission_xml(self.config)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.arms = {side: self._resolve_arm(side) for side in ("left", "right")}
        self.has_cup = bool(self.config.include_cup)
        if self.has_cup:
            self.cup_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "mission_cup_freejoint")
            self.cup_qpos_address = int(self.model.jnt_qposadr[self.cup_joint_id])
            self.cup_dof_address = int(self.model.jnt_dofadr[self.cup_joint_id])
            self.model.dof_damping[self.cup_dof_address : self.cup_dof_address + 3] = self.config.cup_linear_damping
            self.model.dof_damping[self.cup_dof_address + 3 : self.cup_dof_address + 6] = (
                self.config.cup_angular_damping
            )
            self.cup_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "mission_cup")
            self.cup_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "mission_cup_body")
            self.cup_weld_ids = {
                side: self._id(
                    mujoco.mjtObj.mjOBJ_EQUALITY,
                    f"mission_{side}_cup_weld",
                )
                for side in ("left", "right")
            }
        else:
            self.cup_joint_id = None
            self.cup_qpos_address = None
            self.cup_dof_address = None
            self.cup_body_id = None
            self.cup_geom_id = None
            self.cup_weld_ids = {}
        self.finger_pad_geom_ids = {
            side: {
                finger: mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"mission_{side}_{finger}_finger_pad",
                )
                for finger in ("left", "right")
            }
            for side in ("left", "right")
        }
        self.reset()

    def _id(self, object_type, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return int(object_id)

    def _resolve_arm(self, side: str) -> ArmModelHandles:
        joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_joint{i}") for i in range(1, 8)],
            dtype=np.intp,
        )
        finger_joint_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint{i}") for i in (1, 2)],
            dtype=np.intp,
        )
        actuator_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_joint{i}_ctrl") for i in range(1, 8)],
            dtype=np.intp,
        )
        finger_actuator_ids = np.array(
            [self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{side}_finger{i}_ctrl") for i in (1, 2)],
            dtype=np.intp,
        )
        return ArmModelHandles(
            side=side,
            joint_ids=joint_ids,
            qpos_indices=self.model.jnt_qposadr[joint_ids].astype(np.intp),
            dof_indices=self.model.jnt_dofadr[joint_ids].astype(np.intp),
            actuator_ids=actuator_ids,
            finger_joint_ids=finger_joint_ids,
            finger_qpos_indices=self.model.jnt_qposadr[finger_joint_ids].astype(np.intp),
            finger_actuator_ids=finger_actuator_ids,
            tcp_site_id=self._id(mujoco.mjtObj.mjOBJ_SITE, f"mission_{side}_tcp"),
        )

    def reset(self) -> None:
        """Reset arms, open grippers and place the cup in region A."""
        mujoco.mj_resetData(self.model, self.data)
        for equality_id in self.cup_weld_ids.values():
            if hasattr(self.data, "eq_active"):
                self.data.eq_active[equality_id] = 0
            else:
                self.model.eq_active[equality_id] = 0
        home = np.asarray(self.config.home_arm_qpos)
        for arm in self.arms.values():
            self.data.qpos[arm.qpos_indices] = home
            self.data.qpos[arm.finger_qpos_indices] = self.config.open_finger_qpos
            self.data.ctrl[arm.finger_actuator_ids] = self.config.open_finger_qpos

        if self.has_cup:
            cup_position = np.asarray(self.config.cup_initial_position)
            self.data.qpos[self.cup_qpos_address : self.cup_qpos_address + 7] = np.concatenate(
                [cup_position, np.array([1.0, 0.0, 0.0, 0.0])]
            )
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def tcp_pose(self, side: str, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Return TCP position and scalar-first quaternion."""
        data = data or self.data
        site_id = self.arms[side].tcp_site_id
        position = np.array(data.site_xpos[site_id], copy=True)
        rotation = np.array(data.site_xmat[site_id], copy=True).reshape(3, 3)
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, rotation.reshape(-1))
        return position, quaternion

    def _require_cup(self) -> None:
        if not self.has_cup:
            raise RuntimeError("this scene configuration has no cup")

    def cup_position(self) -> np.ndarray:
        self._require_cup()
        return np.array(self.data.qpos[self.cup_qpos_address : self.cup_qpos_address + 3], copy=True)

    def cup_quaternion(self) -> np.ndarray:
        self._require_cup()
        return np.array(
            self.data.qpos[self.cup_qpos_address + 3 : self.cup_qpos_address + 7],
            copy=True,
        )

    def cup_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        self._require_cup()
        return (
            np.array(
                self.data.qvel[self.cup_dof_address : self.cup_dof_address + 3],
                copy=True,
            ),
            np.array(
                self.data.qvel[self.cup_dof_address + 3 : self.cup_dof_address + 6],
                copy=True,
            ),
        )

    def save_expanded_xml(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.xml, encoding="utf-8")
        return output
