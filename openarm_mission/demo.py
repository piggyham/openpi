"""Render a presentation-ready OpenArm v1 bimanual paper-cup relay."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

from openarm_mission.controller import BimanualCartesianController
from openarm_mission.model import OpenArmMission


@dataclass(frozen=True)
class PhaseStyle:
    """Text and route-highlight metadata for one animation phase."""

    title: str
    subtitle: str
    route_step: int
    accent: tuple[int, int, int]


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/opentype/noto/" + ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Medium.ttc")),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


class RelayDemo:
    """Deterministic scripted showcase built on the real OpenArm dynamics."""

    def __init__(
        self,
        *,
        output_dir: Path,
        fps: int,
        width: int,
        height: int,
        duration_scale: float,
        write_gif: bool,
    ):
        if fps <= 0 or width < 640 or height < 480 or duration_scale <= 0:
            raise ValueError("fps/size/duration_scale values are outside supported limits")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.duration_scale = duration_scale
        self.write_gif = write_gif

        self.mission = OpenArmMission()
        self.controller = BimanualCartesianController(self.mission)
        self.renderer = mujoco.Renderer(
            self.mission.model,
            height=self.height,
            width=self.width,
        )
        self.home_qpos = {
            side: np.array(
                self.mission.data.qpos[arm.qpos_indices],
                copy=True,
            )
            for side, arm in self.mission.arms.items()
        }

        self.cup_dof_address = int(self.mission.model.jnt_dofadr[self.mission.cup_joint_id])
        self.cup_anchor = np.asarray(
            self.mission.config.cup_initial_position,
            dtype=np.float64,
        )
        self.attached_side: str | None = None
        self._cup_tcp_offset = np.array([0.0, 0.0, self.mission.config.cup_tcp_offset_z])
        self._disable_scripted_cup_contacts()
        self._sync_cup()

        self.mp4_path = self.output_dir / "openarm_paper_cup_relay.mp4"
        self.gif_path = self.output_dir / "openarm_paper_cup_relay.gif"
        self.storyboard_path = self.output_dir / "openarm_paper_cup_relay_storyboard.png"
        self.summary_path = self.output_dir / "openarm_paper_cup_relay.json"
        self._writer = imageio.get_writer(
            self.mp4_path,
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        self._gif_frames: list[Image.Image] = []
        self._gif_stride = max(1, round(self.fps / 8))
        self._frame_index = 0
        self._storyboard_frames: dict[str, Image.Image] = {}
        self._last_frame: Image.Image | None = None

        # Used only for the overall progress bar.
        self._total_duration = 17.05 * self.duration_scale
        self._total_frames = max(1, round(self._total_duration * self.fps))

    def _disable_scripted_cup_contacts(self) -> None:
        """Keep the showcase grasp deterministic across MuJoCo versions.

        The arms remain torque-controlled physical bodies. The cup is moved by
        a small scripted grasp latch so that this P0-P2 visualization does not
        pretend to be a learned grasp policy.
        """
        for geom_id in range(self.mission.model.ngeom):
            name = mujoco.mj_id2name(
                self.mission.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            if name and name.startswith("mission_cup_"):
                self.mission.model.geom_contype[geom_id] = 0
                self.mission.model.geom_conaffinity[geom_id] = 0

    def _sync_cup(self) -> None:
        if self.attached_side is None:
            position = self.cup_anchor
        else:
            tcp_position, _ = self.mission.tcp_pose(self.attached_side)
            position = tcp_position + self._cup_tcp_offset
        address = self.mission.cup_qpos_address
        self.mission.data.qpos[address : address + 7] = np.concatenate([position, np.array([1.0, 0.0, 0.0, 0.0])])
        self.mission.data.qvel[self.cup_dof_address : self.cup_dof_address + 6] = 0.0
        mujoco.mj_forward(self.mission.model, self.mission.data)

    def _attach(self, side: str) -> None:
        self.attached_side = side
        self._sync_cup()

    def _release_at(self, xy: tuple[float, float]) -> None:
        self.attached_side = None
        self.cup_anchor = np.array(
            [
                xy[0],
                xy[1],
                self.mission.config.table_top_z + self.mission.config.cup_half_height + 0.002,
            ],
            dtype=np.float64,
        )
        self._sync_cup()

    def _step_physics(self) -> None:
        substeps = max(
            1,
            round(1.0 / (self.fps * self.mission.config.timestep)),
        )
        for _ in range(substeps):
            self.controller.compute_ctrl()
            mujoco.mj_step(self.mission.model, self.mission.data)
            self._sync_cup()

    def _solve_position(self, side: str, target: Iterable[float]) -> np.ndarray:
        target_array = np.asarray(tuple(target), dtype=np.float64)
        result = self.controller.solve_ik(side, target_array)
        if not result.converged:
            raise RuntimeError(
                f"{side} position-only IK failed for {target_array.tolist()}: {result.position_error:.4f} m"
            )
        return result.qpos

    def _render_raw(self, camera: str) -> Image.Image:
        self.renderer.update_scene(self.mission.data, camera=camera)
        return Image.fromarray(self.renderer.render()).convert("RGBA")

    def _overlay(
        self,
        front: Image.Image,
        overhead: Image.Image,
        style: PhaseStyle,
        local_progress: float,
    ) -> Image.Image:
        title_font = _font(max(22, self.width // 34), bold=True)
        phase_font = _font(max(18, self.width // 48), bold=True)
        small_font = _font(max(13, self.width // 68))
        tiny_font = _font(max(11, self.width // 82), bold=True)

        header_h = max(78, self.height // 8)
        route_y = self.height - 58
        panel_w = max(245, self.width // 4)
        scene_w = self.width - panel_w
        content_h = route_y - header_h

        canvas = Image.new("RGBA", (self.width, self.height), (12, 16, 21, 255))
        scene = ImageOps.contain(
            front,
            (scene_w, content_h),
            Image.Resampling.LANCZOS,
        )
        scene_x = (scene_w - scene.width) // 2
        scene_y = header_h + (content_h - scene.height) // 2
        canvas.alpha_composite(scene, (scene_x, scene_y))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.line(
            (scene_w, header_h, scene_w, route_y),
            fill=(85, 94, 108, 210),
            width=2,
        )

        draw.rectangle((0, 0, self.width, header_h), fill=(12, 16, 21, 255))
        draw.text(
            (24, 12),
            "OpenArm v1 · 双臂纸杯接力",
            font=title_font,
            fill=(248, 249, 252, 255),
        )
        draw.text(
            (26, 49),
            "右手: 红区取杯 → 中央放置    左手: 中央取杯 → 蓝区放置",
            font=small_font,
            fill=(190, 198, 210, 255),
        )

        inset_w = panel_w - 24
        inset_h = round(inset_w * 0.64)
        inset = overhead.resize((inset_w, inset_h), Image.Resampling.LANCZOS)
        inset_x = scene_w + 12
        inset_y = header_h + 12
        draw.rounded_rectangle(
            (
                inset_x - 3,
                inset_y - 3,
                inset_x + inset_w + 3,
                inset_y + inset_h + 3,
            ),
            radius=8,
            fill=(8, 11, 15, 255),
            outline=(235, 238, 244, 220),
            width=2,
        )
        canvas.alpha_composite(inset, (inset_x, inset_y))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rounded_rectangle(
            (inset_x + 8, inset_y + 8, inset_x + 118, inset_y + 34),
            radius=8,
            fill=(10, 14, 20, 215),
        )
        draw.text(
            (inset_x + 17, inset_y + 11),
            "俯视任务地图",
            font=tiny_font,
            fill=(245, 247, 251, 255),
        )

        badge_x = scene_w + 12
        badge_y = inset_y + inset_h + 16
        badge_w = panel_w - 24
        badge_h = min(126, route_y - badge_y - 50)
        draw.rounded_rectangle(
            (badge_x, badge_y, badge_x + badge_w, badge_y + badge_h),
            radius=12,
            fill=(21, 27, 35, 255),
            outline=(*style.accent, 255),
            width=3,
        )
        draw.text(
            (badge_x + 16, badge_y + 8),
            style.title,
            font=phase_font,
            fill=(*style.accent, 255),
        )
        max_chars = max(8, (panel_w - 48) // max(13, self.width // 68))
        subtitle_lines = [
            style.subtitle[index : index + max_chars] for index in range(0, len(style.subtitle), max_chars)
        ]
        for line_index, line in enumerate(subtitle_lines[:3]):
            draw.text(
                (badge_x + 16, badge_y + 39 + line_index * 22),
                line,
                font=small_font,
                fill=(235, 237, 241, 255),
            )

        legend_y = min(route_y - 33, badge_y + badge_h + 15)
        draw.ellipse(
            (scene_w + 22, legend_y, scene_w + 36, legend_y + 14),
            fill=(239, 56, 61, 255),
        )
        draw.text(
            (scene_w + 43, legend_y - 3),
            "A · 右前方",
            font=tiny_font,
            fill=(235, 237, 241, 255),
        )
        blue_x = scene_w + panel_w // 2 + 12
        draw.ellipse(
            (blue_x, legend_y, blue_x + 14, legend_y + 14),
            fill=(45, 95, 244, 255),
        )
        draw.text(
            (blue_x + 21, legend_y - 3),
            "B · 左前方",
            font=tiny_font,
            fill=(235, 237, 241, 255),
        )

        draw.rectangle((0, route_y, self.width, self.height), fill=(12, 16, 21, 225))
        route = [
            ("红区取杯", (239, 56, 61)),
            ("中央交接", (226, 188, 73)),
            ("左手取杯", (92, 176, 255)),
            ("蓝区放置", (45, 95, 244)),
        ]
        margin = 82
        gap = (self.width - 2 * margin) / (len(route) - 1)
        for index in range(len(route) - 1):
            x0 = margin + index * gap
            x1 = margin + (index + 1) * gap
            line_color = (117, 211, 156, 235) if index < style.route_step else (103, 111, 124, 180)
            draw.line((x0, route_y + 23, x1, route_y + 23), fill=line_color, width=5)
        for index, (label, color) in enumerate(route):
            x = margin + index * gap
            active = index == style.route_step
            radius = 11 if active else 8
            draw.ellipse(
                (x - radius, route_y + 23 - radius, x + radius, route_y + 23 + radius),
                fill=(*color, 255),
                outline=(255, 255, 255, 255) if active else None,
                width=3 if active else 1,
            )
            text_box = draw.textbbox((0, 0), label, font=tiny_font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x - text_width / 2, route_y + 37),
                label,
                font=tiny_font,
                fill=(242, 244, 248, 255),
            )

        overall = min(1.0, (self._frame_index + local_progress) / self._total_frames)
        draw.rectangle(
            (0, self.height - 4, self.width * overall, self.height),
            fill=(*style.accent, 255),
        )
        return canvas.convert("RGB")

    def _emit_frame(self, style: PhaseStyle, local_progress: float) -> Image.Image:
        front = self._render_raw("mission_front_camera")
        overhead = self._render_raw("mission_overhead_camera")
        frame = self._overlay(front, overhead, style, local_progress)
        self._writer.append_data(np.asarray(frame))
        if self.write_gif and self._frame_index % self._gif_stride == 0:
            gif_width = min(560, self.width)
            gif_height = round(self.height * gif_width / self.width)
            self._gif_frames.append(frame.resize((gif_width, gif_height), Image.Resampling.LANCZOS))
        self._last_frame = frame
        self._frame_index += 1
        return frame

    def _phase(
        self,
        style: PhaseStyle,
        seconds: float,
        *,
        positions: dict[str, tuple[float, float, float]] | None = None,
        joint_targets: dict[str, np.ndarray] | None = None,
        grippers: dict[str, float] | None = None,
        capture: str | None = None,
    ) -> None:
        frame_count = max(1, round(seconds * self.duration_scale * self.fps))
        positions = positions or {}
        joint_targets = dict(joint_targets or {})
        grippers = grippers or {}
        for side, target in positions.items():
            joint_targets[side] = self._solve_position(side, target)

        start_joints = {side: np.array(self.controller.target_joint_qpos[side], copy=True) for side in joint_targets}
        start_grippers = {side: self.controller.target_gripper[side] for side in grippers}
        captured_frame: Image.Image | None = None
        for frame_number in range(frame_count):
            progress = 1.0 if frame_count == 1 else frame_number / (frame_count - 1)
            interpolation = _smoothstep(progress)
            for side, target in joint_targets.items():
                self.controller.target_joint_qpos[side] = (1.0 - interpolation) * start_joints[
                    side
                ] + interpolation * target
            for side, target in grippers.items():
                self.controller.target_gripper[side] = (1.0 - interpolation) * start_grippers[
                    side
                ] + interpolation * target
            self._step_physics()
            captured_frame = self._emit_frame(style, progress)
        if capture is not None and captured_frame is not None:
            self._storyboard_frames[capture] = captured_frame.copy()

    def _write_storyboard(self) -> None:
        labels = [
            ("right_pick", "① 右手从红区取杯"),
            ("center", "② 杯子放到中央交接位"),
            ("left_place", "③ 左手放入蓝区"),
        ]
        available = [(self._storyboard_frames[key], label) for key, label in labels if key in self._storyboard_frames]
        if not available:
            return
        panel_width = 480
        panel_height = round(self.height * panel_width / self.width)
        title_height = 62
        board = Image.new(
            "RGB",
            (panel_width * len(available), panel_height + title_height),
            (15, 18, 23),
        )
        draw = ImageDraw.Draw(board)
        label_font = _font(24, bold=True)
        for index, (frame, label) in enumerate(available):
            panel = frame.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
            board.paste(panel, (index * panel_width, title_height))
            draw.text(
                (index * panel_width + 18, 15),
                label,
                font=label_font,
                fill=(245, 247, 251),
            )
        board.save(self.storyboard_path)

    def _write_outputs(self) -> None:
        if self.write_gif and self._gif_frames:
            duration_ms = round(1000 * self._gif_stride / self.fps)
            self._gif_frames[0].save(
                self.gif_path,
                save_all=True,
                append_images=self._gif_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=False,
            )
        self._write_storyboard()
        final_position = self.mission.cup_position()
        blue = np.asarray(self.mission.config.region_b_center)
        summary = {
            "robot": "OpenArm v1 bimanual",
            "task": "right red pick -> center place -> left center pick -> blue place",
            "cup": "handleless tapered disposable paper cup",
            "cup_dimensions_mm": {
                "top_diameter": round(2000 * self.mission.config.cup_top_radius, 3),
                "bottom_diameter": round(
                    2000 * self.mission.config.cup_bottom_radius,
                    3,
                ),
                "height": round(2000 * self.mission.config.cup_half_height, 3),
            },
            "gripper_max_opening_mm": round(
                2000 * self.mission.config.open_finger_qpos,
                3,
            ),
            "frames": self._frame_index,
            "fps": self.fps,
            "duration_seconds": round(self._frame_index / self.fps, 3),
            "final_cup_position": final_position.round(6).tolist(),
            "blue_region_center": blue.tolist(),
            "final_xy_error_m": round(float(np.linalg.norm(final_position[:2] - blue)), 6),
            "success": bool(np.linalg.norm(final_position[:2] - blue) <= self.mission.config.region_radius),
            "grasp_mode": "deterministic scripted cup latch for P0-P2 visualization",
        }
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def run(self) -> None:
        config = self.mission.config
        table_z = config.table_top_z
        grasp_z = config.cup_grasp_tcp_z
        travel_z = table_z + 0.31
        red = config.region_a_center
        center = config.handoff_center
        blue = config.region_b_center
        open_grip = self.controller.config.gripper_open_position
        closed_grip = 0.008

        idle = PhaseStyle(
            "任务预览",
            "红区位于机器人右前方, 蓝区位于左前方",
            0,
            (242, 83, 89),
        )
        right_move = PhaseStyle(
            "右手取杯",
            "右臂接近红色区域 A 的一次性纸杯",
            0,
            (242, 83, 89),
        )
        right_transfer = PhaseStyle(
            "右手搬运",
            "保持纸杯直立, 移动到桌面中央交接位",
            1,
            (232, 190, 72),
        )
        center_place = PhaseStyle(
            "中央放置",
            "右手松开纸杯并退出交接空间",
            1,
            (232, 190, 72),
        )
        left_move = PhaseStyle(
            "左手接力",
            "左臂从中央交接位重新夹取纸杯",
            2,
            (99, 183, 255),
        )
        left_transfer = PhaseStyle(
            "左手搬运",
            "左臂将纸杯移动到左前方蓝色区域 B",
            3,
            (71, 111, 255),
        )
        success = PhaseStyle(
            "任务完成",
            "纸杯已直立放入蓝色目标区域",
            3,
            (91, 214, 147),
        )

        try:
            self._phase(idle, 0.8)
            self._phase(
                right_move,
                1.0,
                positions={"right": (red[0], red[1], travel_z)},
                grippers={"right": open_grip},
            )
            self._phase(
                right_move,
                0.7,
                positions={"right": (red[0], red[1], grasp_z)},
            )
            self._phase(
                right_move,
                0.55,
                grippers={"right": closed_grip},
            )
            self._attach("right")
            self._phase(
                right_move,
                0.8,
                positions={"right": (red[0], red[1], travel_z)},
                capture="right_pick",
            )
            self._phase(
                right_transfer,
                1.25,
                positions={"right": (center[0], center[1], travel_z)},
            )
            self._phase(
                right_transfer,
                0.7,
                positions={"right": (center[0], center[1], grasp_z)},
            )
            self._phase(
                center_place,
                0.5,
                grippers={"right": open_grip},
            )
            self._release_at(center)
            self._phase(center_place, 0.45, capture="center")
            self._phase(
                center_place,
                0.75,
                positions={"right": (center[0], center[1], travel_z)},
            )
            self._phase(
                center_place,
                0.8,
                joint_targets={"right": self.home_qpos["right"]},
            )
            self._phase(
                left_move,
                1.0,
                positions={"left": (center[0], center[1], travel_z)},
                grippers={"left": open_grip},
            )
            self._phase(
                left_move,
                0.7,
                positions={"left": (center[0], center[1], grasp_z)},
            )
            self._phase(
                left_move,
                0.55,
                grippers={"left": closed_grip},
            )
            self._attach("left")
            self._phase(
                left_move,
                0.8,
                positions={"left": (center[0], center[1], travel_z)},
            )
            self._phase(
                left_transfer,
                1.25,
                positions={"left": (blue[0], blue[1], travel_z)},
            )
            self._phase(
                left_transfer,
                0.7,
                positions={"left": (blue[0], blue[1], grasp_z)},
            )
            self._phase(
                left_transfer,
                0.5,
                grippers={"left": open_grip},
            )
            self._release_at(blue)
            self._phase(success, 0.5, capture="left_place")
            self._phase(
                success,
                0.75,
                positions={"left": (blue[0], blue[1], travel_z)},
            )
            self._phase(
                success,
                0.8,
                joint_targets={"left": self.home_qpos["left"]},
            )
            self._phase(success, 1.2)
        finally:
            self._writer.close()
            close = getattr(self.renderer, "close", None)
            if close is not None:
                close()
            else:
                gl_context = getattr(self.renderer, "_gl_context", None)
                if gl_context is not None:
                    gl_context.free()

        self._write_outputs()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the OpenArm v1 bimanual paper-cup relay showcase.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument(
        "--duration-scale",
        type=float,
        default=1.0,
        help="Scale every phase duration; 0.4 is useful for a quick preview.",
    )
    parser.add_argument(
        "--no-gif",
        action="store_true",
        help="Only write MP4, storyboard and JSON summary.",
    )
    args = parser.parse_args()

    demo = RelayDemo(
        output_dir=args.output_dir,
        fps=args.fps,
        width=args.width,
        height=args.height,
        duration_scale=args.duration_scale,
        write_gif=not args.no_gif,
    )
    demo.run()
    print(f"MP4: {demo.mp4_path}")
    if not args.no_gif:
        print(f"GIF: {demo.gif_path}")
    print(f"storyboard: {demo.storyboard_path}")
    print(f"summary: {demo.summary_path}")


if __name__ == "__main__":
    main()
