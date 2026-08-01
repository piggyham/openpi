"""Run and record P3 contact-gated OpenArm paper-cup relay episodes."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

from openarm_mission.controller import BimanualCartesianController
from openarm_mission.model import OpenArmMission
from openarm_mission.task import OpenArmRelayTask
from openarm_mission.task import RelayStage


def _font(size: int, *, bold: bool = False):
    names = (
        "NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Medium.ttc",
        "DroidSansFallbackFull.ttf",
        "DejaVuSans.ttf",
    )
    roots = (
        Path("/usr/share/fonts/opentype/noto"),
        Path("/usr/share/fonts/truetype/droid"),
        Path("/usr/share/fonts/truetype/dejavu"),
    )
    for index, root in enumerate(roots):
        name = names[index]
        path = root / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _yaw_from_quaternion(quaternion: np.ndarray) -> float:
    w, x, y, z = quaternion
    return float(
        np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
    )


class P3EpisodeRunner:
    """Physical episode runner with optional annotated video output."""

    def __init__(
        self,
        *,
        mode: str,
        seed: int,
        output_dir: Path,
        fps: int,
        width: int,
        height: int,
        write_video: bool,
        artifact_prefix: str | None = None,
        visual_title: str = "OpenArm v1 · P3 接触门控双臂接力",
        visual_subtitle: str = ("双指接触 → MuJoCo weld → 物理释放 → 连续 0.5 s 成功保持"),
        mission: OpenArmMission | None = None,
        task: OpenArmRelayTask | None = None,
    ):
        if mode not in {"success", "failure"}:
            raise ValueError("mode must be success or failure")
        self.mode = mode
        self.seed = seed
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.width = width
        self.height = height
        self.write_video = write_video
        self.visual_title = visual_title
        self.visual_subtitle = visual_subtitle

        self.mission = mission or OpenArmMission()
        self.task = task or OpenArmRelayTask(self.mission)
        if self.task.mission is not self.mission:
            raise ValueError("Task and episode runner must share one mission")
        self.task.reset(seed)
        self.controller = BimanualCartesianController(self.mission)
        self.home_qpos = {
            side: np.array(
                self.mission.data.qpos[arm.qpos_indices],
                copy=True,
            )
            for side, arm in self.mission.arms.items()
        }
        self._frame_index = 0
        self._phase_title = "初始化"
        self._phase_subtitle = "随机化纸杯位姿、质量和摩擦"
        self._accent = (242, 83, 89)

        stem = artifact_prefix or f"p3_{mode}_seed{seed:03d}"
        self.video_path = self.output_dir / f"{stem}.mp4"
        self.summary_path = self.output_dir / f"{stem}.json"
        self.renderer = None
        self.writer = None
        if self.write_video:
            self.renderer = mujoco.Renderer(
                self.mission.model,
                height=self.height,
                width=self.width,
            )
            self.writer = imageio.get_writer(
                self.video_path,
                fps=self.fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )

    def _render_camera(self, camera: str) -> Image.Image:
        assert self.renderer is not None
        self.renderer.update_scene(self.mission.data, camera=camera)
        return Image.fromarray(self.renderer.render()).convert("RGBA")

    def _render_frame(self) -> None:
        if not self.write_video:
            return
        front = self._render_camera("mission_front_camera")
        overhead = self._render_camera("mission_overhead_camera")

        header_h = max(72, self.height // 9)
        footer_h = 50
        panel_w = max(245, self.width // 4)
        scene_w = self.width - panel_w
        content_h = self.height - header_h - footer_h
        canvas = Image.new("RGBA", (self.width, self.height), (12, 16, 21, 255))
        scene = ImageOps.contain(
            front,
            (scene_w, content_h),
            Image.Resampling.LANCZOS,
        )
        canvas.alpha_composite(
            scene,
            (
                (scene_w - scene.width) // 2,
                header_h + (content_h - scene.height) // 2,
            ),
        )
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle((0, 0, self.width, header_h), fill=(12, 16, 21, 255))
        draw.text(
            (22, 10),
            self.visual_title,
            font=_font(max(22, self.width // 36), bold=True),
            fill=(248, 249, 252, 255),
        )
        draw.text(
            (23, 44),
            self.visual_subtitle,
            font=_font(max(13, self.width // 70)),
            fill=(190, 198, 210, 255),
        )
        draw.line(
            (scene_w, header_h, scene_w, self.height - footer_h),
            fill=(85, 94, 108, 220),
            width=2,
        )

        inset_w = panel_w - 24
        inset_h = round(inset_w * 0.62)
        inset = overhead.resize((inset_w, inset_h), Image.Resampling.LANCZOS)
        inset_x = scene_w + 12
        inset_y = header_h + 12
        canvas.alpha_composite(inset, (inset_x, inset_y))
        draw = ImageDraw.Draw(canvas, "RGBA")
        draw.rectangle(
            (inset_x, inset_y, inset_x + inset_w, inset_y + inset_h),
            outline=(230, 234, 241, 255),
            width=2,
        )

        card_y = inset_y + inset_h + 14
        card_h = min(132, self.height - footer_h - card_y - 90)
        draw.rounded_rectangle(
            (inset_x, card_y, inset_x + inset_w, card_y + card_h),
            radius=12,
            fill=(21, 27, 35, 255),
            outline=(*self._accent, 255),
            width=3,
        )
        draw.text(
            (inset_x + 14, card_y + 10),
            self._phase_title,
            font=_font(max(18, self.width // 50), bold=True),
            fill=(*self._accent, 255),
        )
        subtitle = self._phase_subtitle
        for line_index in range(3):
            line = subtitle[line_index * 12 : (line_index + 1) * 12]
            if not line:
                break
            draw.text(
                (inset_x + 14, card_y + 43 + line_index * 21),
                line,
                font=_font(max(13, self.width // 72)),
                fill=(235, 237, 241, 255),
            )

        status_y = card_y + card_h + 15
        status_font = _font(max(12, self.width // 78))
        status_lines = [
            f"阶段: {self.task.stage.value}",
            f"约束: {self.task.summary()['active_weld'] or 'none'}",
            f"倾角: {self.task.cup_upright_angle_deg():.1f}°",
            (f"接触 R/L: {len(self.task.finger_contacts('right'))}/{len(self.task.finger_contacts('left'))}"),
        ]
        for index, text in enumerate(status_lines):
            draw.text(
                (inset_x + 5, status_y + index * 21),
                text,
                font=status_font,
                fill=(205, 211, 222, 255),
            )

        footer_y = self.height - footer_h
        terminal_color = (
            (91, 214, 147)
            if self.task.success
            else (242, 83, 89)
            if self.task.stage is RelayStage.FAILURE
            else self._accent
        )
        draw.rectangle((0, footer_y, self.width, self.height), fill=(12, 16, 21, 255))
        draw.text(
            (22, footer_y + 12),
            (f"t={self.task.elapsed:05.2f}s   seed={self.seed}   mode={self.mode}   state={self.task.stage.value}"),
            font=_font(max(13, self.width // 70), bold=True),
            fill=(*terminal_color, 255),
        )
        progress = min(1.0, self.task.elapsed / self.task.config.timeout_seconds)
        draw.rectangle(
            (0, self.height - 4, self.width * progress, self.height),
            fill=(*terminal_color, 255),
        )
        assert self.writer is not None
        self.writer.append_data(np.asarray(canvas.convert("RGB")))
        self._frame_index += 1

    def _step_frame(self) -> None:
        substeps = max(
            1,
            round(1.0 / (self.fps * self.mission.config.timestep)),
        )
        for _ in range(substeps):
            self.controller.compute_ctrl()
            mujoco.mj_step(self.mission.model, self.mission.data)
            self.task.update()
        self._render_frame()

    def _set_phase(
        self,
        title: str,
        subtitle: str,
        accent: tuple[int, int, int],
    ) -> None:
        self._phase_title = title
        self._phase_subtitle = subtitle
        self._accent = accent

    def _hold(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds * self.fps))):
            self._step_frame()

    def _move_joints(
        self,
        side: str,
        target_qpos: np.ndarray,
        seconds: float,
    ) -> None:
        start = np.array(self.controller.target_joint_qpos[side], copy=True)
        frame_count = max(1, round(seconds * self.fps))
        for frame in range(frame_count):
            interpolation = _smoothstep((frame + 1) / frame_count)
            self.controller.target_joint_qpos[side] = (1.0 - interpolation) * start + interpolation * target_qpos
            self._step_frame()

    def _move_tcp(
        self,
        side: str,
        position: np.ndarray,
        seconds: float,
        quaternion: np.ndarray | None = None,
    ) -> None:
        result = self.controller.solve_ik(side, position, quaternion)
        if not result.converged:
            raise RuntimeError(
                f"{side} IK failed: position={position.tolist()}, "
                f"position_error={result.position_error:.4f}, "
                f"orientation_error={result.orientation_error:.4f}"
            )
        self._move_joints(side, result.qpos, seconds)

    def _set_gripper(self, side: str, target: float, seconds: float) -> None:
        start = self.controller.target_gripper[side]
        frame_count = max(1, round(seconds * self.fps))
        for frame in range(frame_count):
            interpolation = _smoothstep((frame + 1) / frame_count)
            self.controller.target_gripper[side] = (1.0 - interpolation) * start + interpolation * target
            self._step_frame()

    def _find_upright_target(
        self,
        side: str,
        cup_position: np.ndarray,
        preferred_yaw: float,
    ) -> tuple[np.ndarray, float]:
        current = np.asarray(self.mission.data.qpos[self.mission.arms[side].qpos_indices])
        candidates: list[tuple[float, np.ndarray, float]] = []
        yaw_offsets = np.linspace(-np.pi, np.pi, 33)
        for offset in yaw_offsets:
            yaw = preferred_yaw + float(offset)
            tcp_position, tcp_quaternion = self.task.upright_tcp_pose(
                side,
                cup_position,
                yaw,
            )
            result = self.controller.solve_ik(
                side,
                tcp_position,
                tcp_quaternion,
            )
            if result.converged:
                joint_distance = float(np.linalg.norm(result.qpos - current))
                candidates.append((joint_distance, result.qpos, yaw))
        if not candidates:
            raise RuntimeError(f"No upright welded-cup IK solution for {side} at {cup_position.tolist()}")
        _, qpos, yaw = min(candidates, key=lambda candidate: candidate[0])
        return qpos, yaw

    def _approach_and_attach(self, side: str) -> float:
        cup_position = self.mission.cup_position()
        self._move_tcp(
            side,
            np.array([cup_position[0], cup_position[1], 0.50]),
            0.8,
        )
        cup_position = self.mission.cup_position()
        grasp_tcp = np.array(
            [
                cup_position[0],
                cup_position[1],
                cup_position[2] + self.task.config.physical_grasp_tcp_offset_z,
            ]
        )
        self._move_tcp(side, grasp_tcp, 0.7)
        self._set_gripper(side, 0.008, 0.65)
        attached, details = self.task.try_attach(side)
        if not attached:
            raise RuntimeError(f"{side} contact-gated grasp rejected: {details}")
        return _yaw_from_quaternion(self.mission.cup_quaternion())

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
                self.mission.config.table_top_z + 0.22,
            ]
        )
        qpos, yaw = self._find_upright_target(side, cup_target, cup_yaw)
        self._move_joints(side, qpos, 0.85)
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
                self.mission.config.table_top_z + 0.22,
            ]
        )
        high_qpos, high_yaw = self._find_upright_target(
            side,
            high_target,
            cup_yaw,
        )
        self._move_joints(side, high_qpos, 1.1)

        place_target = np.array(
            [
                target_xy[0],
                target_xy[1],
                self.mission.config.table_top_z + self.mission.config.cup_half_height + 0.002,
            ]
        )
        place_qpos, place_yaw = self._find_upright_target(
            side,
            place_target,
            high_yaw,
        )
        self._move_joints(side, place_qpos, 0.85)
        self._hold(0.35)
        return place_yaw

    def _release_and_retreat(
        self,
        side: str,
        retreat_xy: tuple[float, float],
    ) -> None:
        self._set_gripper(
            side,
            self.controller.config.gripper_open_position,
            0.5,
        )
        self._hold(0.25)
        released, details = self.task.release(side)
        if not released:
            raise RuntimeError(f"{side} physical release rejected: {details}")
        self._move_tcp(
            side,
            np.array([retreat_xy[0], retreat_xy[1], 0.52]),
            0.65,
        )
        self._move_joints(side, self.home_qpos[side], 0.75)

    def _run_success(self) -> None:
        red = self.mission.config.region_a_center
        center = self.mission.config.handoff_center
        blue = self.mission.config.region_b_center

        self._set_phase("任务初始化", "A 区随机化纸杯并等待稳定", (242, 83, 89))
        self._hold(0.55)

        self._set_phase("右手接触夹取", "双指接触成立后激活右手 weld", (242, 83, 89))
        right_yaw = self._approach_and_attach("right")
        right_yaw = self._lift_welded("right", red, right_yaw)

        self._set_phase("中央物理放置", "保持杯体直立并释放右手约束", (232, 190, 72))
        self._place_welded("right", center, right_yaw)
        self._release_and_retreat("right", center)
        self._hold(0.45)
        if self.task.stage is not RelayStage.WAIT_LEFT_GRASP:
            raise RuntimeError(f"Center handoff did not stabilize: {self.task.summary()}")

        self._set_phase("左手接触接力", "中央双指接触成立后激活左手 weld", (99, 183, 255))
        left_yaw = self._approach_and_attach("left")
        left_yaw = self._lift_welded("left", center, left_yaw)

        self._set_phase("蓝区物理放置", "解除左手约束并连续验证 0.5 秒", (71, 111, 255))
        self._place_welded("left", blue, left_yaw)
        self._release_and_retreat("left", blue)

        self._set_phase("成功保持判定", "蓝区、直立、双手释放、桌面接触", (91, 214, 147))
        self._hold(0.85)
        if not self.task.success:
            raise RuntimeError(f"P3 episode did not succeed: {self.task.summary()}")

    def _run_failure(self) -> None:
        red = self.mission.config.region_a_center
        self._set_phase("失败用例初始化", "使用同一套接触门控执行右手夹取", (242, 83, 89))
        self._hold(0.45)
        cup_yaw = self._approach_and_attach("right")
        self._lift_welded("right", red, cup_yaw)
        self._set_phase("注入抓取丢失", "抬升后解除 weld, 验证失败检测", (242, 83, 89))
        self.task.inject_grasp_loss("right")
        self._set_gripper(
            "right",
            self.controller.config.gripper_open_position,
            0.35,
        )
        self._hold(0.8)
        if self.task.stage is not RelayStage.FAILURE:
            raise RuntimeError("Injected failure episode did not terminate")

    def run(self) -> dict:
        try:
            if self.mode == "success":
                self._run_success()
            else:
                self._run_failure()
        finally:
            if self.writer is not None:
                self.writer.close()
            if self.renderer is not None:
                close = getattr(self.renderer, "close", None)
                if close is not None:
                    close()
                else:
                    gl_context = getattr(self.renderer, "_gl_context", None)
                    if gl_context is not None:
                        gl_context.free()

        summary = self.task.summary()
        summary.update(
            {
                "mode": self.mode,
                "seed": self.seed,
                "frames": self._frame_index,
                "fps": self.fps,
                "video": str(self.video_path) if self.write_video else None,
                "task_config": asdict(self.task.config),
            }
        )
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run P3 OpenArm contact-gated relay episodes.")
    parser.add_argument(
        "--mode",
        choices=("success", "failure", "both"),
        default="both",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p3"),
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    if args.mode == "both" and not args.no_video:
        for mode in ("success", "failure"):
            command = [
                sys.executable,
                "-m",
                "openarm_mission.p3_episode",
                "--mode",
                mode,
                "--seed",
                str(args.seed),
                "--output-dir",
                str(args.output_dir),
                "--fps",
                str(args.fps),
                "--width",
                str(args.width),
                "--height",
                str(args.height),
            ]
            subprocess.run(command, check=True)
        return

    modes = ("success", "failure") if args.mode == "both" else (args.mode,)
    for mode in modes:
        runner = P3EpisodeRunner(
            mode=mode,
            seed=args.seed,
            output_dir=args.output_dir,
            fps=args.fps,
            width=args.width,
            height=args.height,
            write_video=not args.no_video,
        )
        summary = runner.run()
        print(f"{mode}: stage={summary['stage']}, success={summary['success']}, reason={summary['failure_reason']}")
        print(f"summary: {runner.summary_path}")
        if not args.no_video:
            print(f"video: {runner.video_path}")


if __name__ == "__main__":
    main()
