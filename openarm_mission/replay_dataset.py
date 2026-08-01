"""Replay and inspect one P5 raw trajectory as an annotated video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw

from openarm_mission.dataset import load_episode
from openarm_mission.dataset import validate_trajectory


def replay_episode(
    *,
    episode_path: Path,
    output_path: Path,
    fps: int = 10,
) -> dict:
    trajectory = load_episode(episode_path)
    validation = validate_trajectory(trajectory, fps=fps)
    if not validation["valid"]:
        raise RuntimeError(f"Cannot replay invalid trajectory: {validation}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )
    try:
        for index, timestamp in enumerate(trajectory["timestamp"]):
            front = Image.fromarray(trajectory["image_front"][index])
            left = Image.fromarray(trajectory["image_left_wrist"][index])
            right = Image.fromarray(trajectory["image_right_wrist"][index])
            width = front.width * 2
            header = 42
            canvas = Image.new(
                "RGB",
                (width, front.height * 2 + header),
                (17, 20, 25),
            )
            canvas.paste(
                front,
                ((width - front.width) // 2, header),
            )
            canvas.paste(left, (0, header + front.height))
            canvas.paste(
                right,
                (front.width, header + front.height),
            )
            draw = ImageDraw.Draw(canvas)
            action_norm = float(np.linalg.norm(trajectory["action"][index]))
            draw.text(
                (10, 7),
                (f"frame={index:04d}  t={timestamp:05.2f}s  {trajectory['stage'][index]}  |action|={action_norm:.3f}"),
                fill=(235, 238, 242),
            )
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()

    report = {
        "episode": str(episode_path),
        "video": str(output_path),
        "validation": validation,
        "frames_replayed": len(trajectory["timestamp"]),
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay one OpenArm P5 raw trajectory.")
    parser.add_argument("episode_path", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("openarm_mission/artifacts/p5/replay.mp4"),
    )
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    report = replay_episode(
        episode_path=args.episode_path,
        output_path=args.output,
        fps=args.fps,
    )
    print(f"Replay: frames={report['frames_replayed']}, video={report['video']}")


if __name__ == "__main__":
    main()
