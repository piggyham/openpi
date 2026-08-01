"""Convert P5 raw episodes to a local LeRobot v2 dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
from typing import Any

from openarm_mission.dataset import ACTION_NAMES
from openarm_mission.dataset import CAMERAS
from openarm_mission.dataset import STATE_NAMES
from openarm_mission.dataset import TASK_INSTRUCTION
from openarm_mission.dataset import load_episode


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def convert_dataset(
    *,
    source_dir: Path,
    output_dir: Path,
    repo_id: str = "local/openarm_paper_cup_relay",
    overwrite: bool = False,
) -> dict[str, Any]:
    os.environ.setdefault(
        "HF_HOME",
        str((output_dir.parent / ".hf_cache").resolve()),
    )
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    manifest = _load_json(source_dir / "manifest.json")
    schema = _load_json(source_dir / "schema.json")
    if not manifest["all_valid"]:
        raise RuntimeError("Raw P5 manifest contains invalid trajectories")
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_dir)

    height, width = schema["images"]["front"]["shape"][:2]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [list(STATE_NAMES)],
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": [list(ACTION_NAMES)],
        },
        "observation.cup_pose": {
            "dtype": "float32",
            "shape": (7,),
            "names": [
                [
                    "x_m",
                    "y_m",
                    "z_m",
                    "qw",
                    "qx",
                    "qy",
                    "qz",
                ]
            ],
        },
        **{
            f"observation.images.{key}": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            }
            for key in CAMERAS
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="openarm_v1_bimanual",
        fps=int(manifest["fps"]),
        features=features,
        use_videos=False,
        image_writer_threads=4,
    )

    image_records = [record for record in manifest["records"] if record.get("capture_images", True)]
    if not image_records:
        raise RuntimeError("No image-bearing raw episodes are available")
    records_by_seed = {int(record["seed"]): record for record in image_records}
    ordered_seeds: list[int] = []
    split_ranges: dict[str, str] = {}
    source_map: list[dict[str, Any]] = []
    start = 0
    for split in ("train", "validation", "test"):
        seeds = [seed for seed in manifest["splits"][split]["seeds"] if seed in records_by_seed]
        ordered_seeds.extend(seeds)
        stop = start + len(seeds)
        split_ranges[split] = f"{start}:{stop}"
        start = stop

    for episode_index, seed in enumerate(ordered_seeds):
        record = records_by_seed[seed]
        trajectory = load_episode(Path(record["path"]))
        for frame_index in range(len(trajectory["timestamp"])):
            dataset.add_frame(
                {
                    "observation.state": trajectory["state"][frame_index],
                    "action": trajectory["action"][frame_index],
                    "observation.cup_pose": trajectory["cup_pose"][frame_index],
                    **{f"observation.images.{key}": trajectory[f"image_{key}"][frame_index] for key in CAMERAS},
                    "task": TASK_INSTRUCTION,
                }
            )
        dataset.save_episode()
        source_map.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "split": record["split"],
                "raw_path": record["path"],
                "raw_sha256": record["sha256"],
            }
        )
        if (episode_index + 1) % 10 == 0:
            print(f"conversion: {episode_index + 1}/{len(ordered_seeds)}")

    info_path = output_dir / "meta" / "info.json"
    info = _load_json(info_path)
    info["splits"] = split_ranges
    info_path.write_text(
        json.dumps(info, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    mapping_path = output_dir / "meta" / "openarm_source_map.json"
    mapping_path.write_text(
        json.dumps(source_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = LeRobotDataset(repo_id=repo_id, root=output_dir)
    report = {
        "repo_id": repo_id,
        "root": str(output_dir),
        "episodes": loaded.num_episodes,
        "frames": loaded.num_frames,
        "fps": loaded.fps,
        "features": sorted(loaded.features),
        "splits": split_ranges,
        "source_total_episodes": manifest["episodes"],
        "multimodal_image_episodes": len(ordered_seeds),
        "load_validation": (
            loaded.num_episodes == len(ordered_seeds)
            and loaded.num_frames == sum(records_by_seed[seed]["frames"] for seed in ordered_seeds)
        ),
    }
    report_path = output_dir / "meta" / "openarm_conversion.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert OpenArm P5 raw data to LeRobot v2.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p5/lerobot/openarm_paper_cup_relay"),
    )
    parser.add_argument(
        "--repo-id",
        default="local/openarm_paper_cup_relay",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = convert_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        overwrite=args.overwrite,
    )
    print(
        f"LeRobot conversion: episodes={report['episodes']}, "
        f"frames={report['frames']}, valid={report['load_validation']}"
    )


if __name__ == "__main__":
    main()
