"""Collect P5 synchronized OpenArm demonstrations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from openarm_mission.dataset import DATASET_VERSION
from openarm_mission.dataset import record_episode
from openarm_mission.dataset import schema_dict


def _worker(arguments: tuple[int, str, int, int, int, bool]) -> dict[str, Any]:
    seed, output_dir, fps, width, height, capture_images = arguments
    return record_episode(
        seed=seed,
        output_dir=Path(output_dir),
        fps=fps,
        width=width,
        height=height,
        capture_images=capture_images,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def collect_dataset(
    *,
    output_dir: Path,
    episodes: int = 200,
    start_seed: int = 0,
    workers: int = 2,
    fps: int = 20,
    width: int = 160,
    height: int = 120,
    image_episodes: int = 20,
    resume: bool = False,
    max_new_episodes: int | None = None,
) -> dict[str, Any]:
    if episodes <= 0 or workers <= 0:
        raise ValueError("episodes and workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "schema.json",
        schema_dict(fps=fps, height=height, width=width),
    )

    started = time.perf_counter()
    seeds = list(range(start_seed, start_seed + episodes))
    image_seed_set = set(seeds[: max(0, min(image_episodes, episodes))])
    previous_records: dict[int, dict[str, Any]] = {}
    manifest_path = output_dir / "manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_records = {int(record["seed"]): record for record in previous["records"]}
    results: list[dict[str, Any]] = []
    arguments = []
    scheduled = 0
    for seed in seeds:
        needs_images = seed in image_seed_set
        previous = previous_records.get(seed)
        if (
            previous is not None
            and needs_images
            and not previous.get(
                "capture_images",
                False,
            )
        ):
            metadata_path = Path(previous["metadata_path"])
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("capture_images", False):
                    episode_path = Path(previous["path"])
                    previous = {
                        **previous,
                        "sha256": hashlib.sha256(episode_path.read_bytes()).hexdigest(),
                        "bytes": episode_path.stat().st_size,
                        "frames": metadata["validation"]["frames"],
                        "duration_seconds": metadata["validation"]["duration_seconds"],
                        "capture_images": True,
                        "validation": metadata["validation"],
                    }
        reusable = (
            previous is not None
            and Path(previous["path"]).is_file()
            and previous["validation"]["valid"]
            and (not needs_images or previous.get("capture_images", False))
        )
        deferred = max_new_episodes is not None and scheduled >= max_new_episodes and previous is not None
        if reusable or deferred:
            results.append(previous)
        else:
            arguments.append(
                (
                    seed,
                    str(output_dir),
                    fps,
                    width,
                    height,
                    needs_images,
                )
            )
            scheduled += 1
    reused = len(results)
    if reused:
        print(f"resume: reusing {reused}/{episodes} episodes")
    if workers == 1:
        for completed, item in enumerate(map(_worker, arguments), 1):
            results.append(item)
            print(f"progress: {reused + completed}/{episodes}, seed={item['seed']}")
    else:
        # max_tasks_per_child=1: each worker process handles exactly one
        # episode then exits, so the per-episode MuJoCo EGL render context is
        # fully released when the process dies. Without this, contexts leak
        # across episodes within long-lived workers and the GPU's EGL context
        # limit is hit (~14 contexts on this driver), failing with
        # EGL_BAD_ALLOC. Python 3.11+ required for max_tasks_per_child.
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as executor:
            futures = {executor.submit(_worker, argument): argument[0] for argument in arguments}
            for completed, future in enumerate(
                as_completed(futures),
                1,
            ):
                item = future.result()
                results.append(item)
                total_completed = reused + completed
                if completed % 10 == 0 or completed == len(arguments):
                    print(f"progress: {total_completed}/{episodes}, seed={item['seed']}")

    results.sort(key=lambda item: item["seed"])
    split_seeds = {
        split: [item["seed"] for item in results if item["split"] == split] for split in ("train", "validation", "test")
    }
    total_bytes = sum(item["bytes"] for item in results)
    all_valid = all(item["validation"]["valid"] for item in results)
    manifest = {
        "version": DATASET_VERSION,
        "episodes": episodes,
        "successful_episodes": len(results),
        "all_valid": all_valid,
        "fps": fps,
        "image_size": [height, width],
        "image_episodes": sum(bool(item.get("capture_images", False)) for item in results),
        "start_seed": start_seed,
        "splits": {
            key: {
                "count": len(value),
                "seeds": value,
            }
            for key, value in split_seeds.items()
        },
        "total_frames": sum(item["frames"] for item in results),
        "total_bytes": total_bytes,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "records": results,
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "splits.json", split_seeds)

    alignment = {
        "checked_episodes": min(20, len(results)),
        "seeds": [item["seed"] for item in results[:20]],
        "all_valid": all(item["validation"]["valid"] for item in results[:20]),
        "reports": [
            {
                "seed": item["seed"],
                **item["validation"],
            }
            for item in results[:20]
        ],
    }
    _write_json(output_dir / "alignment_20.json", alignment)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect P5 OpenArm friction demonstrations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p5"),
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument(
        "--image-episodes",
        type=int,
        default=20,
        help="Number of leading episodes with synchronized RGB images.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid episodes and upgrade episodes missing RGB.",
    )
    parser.add_argument(
        "--max-new-episodes",
        type=int,
        default=None,
        help="Limit work in one resumable invocation.",
    )
    args = parser.parse_args()

    manifest = collect_dataset(
        output_dir=args.output_dir,
        episodes=args.episodes,
        start_seed=args.start_seed,
        workers=args.workers,
        fps=args.fps,
        width=args.width,
        height=args.height,
        image_episodes=args.image_episodes,
        resume=args.resume,
        max_new_episodes=args.max_new_episodes,
    )
    print(
        f"P5 collection: {manifest['successful_episodes']}/"
        f"{manifest['episodes']} successful, "
        f"frames={manifest['total_frames']}, "
        f"valid={manifest['all_valid']}"
    )


if __name__ == "__main__":
    main()
