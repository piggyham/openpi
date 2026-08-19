"""Trim the leading duration from one OpenArm dataset-format v0.3.0 episode."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil

import polars as pl


def _timestamp_ns(series: pl.Series) -> list[int]:
    return series.cast(pl.Int64).to_list()


def trim_episode(dataset: Path, episode_id: str, seconds: float) -> tuple[Path, dict[str, tuple[int, int]]]:
    """Back up and trim an episode, returning backup path and per-stream counts."""
    if seconds <= 0:
        raise ValueError("seconds must be positive")

    episode = dataset / "episodes" / episode_id
    if not episode.is_dir():
        raise FileNotFoundError(f"episode directory not found: {episode}")

    parquet_paths = sorted(episode.glob("**/*.parquet"))
    camera_paths = sorted(episode.glob("cameras/*/*.jpeg"))
    if not parquet_paths:
        raise ValueError(f"episode contains no parquet streams: {episode}")

    starts_ns: list[int] = []
    tables: dict[Path, pl.DataFrame] = {}
    for path in parquet_paths:
        table = pl.read_parquet(path)
        if "timestamp" not in table.columns or table.is_empty():
            raise ValueError(f"missing or empty timestamp column: {path}")
        tables[path] = table
        starts_ns.append(_timestamp_ns(table.get_column("timestamp"))[0])
    for path in camera_paths:
        try:
            starts_ns.append(int(path.stem))
        except ValueError as exc:
            raise ValueError(f"camera filename is not a nanosecond timestamp: {path}") from exc

    cutoff_ns = min(starts_ns) + round(seconds * 1_000_000_000)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    backup = dataset / "backups" / f"episode_{episode_id}_before_trim_{stamp}"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(episode, backup, copy_function=os.link)

    counts: dict[str, tuple[int, int]] = {}
    for path, table in tables.items():
        timestamps = table.get_column("timestamp").cast(pl.Int64)
        trimmed = table.filter(timestamps >= cutoff_ns)
        if trimmed.is_empty():
            raise ValueError(f"trim would remove every row from {path}")
        temporary = path.with_suffix(path.suffix + ".trim-tmp")
        trimmed.write_parquet(temporary)
        os.replace(temporary, path)
        counts[str(path.relative_to(episode))] = (len(table), len(trimmed))

    for camera_dir in sorted((episode / "cameras").glob("*")):
        if not camera_dir.is_dir():
            continue
        frames = sorted(camera_dir.glob("*.jpeg"), key=lambda path: int(path.stem))
        kept = [path for path in frames if int(path.stem) >= cutoff_ns]
        if frames and not kept:
            raise ValueError(f"trim would remove every camera frame from {camera_dir}")
        for path in frames:
            if int(path.stem) < cutoff_ns:
                path.unlink()
        counts[str(camera_dir.relative_to(episode))] = (len(frames), len(kept))

    return backup, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", required=True, help="episode directory name, for example 0")
    parser.add_argument("--seconds", type=float, required=True)
    args = parser.parse_args()

    backup, counts = trim_episode(args.dataset.resolve(), args.episode, args.seconds)
    print(f"backup: {backup}")
    for stream, (before, after) in counts.items():
        print(f"{stream}: {before} -> {after}")


if __name__ == "__main__":
    main()
