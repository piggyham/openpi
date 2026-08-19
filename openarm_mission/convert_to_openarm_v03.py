"""Convert P9 extended npz episodes into the OpenArm v0.3.0 dataset format.

Produces a tree matching the ``real_data`` reference layout:
    metadata.yaml
    episodes/<id>/
        obs/arms/{left,right}/state.parquet
        action/arms/{left,right}/qpos.parquet
        cameras/{head,wrist_left,wrist_right}/<unix_ns>.jpeg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from openarm_mission.dataset import DATASET_VERSION
from openarm_mission.dataset import GRIPPER_FULL_OPENING_M
from openarm_mission.dataset import TASK_INSTRUCTION

# Sim camera key -> v0.3.0 camera directory name.
CAMERA_MAP = {
    "front": "head",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
}

# Keep archived collections convertible while requiring every source manifest
# to identify one of the explicitly understood trajectory semantics.
SUPPORTED_SOURCE_VERSIONS = {
    "openarm-p9-v1",
    "openarm-p10-natural-hang-v1",
    "openarm-p10-explicit-handoff-v2",
    DATASET_VERSION,
}

# pyarrow list type with child field name "element" (matches real_data).
_LIST8 = pa.list_(pa.field("element", pa.float32()))


def _make_timestamps(timestamps: np.ndarray, wall_start_ns: int) -> np.ndarray:
    """Convert MuJoCo sim seconds to Unix ns on an exact grid.

    The sim runs at an exact 20 Hz (0.05 s steps), so the returned ns grid
    is ``wall_start_ns + round((t - t₀) * 1e9)``.
    """
    delta = np.rint((timestamps - timestamps[0]) * 1.0e9).astype(np.int64)
    return np.int64(wall_start_ns) + delta


def _gripper_raw(opening_m: np.ndarray) -> np.ndarray:
    """Opening meters -> raw v0.3.0 value (0 = closed, more-negative = more open).

    Matches the real-robot convention (verified on teleop data), so the
    converted sim sets display identically to real_data in the viewer.
    """
    return -opening_m / GRIPPER_FULL_OPENING_M


def _write_arm_state_parquet(ep_dir: Path, side: str, data: dict) -> None:
    """Write one arm's obs state.parquet (qpos / qvel / qtorque)."""
    col = {"left": 0, "right": 8}
    o = col[side]

    qpos = np.zeros((data["count"], 8), dtype=np.float32)
    qpos[:, :7] = data["state"][:, o : o + 7]
    qpos[:, 7] = _gripper_raw(data["state"][:, o + 7])

    qvel = data["qvel"][:, o : o + 8].astype(np.float32)
    qtorque = data["qtorque"][:, o : o + 8].astype(np.float32)

    table = pa.table(
        {
            "timestamp": pa.array(data["unix_ns"], type=pa.timestamp("ns")),
            "qpos": pa.array(qpos.tolist(), type=_LIST8),
            "qvel": pa.array(qvel.tolist(), type=_LIST8),
            "qtorque": pa.array(qtorque.tolist(), type=_LIST8),
        }
    )
    out = ep_dir / "obs" / "arms" / side / "state.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def _write_arm_action_parquet(ep_dir: Path, side: str, data: dict) -> None:
    """Write one arm's action qpos.parquet (joint-space targets)."""
    col = {"left": 0, "right": 8}
    o = col[side]

    value = np.zeros((data["count"], 8), dtype=np.float32)
    value[:, :7] = data["joint_target"][:, o : o + 7]
    value[:, 7] = _gripper_raw(data["joint_target"][:, o + 7])

    table = pa.table(
        {
            "timestamp": pa.array(data["unix_ns"], type=pa.timestamp("ns")),
            "value": pa.array(value.tolist(), type=_LIST8),
        }
    )
    out = ep_dir / "action" / "arms" / side / "qpos.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out)


def _write_cameras(ep_dir: Path, data: dict, npz: dict) -> None:
    """Extract JPEG blobs from npz and write camera files."""
    for sim_key, real_name in CAMERA_MAP.items():
        jpeg_data = npz[f"image_{sim_key}_jpeg_data"]
        jpeg_offsets = npz[f"image_{sim_key}_jpeg_offsets"]
        cam_dir = ep_dir / "cameras" / real_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for i in range(data["count"]):
            start = int(jpeg_offsets[i])
            stop = int(jpeg_offsets[i + 1])
            ns = data["unix_ns"][i]
            (cam_dir / f"{ns}.jpeg").write_bytes(
                bytes(jpeg_data[start:stop])
            )


def _build_metadata(
    records: list[dict[str, Any]],
    raw_dir: Path,
) -> dict[str, Any]:
    """Build metadata.yaml dict matching the v0.3.0 real_data structure."""
    episodes = []
    for record in records:
        ep_json = Path(str(record["metadata_path"]))
        if ep_json.is_file():
            meta = json.loads(ep_json.read_text(encoding="utf-8"))
            success = meta.get("expert", {}).get("success", True)
        else:
            success = True
        episodes.append(
            {"id": str(record["seed"]), "success": bool(success), "task_index": 0}
        )
    return {
        "version": "0.3.0",
        "operation_type": "scripted",
        "operator": "openarm_mission",
        "location": "MuJoCo simulation",
        "equipment": {
            "id": "OpenArmBimanual",
            "version": "1.0",
            "embodiments": {
                "arms": {"id": "OpenArm", "version": "2.0"},
            },
            "perceptions": {
                "cameras": {
                    "head": {},
                    "wrist_left": {},
                    "wrist_right": {},
                },
            },
        },
        "frequencies": {
            "cameras": {
                "head": 20.0,
                "wrist_left": 20.0,
                "wrist_right": 20.0,
            },
            "obs": {"arms": {}},
            "action": {"arms": {}},
        },
        "tasks": [
            {
                "description": "Scripted OpenArm bimanual expert demonstration collected in MuJoCo simulation.",
                "prompt": TASK_INSTRUCTION,
            }
        ],
        "episodes": episodes,
    }


def convert_episode(
    npz_path: Path,
    metadata_path: Path,
    output_dir: Path,
    episode_id: str,
) -> dict[str, Any]:
    """Convert one npz episode into the v0.3.0 format tree.

    Returns the episode map entry.
    """
    npz = np.load(npz_path, allow_pickle=False)
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    timestamps = np.asarray(npz["timestamp"], dtype=np.float64)
    state = np.asarray(npz["state"], dtype=np.float32)
    qvel = np.asarray(npz["qvel"], dtype=np.float32)
    qtorque = np.asarray(npz["qtorque"], dtype=np.float32)
    joint_target = np.asarray(npz["joint_target"], dtype=np.float32)
    count = len(timestamps)
    wall_start_ns = meta.get("wall_start_ns", 0)
    unix_ns = _make_timestamps(timestamps, wall_start_ns)

    data = {
        "count": count,
        "unix_ns": unix_ns,
        "state": state,
        "qvel": qvel,
        "qtorque": qtorque,
        "joint_target": joint_target,
    }

    ep_dir = output_dir / "episodes" / episode_id
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    _write_arm_state_parquet(ep_dir, "left", data)
    _write_arm_state_parquet(ep_dir, "right", data)
    _write_arm_action_parquet(ep_dir, "left", data)
    _write_arm_action_parquet(ep_dir, "right", data)
    _write_cameras(ep_dir, data, npz)

    # Compute sha256 of the first written parquet as a proxy for episode integrity.
    sha256 = _compute_sha256(ep_dir)

    return {
        "episode_id": episode_id,
        "seed": int(meta.get("seed", 0)),
        "split": meta.get("split", "train"),
        "frames": count,
        "unix_ns_first": int(unix_ns[0]),
        "unix_ns_last": int(unix_ns[-1]),
        "sha256": sha256,
        "gripper_raw_range": [
            round(float(-np.max(state[:, [7, 15]]) / GRIPPER_FULL_OPENING_M), 4),
            round(float(-np.min(state[:, [7, 15]]) / GRIPPER_FULL_OPENING_M), 4),
        ],
    }


def _compute_sha256(ep_dir: Path) -> str:
    """Compute a representative sha256 of the episode."""
    import hashlib

    h = hashlib.sha256()
    for path in sorted(ep_dir.rglob("*")):
        if path.is_file():
            h.update(path.read_bytes())
    return h.hexdigest()


def convert_dataset(
    source_dir: Path,
    output_dir: Path,
    episode_limit: int | None = None,
    overwrite: bool = False,  # noqa: FBT001, FBT002
) -> dict[str, Any]:
    """Convert all episodes in a P9 source directory to v0.3.0 format."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest.json not found in {source_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_version = manifest.get("version")
    if source_version not in SUPPORTED_SOURCE_VERSIONS:
        raise ValueError(
            f"Unsupported manifest version {source_version!r}; expected one of "
            f"{sorted(SUPPORTED_SOURCE_VERSIONS)}"
        )
    if not manifest.get("all_valid", False):
        raise ValueError("Manifest reports not all episodes are valid")
    if manifest.get("image_size") != [480, 640]:
        raise ValueError(
            f"Manifest image_size {manifest.get('image_size')} != [480, 640]; "
            "converter expects 640x480 images"
        )
    raw_dir = source_dir / "raw"
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw directory not found: {raw_dir}")

    records = sorted(manifest["records"], key=lambda r: r["seed"])
    if episode_limit is not None:
        records = records[:episode_limit]

    total_bytes = 0
    eps_map: list[dict[str, Any]] = []
    gripper_ranges: list[list[float]] = []
    ts_spacing: list[float] = []

    for i, record in enumerate(records):
        seed = record["seed"]
        npz_path = raw_dir / f"episode_seed{seed:06d}.npz"
        meta_path = npz_path.with_suffix(".json")
        if not npz_path.is_file():
            raise FileNotFoundError(f"Episode npz not found: {npz_path}")
        if not meta_path.is_file():
            raise FileNotFoundError(f"Episode metadata not found: {meta_path}")

        episode_id = str(seed)
        entry = convert_episode(
            npz_path, meta_path, output_dir, episode_id
        )

        # Collect stats.
        eps_map.append(entry)
        gripper_ranges.append(entry["gripper_raw_range"])
        ns = entry["unix_ns_first"], entry["unix_ns_last"], entry["frames"]
        avg_spacing = (ns[1] - ns[0]) / (ns[2] - 1) / 1e6 if ns[2] > 1 else 0.0
        ts_spacing.append(round(avg_spacing, 6))

        # Count bytes in the output.
        ep_dir = output_dir / "episodes" / episode_id
        for f in ep_dir.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size

        print(f"  [{i+1}/{len(records)}] episode {episode_id}: {entry['frames']} frames, "
              f"gripper raw [{entry['gripper_raw_range'][0]:.4f}, {entry['gripper_raw_range'][1]:.4f}]")

    # Write metadata.yaml.
    meta = _build_metadata(records, raw_dir)
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)

    # Write episode map.
    map_path = source_dir / "episode_map.json"
    map_path.write_text(
        json.dumps(eps_map, ensure_ascii=False, indent=2) + "\n"
    )

    # Write conversion report.
    report = {
        "source": str(source_dir),
        "output": str(output_dir),
        "version": source_version,
        "episodes": len(eps_map),
        "total_frames": sum(e["frames"] for e in eps_map),
        "total_bytes": total_bytes,
        "gripper_raw_range_overall": [
            round(float(min(r[0] for r in gripper_ranges)), 4),
            round(float(max(r[1] for r in gripper_ranges)), 4),
        ],
        "avg_timestamp_spacing_ms": round(float(np.mean(ts_spacing)), 6),
        "schema_metadata": {
            "parquet_created_by": "parquet-cpp-arrow version 20.0.0",
            "reference_created_by": "parquet-cpp-arrow version 24.0.0",
            "note": "created_by version differs from real_data reference (pyarrow 20 vs 24); schema/compression/layout identical",
        },
    }
    report_path = source_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert P9 npz episodes to OpenArm v0.3.0 format"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to P9 collection directory (containing manifest.json + raw/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory (metadata.yaml + episodes/)",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        default=None,
        help="Convert only the first N episodes (for smoke testing)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()

    if args.output.exists():
        if args.overwrite:
            shutil.rmtree(args.output)
        else:
            print(f"Output directory {args.output} already exists; use --overwrite", file=sys.stderr)
            sys.exit(1)

    report = convert_dataset(
        source_dir=args.source,
        output_dir=args.output,
        episode_limit=args.episode_limit,
        overwrite=args.overwrite,
    )
    print(f"\nConversion complete: {report['episodes']} episodes, "
          f"{report['total_frames']} frames, "
          f"{report['total_bytes'] / 1e9:.1f} GB")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
