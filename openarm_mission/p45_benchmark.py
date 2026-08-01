"""Benchmark the P4.5 weld-free friction expert and export reports."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import csv
import json
import multiprocessing
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from openarm_mission.config import FrictionExpertConfig
from openarm_mission.friction_expert import FrictionScriptedExpert


def _font(size: int, *, bold: bool = False):
    candidates = (
        Path("/usr/share/fonts/opentype/noto/" + ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Medium.ttc")),
        Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _evaluate_seed(seed: int, scratch_dir: str) -> dict[str, Any]:
    started = time.monotonic()
    try:
        expert = FrictionScriptedExpert(
            seed=seed,
            output_dir=Path(scratch_dir),
            write_video=False,
        )
        summary = expert.run_expert(write_summary=False)
        randomization = summary["randomization"]
        return {
            "seed": seed,
            "success": bool(summary["expert_success"]),
            "stage": summary["stage"],
            "failure_reason": summary["failure_reason"],
            "expert_exception": summary["expert_exception"],
            "simulation_seconds": summary["elapsed_seconds"],
            "wall_seconds": round(time.monotonic() - started, 6),
            "episode_attempts": summary["episode_attempts"],
            "right_grasp_attempts": summary["grasp_attempts"]["right"],
            "left_grasp_attempts": summary["grasp_attempts"]["left"],
            "recovery_count": summary["recovery_count"],
            "collision_count": summary["collision_count"],
            "final_xy_error_m": summary["final_xy_error_m"],
            "final_upright_angle_deg": summary["cup_upright_angle_deg"],
            "cup_mass_kg": randomization["cup_mass_kg"],
            "cup_slide_friction": randomization["cup_slide_friction"],
            "pad_slide_friction": randomization["pad_slide_friction"],
            "gripper_force_limit_n": randomization["gripper_force_limit_n"],
            "weld_used": summary["active_weld"] is not None
            or summary["failure_reason"] == "weld_used_in_friction_mode",
            "stage_durations_seconds": summary["stage_durations_seconds"],
            "recovery_events": summary["recovery_events"],
        }
    except Exception as error:
        return {
            "seed": seed,
            "success": False,
            "stage": "worker_exception",
            "failure_reason": type(error).__name__,
            "expert_exception": str(error),
            "simulation_seconds": 0.0,
            "wall_seconds": round(time.monotonic() - started, 6),
            "episode_attempts": 0,
            "right_grasp_attempts": 0,
            "left_grasp_attempts": 0,
            "recovery_count": 0,
            "collision_count": 0,
            "final_xy_error_m": None,
            "final_upright_angle_deg": None,
            "cup_mass_kg": None,
            "cup_slide_friction": None,
            "pad_slide_friction": None,
            "gripper_force_limit_n": None,
            "weld_used": False,
            "stage_durations_seconds": {},
            "recovery_events": [],
        }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _aggregate(
    episodes: list[dict[str, Any]],
    *,
    wall_seconds: float,
    threshold: float,
) -> dict[str, Any]:
    successes = [item for item in episodes if item["success"]]
    failures = [item for item in episodes if not item["success"]]
    errors = [float(item["final_xy_error_m"]) for item in successes if item["final_xy_error_m"] is not None]
    tilts = [
        float(item["final_upright_angle_deg"]) for item in successes if item["final_upright_angle_deg"] is not None
    ]
    success_rate = len(successes) / max(1, len(episodes))
    return {
        "episodes": len(episodes),
        "successes": len(successes),
        "failures": len(failures),
        "success_rate": round(success_rate, 6),
        "required_success_rate": threshold,
        "threshold_met": success_rate >= threshold,
        "wall_seconds": round(wall_seconds, 3),
        "mean_simulation_seconds": round(
            _mean([float(item["simulation_seconds"]) for item in episodes]),
            6,
        ),
        "mean_episode_wall_seconds": round(
            _mean([float(item["wall_seconds"]) for item in episodes]),
            6,
        ),
        "mean_final_xy_error_m": round(_mean(errors), 6),
        "max_final_xy_error_m": round(max(errors, default=0.0), 6),
        "p95_final_xy_error_m": round(
            float(np.percentile(errors, 95)) if errors else 0.0,
            6,
        ),
        "mean_final_upright_angle_deg": round(_mean(tilts), 6),
        "max_final_upright_angle_deg": round(
            max(tilts, default=0.0),
            6,
        ),
        "total_recoveries": sum(int(item["recovery_count"]) for item in episodes),
        "total_collisions": sum(int(item["collision_count"]) for item in episodes),
        "weld_violations": sum(bool(item["weld_used"]) for item in episodes),
        "failure_stages": dict(Counter(item["stage"] for item in failures)),
        "failure_reasons": dict(Counter(item["failure_reason"] or "unknown" for item in failures)),
    }


def _write_csv(path: Path, episodes: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "success",
        "stage",
        "failure_reason",
        "simulation_seconds",
        "wall_seconds",
        "episode_attempts",
        "right_grasp_attempts",
        "left_grasp_attempts",
        "recovery_count",
        "collision_count",
        "final_xy_error_m",
        "final_upright_angle_deg",
        "cup_mass_kg",
        "cup_slide_friction",
        "pad_slide_friction",
        "gripper_force_limit_n",
        "weld_used",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            writer.writerow({key: episode.get(key) for key in fieldnames})


def _card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    label: str,
    value: str,
    color: tuple[int, int, int],
) -> None:
    draw.rounded_rectangle(
        box,
        radius=18,
        fill=(25, 31, 40),
        outline=color,
        width=3,
    )
    draw.text(
        (box[0] + 20, box[1] + 18),
        label,
        font=_font(21),
        fill=(185, 194, 207),
    )
    draw.text(
        (box[0] + 20, box[1] + 56),
        value,
        font=_font(38, bold=True),
        fill=color,
    )


def _write_dashboard(
    path: Path,
    aggregate: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> None:
    width, height = 1400, 860
    image = Image.new("RGB", (width, height), (12, 16, 21))
    draw = ImageDraw.Draw(image)
    green = (91, 214, 147)
    red = (242, 83, 89)
    blue = (99, 183, 255)
    gold = (232, 190, 72)
    primary = green if aggregate["threshold_met"] else red

    draw.text(
        (48, 32),
        "OpenArm v1 · P4.5 纯摩擦双臂接力基准",
        font=_font(40, bold=True),
        fill=(248, 249, 252),
    )
    draw.text(
        (50, 91),
        "软指垫 · 8~12 N 力限位阻抗 · 静置/抬升滑移检测 · weld 全程禁用",
        font=_font(21),
        fill=(178, 188, 202),
    )
    rate = float(aggregate["success_rate"])
    draw.text(
        (50, 145),
        f"{rate:.1%}",
        font=_font(72, bold=True),
        fill=primary,
    )
    draw.text(
        (430, 168),
        (f"{aggregate['successes']} / {aggregate['episodes']} 成功    目标 ≥ {aggregate['required_success_rate']:.0%}"),
        font=_font(29, bold=True),
        fill=(235, 238, 244),
    )
    draw.rounded_rectangle(
        (430, 218, 1320, 252),
        radius=14,
        fill=(52, 61, 74),
    )
    draw.rounded_rectangle(
        (430, 218, 430 + 890 * rate, 252),
        radius=14,
        fill=primary,
    )

    cards = [
        (
            "平均终点误差",
            f"{1000 * aggregate['mean_final_xy_error_m']:.1f} mm",
            blue,
        ),
        (
            "最大最终倾角",
            f"{aggregate['max_final_upright_angle_deg']:.1f}°",
            gold,
        ),
        ("恢复次数", str(aggregate["total_recoveries"]), green),
        ("weld 违规", str(aggregate["weld_violations"]), red),
    ]
    for index, (label, value, color) in enumerate(cards):
        x0 = 50 + 335 * index
        _card(draw, (x0, 296, x0 + 305, 426), label, value, color)

    plot = (50, 520, 950, 780)
    draw.text(
        (50, 470),
        f"{aggregate['episodes']} 个随机种子的纯摩擦终点误差",
        font=_font(25, bold=True),
        fill=(238, 241, 246),
    )
    draw.rounded_rectangle(plot, radius=16, fill=(21, 27, 35))
    valid = [item for item in episodes if item["final_xy_error_m"] is not None and item["success"]]
    max_error = max(
        [float(item["final_xy_error_m"]) for item in valid],
        default=0.05,
    )
    max_error = max(max_error, 0.01)
    minimum_seed = min(item["seed"] for item in episodes)
    maximum_seed = max(item["seed"] for item in episodes)
    for item in valid:
        x = (
            plot[0]
            + 20
            + (plot[2] - plot[0] - 40) * (item["seed"] - minimum_seed) / max(1, maximum_seed - minimum_seed)
        )
        y = plot[3] - 22 - (plot[3] - plot[1] - 44) * (float(item["final_xy_error_m"]) / max_error)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=green)

    side_x = 990
    draw.text(
        (side_x, 470),
        "物理评测摘要",
        font=_font(25, bold=True),
        fill=(238, 241, 246),
    )
    lines = [
        f"总耗时: {aggregate['wall_seconds']:.1f} s",
        f"平均仿真时长: {aggregate['mean_simulation_seconds']:.2f} s",
        f"P95 终点误差: {1000 * aggregate['p95_final_xy_error_m']:.1f} mm",
        f"最大终点误差: {1000 * aggregate['max_final_xy_error_m']:.1f} mm",
        f"平均最终倾角: {aggregate['mean_final_upright_angle_deg']:.2f}°",
        f"意外碰撞: {aggregate['total_collisions']}",
        f"失败原因: {aggregate['failure_reasons'] or '无'}",
    ]
    for index, line in enumerate(lines):
        draw.text(
            (side_x, 520 + 38 * index),
            line,
            font=_font(19),
            fill=(205, 213, 224),
        )
    status = "PASS · P4.5 纯摩擦 ≥95%" if aggregate["threshold_met"] else "未达到 P4.5 目标"
    draw.rounded_rectangle(
        (990, 795, 1350, 842),
        radius=14,
        fill=primary,
    )
    draw.text(
        (1010, 805),
        status,
        font=_font(21, bold=True),
        fill=(12, 16, 21),
    )
    image.save(path)


def run_benchmark(
    *,
    episodes: int,
    start_seed: int,
    workers: int,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if episodes <= 0 or workers <= 0:
        raise ValueError("episodes and workers must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = output_dir / "worker_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    seeds = list(range(start_seed, start_seed + episodes))
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
    ) as executor:
        futures = {executor.submit(_evaluate_seed, seed, str(scratch_dir)): seed for seed in seeds}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 10 == 0 or completed == episodes:
                successes = sum(item["success"] for item in results)
                print(f"progress: {completed}/{episodes}, success={successes}/{completed}")

    results.sort(key=lambda item: item["seed"])
    threshold = FrictionExpertConfig().benchmark_success_threshold
    aggregate = _aggregate(
        results,
        wall_seconds=time.monotonic() - started,
        threshold=threshold,
    )
    report = {
        "robot": "OpenArm v1 bimanual",
        "task": "P4.5 pure-friction paper-cup relay",
        "seed_range": [seeds[0], seeds[-1]],
        "workers": workers,
        "aggregate": aggregate,
        "episodes": results,
    }
    stem = f"p45_friction_benchmark_{episodes}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    dashboard_path = output_dir / f"{stem}.png"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, results)
    _write_dashboard(dashboard_path, aggregate, results)
    return report, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the P4.5 weld-free friction expert.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("openarm_mission/artifacts/p45"),
    )
    args = parser.parse_args()
    report, _ = run_benchmark(
        episodes=args.episodes,
        start_seed=args.start_seed,
        workers=args.workers,
        output_dir=args.output_dir,
    )
    aggregate = report["aggregate"]
    print(
        f"P4.5 benchmark: {aggregate['successes']}/"
        f"{aggregate['episodes']} "
        f"({aggregate['success_rate']:.1%}), "
        f"weld_violations={aggregate['weld_violations']}, "
        f"threshold_met={aggregate['threshold_met']}"
    )
    if not aggregate["threshold_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
