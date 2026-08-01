"""Run the P4 100-episode expert benchmark and export reports."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import csv
from dataclasses import asdict
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

from openarm_mission.config import ScriptedExpertConfig
from openarm_mission.expert import RelayScriptedExpert


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
        expert = RelayScriptedExpert(
            seed=seed,
            output_dir=Path(scratch_dir),
            write_video=False,
        )
        summary = expert.run_expert(write_summary=False)
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
            "cup_mass_kg": summary["randomization"]["cup_mass_kg"],
            "cup_slide_friction": summary["randomization"]["cup_slide_friction"],
            "cup_xy_offset_m": summary["randomization"]["cup_xy_offset_m"],
            "cup_yaw_rad": summary["randomization"]["cup_yaw_rad"],
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
            "cup_xy_offset_m": None,
            "cup_yaw_rad": None,
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
    successes = [episode for episode in episodes if episode["success"]]
    failures = [episode for episode in episodes if not episode["success"]]
    errors = [float(episode["final_xy_error_m"]) for episode in successes if episode["final_xy_error_m"] is not None]
    tilts = [
        float(episode["final_upright_angle_deg"])
        for episode in successes
        if episode["final_upright_angle_deg"] is not None
    ]
    stage_names = sorted({stage for episode in episodes for stage in episode["stage_durations_seconds"]})
    mean_stage_durations = {
        stage: round(
            _mean([float(episode["stage_durations_seconds"].get(stage, 0.0)) for episode in successes]),
            6,
        )
        for stage in stage_names
    }
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
            _mean([float(episode["simulation_seconds"]) for episode in episodes]),
            6,
        ),
        "mean_episode_wall_seconds": round(
            _mean([float(episode["wall_seconds"]) for episode in episodes]),
            6,
        ),
        "mean_final_xy_error_m": round(_mean(errors), 6),
        "max_final_xy_error_m": round(max(errors, default=0.0), 6),
        "p95_final_xy_error_m": round(
            float(np.percentile(errors, 95)) if errors else 0.0,
            6,
        ),
        "mean_final_upright_angle_deg": round(_mean(tilts), 6),
        "max_final_upright_angle_deg": round(max(tilts, default=0.0), 6),
        "total_recoveries": sum(int(episode["recovery_count"]) for episode in episodes),
        "total_collisions": sum(int(episode["collision_count"]) for episode in episodes),
        "failure_stages": dict(Counter(episode["stage"] for episode in failures)),
        "failure_reasons": dict(Counter(episode["failure_reason"] or "unknown" for episode in failures)),
        "mean_stage_durations_seconds": mean_stage_durations,
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
        "cup_xy_offset_m",
        "cup_yaw_rad",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for episode in episodes:
            row = {key: episode.get(key) for key in fieldnames}
            row["cup_xy_offset_m"] = json.dumps(row["cup_xy_offset_m"])
            writer.writerow(row)


def _metric_card(
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
        outline=(*color, 220),
        width=3,
    )
    draw.text(
        (box[0] + 20, box[1] + 18),
        label,
        font=_font(22),
        fill=(185, 194, 207),
    )
    draw.text(
        (box[0] + 20, box[1] + 55),
        value,
        font=_font(42, bold=True),
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
        (48, 34),
        "OpenArm v1 · P4 双臂脚本专家基准",
        font=_font(42, bold=True),
        fill=(248, 249, 252),
    )
    draw.text(
        (50, 92),
        "接触门控 · 备选预抓取 · 后撤重试 · 双臂互锁 · 失败分阶段统计",
        font=_font(22),
        fill=(178, 188, 202),
    )

    success_rate = float(aggregate["success_rate"])
    draw.text(
        (50, 148),
        f"{success_rate:.1%}",
        font=_font(76, bold=True),
        fill=primary,
    )
    draw.text(
        (430, 170),
        (f"{aggregate['successes']} / {aggregate['episodes']} 成功    目标 ≥ {aggregate['required_success_rate']:.0%}"),
        font=_font(30, bold=True),
        fill=(235, 238, 244),
    )
    bar_box = (430, 220, 1320, 254)
    draw.rounded_rectangle(bar_box, radius=14, fill=(52, 61, 74))
    draw.rounded_rectangle(
        (
            bar_box[0],
            bar_box[1],
            bar_box[0] + (bar_box[2] - bar_box[0]) * success_rate,
            bar_box[3],
        ),
        radius=14,
        fill=primary,
    )
    threshold_x = bar_box[0] + (bar_box[2] - bar_box[0]) * aggregate["required_success_rate"]
    draw.line(
        (threshold_x, bar_box[1] - 10, threshold_x, bar_box[3] + 10),
        fill=(255, 255, 255),
        width=4,
    )

    cards = [
        (
            "平均终点误差",
            f"{1000 * aggregate['mean_final_xy_error_m']:.1f} mm",
            blue,
        ),
        (
            "最大终点倾角",
            f"{aggregate['max_final_upright_angle_deg']:.3f}°",
            gold,
        ),
        ("恢复次数", str(aggregate["total_recoveries"]), green),
        ("碰撞次数", str(aggregate["total_collisions"]), red),
    ]
    for index, (label, value, color) in enumerate(cards):
        x0 = 50 + index * 335
        _metric_card(
            draw,
            (x0, 300, x0 + 305, 430),
            label,
            value,
            color,
        )

    draw.text(
        (50, 478),
        f"{aggregate['episodes']} 个随机种子的终点 XY 误差",
        font=_font(26, bold=True),
        fill=(238, 241, 246),
    )
    plot = (50, 530, 950, 780)
    draw.rounded_rectangle(plot, radius=16, fill=(21, 27, 35))
    valid = [episode for episode in episodes if episode["final_xy_error_m"] is not None]
    max_error = max(
        [float(episode["final_xy_error_m"]) for episode in valid],
        default=0.05,
    )
    max_error = max(max_error, 0.01)
    for episode in valid:
        x = (
            plot[0]
            + 20
            + (plot[2] - plot[0] - 40)
            * (episode["seed"] - min(item["seed"] for item in episodes))
            / max(
                1,
                max(item["seed"] for item in episodes) - min(item["seed"] for item in episodes),
            )
        )
        y = plot[3] - 22 - (plot[3] - plot[1] - 44) * (float(episode["final_xy_error_m"]) / max_error)
        color = green if episode["success"] else red
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    draw.text(
        (plot[0] + 18, plot[3] - 35),
        f"seed {min(item['seed'] for item in episodes)}",
        font=_font(16),
        fill=(160, 171, 186),
    )
    draw.text(
        (plot[2] - 110, plot[3] - 35),
        f"seed {max(item['seed'] for item in episodes)}",
        font=_font(16),
        fill=(160, 171, 186),
    )
    draw.text(
        (plot[0] + 18, plot[1] + 14),
        f"max {1000 * max_error:.1f} mm",
        font=_font(16),
        fill=(160, 171, 186),
    )

    side_x = 990
    draw.text(
        (side_x, 478),
        "评测摘要",
        font=_font(26, bold=True),
        fill=(238, 241, 246),
    )
    summary_lines = [
        f"总耗时: {aggregate['wall_seconds']:.1f} s",
        f"平均仿真时长: {aggregate['mean_simulation_seconds']:.2f} s",
        f"P95 终点误差: {1000 * aggregate['p95_final_xy_error_m']:.1f} mm",
        f"最大终点误差: {1000 * aggregate['max_final_xy_error_m']:.1f} mm",
        f"平均终点倾角: {aggregate['mean_final_upright_angle_deg']:.3f}°",
        f"失败阶段: {aggregate['failure_stages'] or '无'}",
        f"失败原因: {aggregate['failure_reasons'] or '无'}",
    ]
    for index, line in enumerate(summary_lines):
        draw.text(
            (side_x, 530 + index * 38),
            line,
            font=_font(20),
            fill=(205, 213, 224),
        )

    status = "PASS · 达到 P4 ≥95% 验收线" if aggregate["threshold_met"] else "FAIL"
    draw.rounded_rectangle(
        (990, 795, 1350, 842),
        radius=14,
        fill=primary,
    )
    draw.text(
        (1010, 805),
        status,
        font=_font(22, bold=True),
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
            result = future.result()
            results.append(result)
            if completed % 10 == 0 or completed == episodes:
                successes = sum(item["success"] for item in results)
                print(f"progress: {completed}/{episodes}, success={successes}/{completed}")

    results.sort(key=lambda result: result["seed"])
    config = ScriptedExpertConfig()
    aggregate = _aggregate(
        results,
        wall_seconds=time.monotonic() - started,
        threshold=config.benchmark_success_threshold,
    )
    report = {
        "robot": "OpenArm v1 bimanual",
        "task": "right red pick -> center handoff -> left blue place",
        "seed_range": [seeds[0], seeds[-1]],
        "workers": workers,
        "expert_config": asdict(config),
        "aggregate": aggregate,
        "episodes": results,
    }
    json_path = output_dir / f"p4_benchmark_{episodes}.json"
    csv_path = output_dir / f"p4_benchmark_{episodes}.csv"
    dashboard_path = output_dir / f"p4_benchmark_{episodes}.png"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, results)
    _write_dashboard(dashboard_path, aggregate, results)
    return report, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the P4 OpenArm scripted expert.")
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
        default=Path("openarm_mission/artifacts/p4"),
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
        f"P4 benchmark: {aggregate['successes']}/{aggregate['episodes']} "
        f"({aggregate['success_rate']:.1%}), "
        f"threshold_met={aggregate['threshold_met']}"
    )
    print(
        f"reports: {args.output_dir / f'p4_benchmark_{args.episodes}.json'}, "
        f"{args.output_dir / f'p4_benchmark_{args.episodes}.csv'}, "
        f"{args.output_dir / f'p4_benchmark_{args.episodes}.png'}"
    )
    if not aggregate["threshold_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
