"""Tests for the P5 data schema and validation utilities."""

from __future__ import annotations

import unittest

import numpy as np

from openarm_mission.dataset import ACTION_NAMES
from openarm_mission.dataset import STATE_NAMES
from openarm_mission.dataset import episode_split
from openarm_mission.dataset import schema_dict
from openarm_mission.dataset import validate_trajectory


class DatasetTest(unittest.TestCase):
    def test_schema_and_split_are_stable(self) -> None:
        schema = schema_dict(fps=10, height=120, width=160)
        assert len(STATE_NAMES) == 16
        assert len(ACTION_NAMES) == 14
        assert schema["action"]["gripper_sign"] == "-1=open, +1=closed"
        counts = {
            split: sum(episode_split(seed) == split for seed in range(200)) for split in ("train", "validation", "test")
        }
        assert counts == {"train": 160, "validation": 20, "test": 20}

    def test_alignment_validator_accepts_valid_trajectory(self) -> None:
        frames = 4
        trajectory = {
            "timestamp": np.arange(frames, dtype=np.float64) / 10.0,
            "state": np.zeros((frames, 16), dtype=np.float32),
            "action": np.zeros((frames, 14), dtype=np.float32),
            "cup_pose": np.zeros((frames, 7), dtype=np.float32),
            "image_front": np.zeros(
                (frames, 12, 16, 3),
                dtype=np.uint8,
            ),
            "image_left_wrist": np.zeros(
                (frames, 12, 16, 3),
                dtype=np.uint8,
            ),
            "image_right_wrist": np.zeros(
                (frames, 12, 16, 3),
                dtype=np.uint8,
            ),
        }
        report = validate_trajectory(trajectory, fps=10)
        assert report["valid"]

    def test_alignment_validator_rejects_bad_timestamp(self) -> None:
        frames = 3
        trajectory = {
            "timestamp": np.array([0.0, 0.1, 0.25]),
            "state": np.zeros((frames, 16), dtype=np.float32),
            "action": np.zeros((frames, 14), dtype=np.float32),
            "cup_pose": np.zeros((frames, 7), dtype=np.float32),
            **{
                f"image_{key}": np.zeros(
                    (frames, 2, 2, 3),
                    dtype=np.uint8,
                )
                for key in ("front", "left_wrist", "right_wrist")
            },
        }
        report = validate_trajectory(trajectory, fps=10)
        assert not report["valid"]


if __name__ == "__main__":
    unittest.main()
