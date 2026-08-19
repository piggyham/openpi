# ruff: noqa: PT009

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
import warnings

import mujoco
import numpy as np

from openarm_mission.expert import RelayScriptedExpert
from openarm_mission.expert import SelfCollisionWarning
from openarm_mission.expert import _self_collision_side


class RelayScriptedExpertTest(unittest.TestCase):
    def test_self_collision_classification(self):
        self.assertEqual(
            _self_collision_side(("openarm_left_link2_collision", "openarm_left_link6_collision")),
            "left",
        )
        self.assertEqual(
            _self_collision_side(("openarm_body_link0_collision", "openarm_right_link4_collision")),
            "right",
        )
        self.assertEqual(
            _self_collision_side(("mission_left_left_finger_pad", "openarm_left_hand_collision")),
            "left",
        )
        self.assertIsNone(
            _self_collision_side(("openarm_left_link6_collision", "openarm_right_link6_collision"))
        )
        self.assertIsNone(
            _self_collision_side(("mission_table_top", "openarm_left_link5_collision"))
        )

    def test_self_collision_warning_is_emitted_once_per_contact_onset(self):
        expert = RelayScriptedExpert.__new__(RelayScriptedExpert)
        expert.self_collision_warnings = []
        expert._active_self_collision_pairs = set()  # noqa: SLF001
        expert._episode_attempt = 0  # noqa: SLF001
        expert.task = SimpleNamespace(elapsed=1.25, stage=SimpleNamespace(value="test"))
        contact = {
            "side": "left",
            "geom1": "openarm_left_link2_collision",
            "geom2": "openarm_left_link6_collision",
            "distance_m": -0.001,
        }
        expert.self_collision_contacts = lambda: [contact]

        with self.assertWarns(SelfCollisionWarning):
            expert._warn_self_collisions()  # noqa: SLF001
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            expert._warn_self_collisions()  # noqa: SLF001

        self.assertEqual(caught, [])
        self.assertEqual(len(expert.self_collision_warnings), 1)

    def test_baseline_has_no_unexpected_contacts(self):
        expert = RelayScriptedExpert(
            seed=0,
            output_dir=Path("/tmp/openarm_p4_test"),
            write_video=False,
        )
        self.assertEqual(expert.unexpected_contacts(), [])
        expert._close_media()  # noqa: SLF001

    def test_first_unfold_waypoint_moves_only_j1_and_j4(self):
        expert = RelayScriptedExpert(
            seed=0,
            output_dir=Path("/tmp/openarm_p4_test"),
            write_video=False,
        )
        try:
            for side, arm in expert.mission.arms.items():
                target = expert._arm_unfold_qpos(side)  # noqa: SLF001
                changed = np.flatnonzero(~np.isclose(target, expert.home_qpos[side]))
                np.testing.assert_array_equal(changed, [0, 3])
                expected_j1 = expert.mission.config.arm_unfold_joint1 * (1 if side == "left" else -1)
                self.assertAlmostEqual(target[0], expected_j1)
                self.assertAlmostEqual(target[3], expert.mission.config.arm_unfold_joint4)

                expert.mission.data.qpos[arm.qpos_indices] = target
            mujoco.mj_forward(expert.mission.model, expert.mission.data)
            table_near_edge = expert.mission.config.table_center[0] - expert.mission.config.table_half_size[0]
            for side in ("left", "right"):
                tcp = expert.mission.tcp_pose(side)[0]
                self.assertGreater(tcp[2], expert.mission.config.table_top_z + 0.03)
                self.assertLess(tcp[0], table_near_edge)
        finally:
            expert._close_media()  # noqa: SLF001

    def test_both_arms_recover_from_first_grasp_rejection(self):
        expert = RelayScriptedExpert(
            seed=7,
            output_dir=Path("/tmp/openarm_p4_test"),
            write_video=False,
            fault_injections={"right_grasp": 1, "left_grasp": 1},
        )
        summary = expert.run_expert(write_summary=False)
        self.assertTrue(summary["expert_success"], summary)
        self.assertEqual(summary["episode_attempts"], 1)
        self.assertEqual(summary["grasp_attempts"], {"left": 2, "right": 2})
        self.assertEqual(summary["recovery_count"], 2)
        self.assertEqual(summary["collision_count"], 0)
        reasons = {event["reason"] for event in summary["recovery_events"] if event["reason"] is not None}
        self.assertIn("injected_right_grasp_rejection", reasons)
        self.assertIn("injected_left_grasp_rejection", reasons)
        for arm in expert.mission.arms.values():
            actual = expert.mission.data.qpos[arm.qpos_indices]
            np.testing.assert_allclose(actual, expert.mission.config.home_arm_qpos, atol=0.01)


if __name__ == "__main__":
    unittest.main()
