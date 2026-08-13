# ruff: noqa: PT009

from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from openarm_mission.expert import RelayScriptedExpert


class RelayScriptedExpertTest(unittest.TestCase):
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
