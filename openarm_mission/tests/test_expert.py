# ruff: noqa: PT009

from __future__ import annotations

from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()
