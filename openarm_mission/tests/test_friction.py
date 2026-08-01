# ruff: noqa: PT009

from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from openarm_mission.config import FrictionMissionConfig
from openarm_mission.controller import BimanualCartesianController
from openarm_mission.friction_expert import FrictionScriptedExpert
from openarm_mission.model import OpenArmMission


class FrictionMissionTest(unittest.TestCase):
    def test_soft_pads_replace_rigid_finger_contacts(self):
        mission = OpenArmMission(FrictionMissionConfig())
        for side in ("left", "right"):
            for finger in ("left", "right"):
                pad_id = mission.finger_pad_geom_ids[side][finger]
                self.assertGreaterEqual(pad_id, 0)
                self.assertEqual(mission.model.geom_condim[pad_id], 6)
                rigid_id = mujoco.mj_name2id(
                    mission.model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    f"openarm_{side}_{finger}_finger_collision",
                )
                self.assertEqual(mission.model.geom_contype[rigid_id], 0)
                self.assertEqual(mission.model.geom_conaffinity[rigid_id], 0)

    def test_gripper_impedance_force_limit_is_configurable(self):
        mission = OpenArmMission(FrictionMissionConfig())
        controller = BimanualCartesianController(mission)
        controller.set_gripper_force_limit("right", 7.25)
        ranges = mission.model.actuator_forcerange[mission.arms["right"].finger_actuator_ids]
        np.testing.assert_allclose(ranges[:, 0], -7.25)
        np.testing.assert_allclose(ranges[:, 1], 7.25)
        with self.assertRaises(ValueError):  # noqa: PT027
            controller.set_gripper_force_limit("right", 0.0)

    def test_full_relay_succeeds_without_weld(self):
        expert = FrictionScriptedExpert(
            seed=7,
            output_dir=Path("/tmp/openarm_p45_test"),
            write_video=False,
        )
        summary = expert.run_expert(write_summary=False)
        self.assertTrue(summary["expert_success"], summary)
        self.assertEqual(summary["p45_grasp_mode"], "pure_friction_no_weld")
        self.assertIsNone(summary["active_weld"])
        self.assertFalse(summary["weld_permitted"])
        self.assertEqual(summary["episode_attempts"], 1)
        self.assertEqual(summary["grasp_attempts"], {"left": 1, "right": 1})
        self.assertEqual(summary["collision_count"], 0)
        self.assertTrue(summary["table_contact"])
        self.assertTrue(all(not expert.task.weld_active(side) for side in ("left", "right")))
        events = {event["event"] for event in summary["events"]}
        self.assertIn("right_friction_lift_confirmed", events)
        self.assertIn("left_friction_lift_confirmed", events)
        self.assertIn("goal_hold_complete", events)


if __name__ == "__main__":
    unittest.main()
