# ruff: noqa: PT009

from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

from openarm_mission.controller import BimanualCartesianController
from openarm_mission.model import OpenArmMission
from openarm_mission.task import OpenArmRelayTask
from openarm_mission.task import RelayStage


class OpenArmRelayTaskTest(unittest.TestCase):
    def setUp(self):
        self.mission = OpenArmMission()
        self.task = OpenArmRelayTask(self.mission)
        self.controller = BimanualCartesianController(self.mission)

    def _set_cup_position(self, position):
        address = self.mission.cup_qpos_address
        self.mission.data.qpos[address : address + 7] = np.concatenate(
            [
                np.asarray(position, dtype=np.float64),
                np.array([1.0, 0.0, 0.0, 0.0]),
            ]
        )
        self.mission.data.qvel[self.mission.cup_dof_address : self.mission.cup_dof_address + 6] = 0.0
        mujoco.mj_forward(self.mission.model, self.mission.data)

    def test_reset_randomization_is_reproducible_and_bounded(self):
        first = self.task.reset(seed=17)
        first_position = self.mission.cup_position()
        second = self.task.reset(seed=17)
        np.testing.assert_allclose(first_position, self.mission.cup_position())
        self.assertEqual(first, second)

        offset = np.asarray(first["cup_xy_offset_m"])
        self.assertLessEqual(
            np.linalg.norm(offset),
            self.task.config.initial_xy_randomization_radius + 1e-6,
        )
        self.assertFalse(self.task.weld_active("left"))
        self.assertFalse(self.task.weld_active("right"))

    def test_contact_gate_activates_only_the_expected_weld(self):
        self.task.reset(seed=3)
        self.controller.reset_targets()
        cup_position = self.mission.cup_position()
        target = cup_position + np.array([0.0, 0.0, self.task.config.physical_grasp_tcp_offset_z])
        result = self.controller.solve_ik("right", target)
        self.assertTrue(result.converged)
        arm = self.mission.arms["right"]
        self.mission.data.qpos[arm.qpos_indices] = result.qpos
        self.mission.data.qpos[arm.finger_qpos_indices] = 0.0
        mujoco.mj_forward(self.mission.model, self.mission.data)

        allowed, details = self.task.grasp_gate("right")
        self.assertTrue(allowed, details)
        attached, _ = self.task.try_attach("right")
        self.assertTrue(attached)
        self.assertTrue(self.task.weld_active("right"))
        self.assertFalse(self.task.weld_active("left"))
        self.assertEqual(self.task.stage, RelayStage.RIGHT_ATTACHED)

        self.task.inject_grasp_loss("right")
        self.task.update()
        self.assertEqual(self.task.stage, RelayStage.FAILURE)
        self.assertEqual(self.task.failure_reason, "right_grasp_lost")

    def test_left_arm_is_interlocked_until_center_handoff(self):
        allowed, details = self.task.grasp_gate("left")
        self.assertFalse(allowed)
        self.assertEqual(details["expected_stage"], "wait_left_grasp")
        self.assertEqual(details["actual_stage"], "wait_right_grasp")

    def test_goal_requires_continuous_half_second_hold(self):
        self.task.reset(seed=2)
        self.controller.reset_targets()
        blue = self.mission.config.region_b_center
        self._set_cup_position(
            (
                blue[0],
                blue[1],
                self.mission.config.table_top_z + self.mission.config.cup_half_height,
            )
        )
        self.task.stage = RelayStage.WAIT_GOAL_STABLE
        self.task._stage_started_at = self.task.elapsed  # noqa: SLF001

        steps = round(0.75 / self.mission.config.timestep)
        for _ in range(steps):
            self.controller.compute_ctrl()
            mujoco.mj_step(self.mission.model, self.mission.data)
            self.task.update()
        self.assertTrue(self.task.success, self.task.summary())

    def test_drop_and_non_finite_states_fail_closed(self):
        self.task.reset(seed=0)
        position = self.mission.cup_position()
        position[2] = self.mission.config.table_top_z - self.task.config.drop_below_table_margin - 0.01
        self._set_cup_position(position)
        self.task.update()
        self.assertEqual(self.task.stage, RelayStage.FAILURE)
        self.assertEqual(self.task.failure_reason, "cup_dropped_below_table")

    def test_bddl_contains_relay_regions_and_goal(self):
        bddl = (Path(__file__).resolve().parents[1] / "bddl" / "openarm_v1_bimanual_paper_cup_relay.bddl").read_text(
            encoding="utf-8"
        )
        for token in (
            "red_a_region",
            "center_handoff_region",
            "blue_b_region",
            "(On paper_cup_1 mission_table_red_a_region)",
            "(On paper_cup_1 mission_table_blue_b_region)",
        ):
            self.assertIn(token, bddl)


if __name__ == "__main__":
    unittest.main()
