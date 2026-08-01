# ruff: noqa: PT009, PT027

from __future__ import annotations

import unittest

import numpy as np

from openarm_mission.controller import BimanualCartesianController
from openarm_mission.controller import InvalidActionError
from openarm_mission.model import OpenArmMission


class BimanualControllerTest(unittest.TestCase):
    def setUp(self):
        self.mission = OpenArmMission()
        self.controller = BimanualCartesianController(self.mission)

    def test_rejects_bad_actions(self):
        with self.assertRaises(InvalidActionError):
            self.controller.apply_action(np.zeros(13))
        invalid = np.zeros(14)
        invalid[3] = np.nan
        with self.assertRaises(InvalidActionError):
            self.controller.apply_action(invalid)

    def test_clips_cartesian_targets_to_workspace(self):
        action = np.full(14, 1e6)
        self.controller.apply_action(action)
        lower = self.controller.config.workspace_min_array()
        upper = self.controller.config.workspace_max_array()
        for target in self.controller.target_position.values():
            self.assertTrue(np.all(target >= lower))
            self.assertTrue(np.all(target <= upper))

    def test_random_nearby_ik_targets(self):
        rng = np.random.default_rng(7)
        successes = 0
        total = 0
        for side in ("left", "right"):
            position, quaternion = self.mission.tcp_pose(side)
            for _ in range(20):
                target = position + rng.uniform(-0.012, 0.012, size=3)
                result = self.controller.solve_ik(side, target, quaternion)
                successes += int(result.converged)
                total += 1
        self.assertGreaterEqual(successes / total, 0.98)

    def test_position_only_ik_reaches_both_task_regions(self):
        table_z = self.mission.config.table_top_z
        targets = {
            "right": (*self.mission.config.region_a_center, table_z + 0.18),
            "left": (*self.mission.config.region_b_center, table_z + 0.18),
        }
        for side, target in targets.items():
            result = self.controller.solve_ik(side, np.asarray(target))
            self.assertTrue(result.converged, (side, result))
            self.assertLess(result.position_error, 0.004)
            self.assertEqual(result.orientation_error, 0.0)

    def test_ctrl_is_finite_and_torque_limited(self):
        ctrl = self.controller.compute_ctrl()
        self.assertTrue(np.all(np.isfinite(ctrl)))
        for arm in self.mission.arms.values():
            torque = ctrl[arm.actuator_ids]
            limits = self.mission.model.actuator_forcerange[arm.actuator_ids]
            self.assertTrue(np.all(torque >= limits[:, 0]))
            self.assertTrue(np.all(torque <= limits[:, 1]))

    def test_home_hold_physics_is_stable(self):
        initial = {
            side: np.array(self.mission.data.qpos[arm.qpos_indices], copy=True)
            for side, arm in self.mission.arms.items()
        }
        self.controller.step(300)
        self.assertTrue(np.all(np.isfinite(self.mission.data.qpos)))
        for side, arm in self.mission.arms.items():
            drift = np.max(np.abs(self.mission.data.qpos[arm.qpos_indices] - initial[side]))
            self.assertLess(drift, 0.08)

    def test_both_arms_track_a_synchronized_cartesian_move(self):
        initial = {side: self.mission.tcp_pose(side)[0] for side in ("left", "right")}
        action = np.zeros(14)
        action[0] = 0.02
        action[7] = 0.02
        results = self.controller.apply_action(action)
        self.assertTrue(all(result.converged for result in results.values()))
        self.controller.step(500)

        displacements = {}
        for side in ("left", "right"):
            position = self.mission.tcp_pose(side)[0]
            displacements[side] = position - initial[side]
            self.assertGreater(displacements[side][0], 0.012)
            self.assertLess(
                np.linalg.norm(position - self.controller.target_position[side]),
                0.01,
            )
        np.testing.assert_allclose(
            displacements["left"],
            displacements["right"],
            atol=2e-4,
        )


if __name__ == "__main__":
    unittest.main()
