# ruff: noqa: PT009

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from openarm_mission.model import OpenArmMission


class OpenArmMissionModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = OpenArmMission()

    def test_official_bimanual_structure_is_present(self):
        self.assertEqual(set(self.mission.arms), {"left", "right"})
        for arm in self.mission.arms.values():
            self.assertEqual(arm.joint_ids.shape, (7,))
            self.assertEqual(arm.actuator_ids.shape, (7,))
            self.assertEqual(arm.finger_joint_ids.shape, (2,))
            self.assertEqual(arm.finger_actuator_ids.shape, (2,))
            self.assertGreaterEqual(arm.tcp_site_id, 0)

    def test_task_scene_is_present(self):
        for geom_name in (
            "mission_table_top",
            "mission_region_a",
            "mission_handoff_marker",
            "mission_region_b",
            "mission_cup_body",
            "mission_cup_inside",
        ):
            geom_id = mujoco.mj_name2id(
                self.mission.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_name,
            )
            self.assertGreaterEqual(geom_id, 0, geom_name)

        red_id = mujoco.mj_name2id(self.mission.model, mujoco.mjtObj.mjOBJ_GEOM, "mission_region_a")
        blue_id = mujoco.mj_name2id(self.mission.model, mujoco.mjtObj.mjOBJ_GEOM, "mission_region_b")
        self.assertGreater(self.mission.model.geom_rgba[red_id, 0], 0.8)
        self.assertGreater(self.mission.model.geom_rgba[blue_id, 2], 0.8)
        red_position = self.mission.data.geom_xpos[red_id]
        blue_position = self.mission.data.geom_xpos[blue_id]
        self.assertLess(red_position[1], 0.0)
        self.assertGreater(blue_position[1], 0.0)

        geom_names = {
            mujoco.mj_id2name(
                self.mission.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                geom_id,
            )
            for geom_id in range(self.mission.model.ngeom)
        }
        self.assertFalse(any(name and "handle" in name for name in geom_names))

    def test_home_pose_and_cup_state_are_finite(self):
        for side in ("left", "right"):
            arm = self.mission.arms[side]
            np.testing.assert_allclose(
                self.mission.data.qpos[arm.qpos_indices],
                np.zeros(7),
                atol=1e-9,
            )
            position, quaternion = self.mission.tcp_pose(side)
            self.assertTrue(np.all(np.isfinite(position)))
            self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=6)
            self.assertLess(position[2], self.mission.config.table_top_z)
        self.assertTrue(np.all(np.isfinite(self.mission.cup_position())))
        self.assertAlmostEqual(
            float(self.mission.cup_position()[2]),
            self.mission.config.cup_initial_position[2],
            places=6,
        )
        self.assertLess(
            2.0 * self.mission.config.cup_top_radius,
            2.0 * self.mission.config.open_finger_qpos,
        )
        self.assertLess(
            self.mission.config.cup_wall_thickness,
            self.mission.config.cup_bottom_radius,
        )

    def test_task_ready_pose_is_distinct_from_natural_hang(self):
        self.assertEqual(self.mission.config.home_arm_qpos, (0.0,) * 7)
        self.assertNotEqual(self.mission.config.ready_arm_qpos, self.mission.config.home_arm_qpos)
        table_near_edge = self.mission.config.table_center[0] - self.mission.config.table_half_size[0]
        self.assertLess(self.mission.config.arm_stow_clearance_x, table_near_edge)
        self.assertGreater(self.mission.config.arm_stow_clearance_z, self.mission.config.table_top_z)

    def test_all_finger_actuators_are_position_actuators(self):
        for arm in self.mission.arms.values():
            gains = self.mission.model.actuator_gainprm[arm.finger_actuator_ids, 0]
            self.assertTrue(np.all(gains > 0))
            self.assertTrue(np.all(self.mission.model.actuator_ctrllimited[arm.finger_actuator_ids]))

    def test_task_cameras_look_forward_and_down(self):
        ready = np.asarray(self.mission.config.ready_arm_qpos)
        for arm in self.mission.arms.values():
            self.mission.data.qpos[arm.qpos_indices] = ready
        mujoco.mj_forward(self.mission.model, self.mission.data)

        for name in (
            "mission_front_camera",
            "mission_left_wrist_camera",
            "mission_right_wrist_camera",
        ):
            camera_id = mujoco.mj_name2id(self.mission.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            rotation = self.mission.data.cam_xmat[camera_id].reshape(3, 3)
            optical_axis = -rotation[:, 2]
            self.assertGreater(optical_axis[0], 0.45, name)
            self.assertLess(optical_axis[2], -0.80, name)

        front_id = mujoco.mj_name2id(
            self.mission.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            "mission_front_camera",
        )
        np.testing.assert_allclose(
            self.mission.data.cam_xpos[front_id],
            self.mission.config.head_camera_position,
            atol=1e-9,
        )
        for arm in self.mission.arms.values():
            self.mission.data.qpos[arm.qpos_indices] = self.mission.config.home_arm_qpos
        mujoco.mj_forward(self.mission.model, self.mission.data)


if __name__ == "__main__":
    unittest.main()
