"""Tests for converting P9 npz episodes to OpenArm v0.3.0 format."""

import json
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pyarrow.parquet as pq

from openarm_mission.convert_to_openarm_v03 import CAMERA_MAP, convert_dataset
from openarm_mission.dataset import (
    CAMERAS,
    DATASET_VERSION,
    GRIPPER_FULL_OPENING_M,
)


class ConverterTest(unittest.TestCase):
    """Smoke test for the converter on the 2-episode smoke set."""

    @classmethod
    def setUpClass(cls):
        cls.source_dir = Path("openarm_mission/artifacts/p9_smoke")
        cls.manifest = cls.source_dir / "manifest.json"
        if not cls.manifest.is_file():
            raise unittest.SkipTest("p9_smoke manifest not found; run smoke collection first")

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp(prefix="p9_converter_test_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_convert_2_episodes(self):
        """Convert the 2 smoke episodes and verify the output tree."""
        report = convert_dataset(
            source_dir=self.source_dir,
            output_dir=self.tmpdir,
            episode_limit=2,
            overwrite=True,
        )
        self.assertEqual(report["episodes"], 2)
        self.assertEqual(report["total_frames"], 864)

        # Check metadata.yaml
        meta_path = self.tmpdir / "metadata.yaml"
        self.assertTrue(meta_path.is_file())
        import yaml
        meta = yaml.safe_load(meta_path.read_text())
        self.assertEqual(meta["version"], "0.3.0")
        self.assertEqual(meta["operation_type"], "scripted")
        self.assertEqual(len(meta["episodes"]), 2)

        # Check episode directories
        for ep_id in ("0", "1"):
            ep_dir = self.tmpdir / "episodes" / ep_id
            self.assertTrue(ep_dir.is_dir())

            # Check parquet schemas
            for side in ("left", "right"):
                obs_path = ep_dir / "obs" / "arms" / side / "state.parquet"
                self.assertTrue(obs_path.is_file())
                t = pq.read_metadata(obs_path)
                self.assertEqual(t.num_rows, 432)

                # Check schema field names
                s = pq.read_schema(obs_path)
                names = s.names
                self.assertIn("timestamp", names)
                self.assertIn("qpos", names)
                self.assertIn("qvel", names)
                self.assertIn("qtorque", names)

                act_path = ep_dir / "action" / "arms" / side / "qpos.parquet"
                self.assertTrue(act_path.is_file())
                t = pq.read_metadata(act_path)
                self.assertEqual(t.num_rows, 432)

            # Check cameras
            for sim_key, real_name in CAMERA_MAP.items():
                cam_dir = ep_dir / "cameras" / real_name
                self.assertTrue(cam_dir.is_dir())
                files = sorted(cam_dir.iterdir())
                self.assertEqual(len(files), 432, f"Expected 432 JPEGs in {real_name}")
                # Check JPEG magic bytes
                blob = files[0].read_bytes()
                self.assertEqual(blob[:3], b"\xff\xd8\xff")

    def test_gripper_raw_range(self):
        """Verify that gripper raw values are in the expected [-1, 0] range."""
        smoke_out = Path("openarm_mission/artifacts/p9_smoke/openarm_paper_cup_relay")
        if not (smoke_out / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet").is_file():
            raise unittest.SkipTest("smoke output not found; run smoke conversion first")
        table = pq.read_table(
            smoke_out / "episodes" / "0" / "obs" / "arms" / "left" / "state.parquet",
            columns=["qpos"],
        )
        qpos = np.array(table.column("qpos").to_pylist(), dtype=np.float64)
        gripper = qpos[:, 7]
        # Gripper raw should be in [-1, 0] (or close to it).
        self.assertLessEqual(gripper.min(), 0.0)
        self.assertGreaterEqual(gripper.max(), -1.0)

    def test_episode_map_sidecar(self):
        """Check that episode_map.json is written."""
        map_path = self.source_dir / "episode_map.json"
        self.assertTrue(map_path.is_file())
        eps = json.loads(map_path.read_text())
        self.assertEqual(len(eps), 2)
        for ep in eps:
            self.assertIn("episode_id", ep)
            self.assertIn("seed", ep)
            self.assertIn("split", ep)
            self.assertIn("frames", ep)
            self.assertIn("sha256", ep)

    def test_conversion_report_sidecar(self):
        """Check that conversion_report.json is written."""
        report_path = self.source_dir / "conversion_report.json"
        self.assertTrue(report_path.is_file())
        report = json.loads(report_path.read_text())
        self.assertEqual(report["episodes"], 2)


if __name__ == "__main__":
    unittest.main()