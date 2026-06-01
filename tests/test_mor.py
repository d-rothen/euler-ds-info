from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from euler_ds_info.mor import (
	_depth_consistency_mask,
	_estimate_beta_from_depth_response,
	_estimate_visibility_mor,
	_project_points,
	_validity_preset,
	estimate_mor_profile_from_sample,
	estimate_mor_from_sample,
)


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
MUSES_FIXTURE_DIR = FIXTURES_DIR / "muses"
PRINCETON_FIXTURE_DIR = FIXTURES_DIR / "princeton_stf"


class MorEstimatorTests(unittest.TestCase):
	def test_estimate_beta_from_depth_response_is_robust_to_outliers(self) -> None:
		depths = np.linspace(5.0, 25.0, 60, dtype=np.float64)
		true_beta = 0.055
		response = true_beta * depths
		response[::13] *= 0.35
		response[::17] *= 1.45

		beta = _estimate_beta_from_depth_response(depths, response, quantile=0.88)

		self.assertIsNotNone(beta)
		assert beta is not None
		self.assertGreater(beta, 0.04)
		self.assertLess(beta, 0.07)

	def test_estimate_mor_from_sample_prefers_visibility_profile(self) -> None:
		with patch(
			"euler_ds_info.mor.estimate_mor_profile_from_sample",
			return_value={"mor_visibility_m": 118.0, "mor_dcp_m": 72.0, "mor_contrast_m": 180.0},
		):
			self.assertEqual(estimate_mor_from_sample({}), 118.0)

	def test_visibility_mor_uses_edge_clarity_to_relax_dcp(self) -> None:
		self.assertAlmostEqual(_estimate_visibility_mor(72.0, 500.0, 0.0) or 0.0, 72.0)
		self.assertGreater(_estimate_visibility_mor(55.0, 230.0, 0.65) or 0.0, 150.0)
		self.assertLess(_estimate_visibility_mor(55.0, 230.0, 0.65) or 999.0, 220.0)

	def test_depth_validity_preserves_far_field_tail(self) -> None:
		depths = np.concatenate(
			[
				np.linspace(8.0, 18.0, 40, dtype=np.float64),
				np.array([40.0, 60.0, 90.0, 120.0], dtype=np.float64),
			]
		)

		mask = _depth_consistency_mask(depths, _validity_preset())

		self.assertTrue(mask[-4:].all())

	def test_depth_validity_rejects_extreme_singleton_outlier(self) -> None:
		depths = np.concatenate(
			[
				np.linspace(8.0, 18.0, 40, dtype=np.float64),
				np.array([40.0, 60.0, 90.0, 120.0, 10000.0], dtype=np.float64),
			]
		)

		mask = _depth_consistency_mask(depths, _validity_preset())

		self.assertTrue(mask[-2])
		self.assertFalse(mask[-1])

	def test_projection_returns_source_indices_for_projected_points(self) -> None:
		points = np.array(
			[
				[0.0, 0.0, -1.0],
				[0.0, 0.0, 10.0],
				[1.0, 1.0, 20.0],
				[1000.0, 0.0, 30.0],
			],
			dtype=np.float64,
		)
		intrinsics = np.array(
			[
				[10.0, 0.0, 2.0],
				[0.0, 10.0, 2.0],
				[0.0, 0.0, 1.0],
			],
			dtype=np.float64,
		)

		projected = _project_points(points, intrinsics, (6, 6))

		np.testing.assert_array_equal(projected["index"], np.array([1, 2], dtype=np.int64))

	def test_fixture_samples_preserve_dataset_ordering(self) -> None:
		try:
			from euler_loading.loaders.cpu.muses import (
				read_extrinsics as read_muses_extrinsics,
				read_intrinsics as read_muses_intrinsics,
				rgb as read_muses_rgb,
				sparse_depth as read_muses_sparse_depth,
			)
			from euler_loading.loaders.cpu.princeton_dense import (
				read_extrinsics as read_princeton_extrinsics,
				read_intrinsics as read_princeton_intrinsics,
				rgb as read_princeton_rgb,
				sparse_depth as read_princeton_sparse_depth,
			)
		except ImportError as exc:
			self.skipTest(f"euler-loading loaders are unavailable: {exc}")

		muses_k = read_muses_intrinsics(MUSES_FIXTURE_DIR / "calib.json")
		muses_t = read_muses_extrinsics(MUSES_FIXTURE_DIR / "calib.json")
		muses_samples = [
			"test/fog/day/677253",
			"test/fog/day/671430",
			"train/fog/day/646133",
		]
		muses_mors = []
		for sample_id in muses_samples:
			with self.subTest(dataset="muses", sample_id=sample_id):
				profile = estimate_mor_profile_from_sample(
					{
						"rgb": read_muses_rgb(MUSES_FIXTURE_DIR / "muses+rgb" / f"{sample_id}.png"),
						"sparse_depth": read_muses_sparse_depth(
							MUSES_FIXTURE_DIR / "muses+sparse_depth" / f"{sample_id}.bin"
						),
						"intrinsics": muses_k,
						"camera_extrinsics": muses_t,
					}
				)
				mor = float(profile["mor_visibility_m"])
				self.assertGreater(mor, 50.0)
				self.assertLess(mor, 95.0)
				muses_mors.append(mor)

		princeton_k = read_princeton_intrinsics(PRINCETON_FIXTURE_DIR / "calib_cam_stereo_left.json")
		princeton_t = read_princeton_extrinsics(PRINCETON_FIXTURE_DIR / "calib_tf_tree_full.json")
		princeton_samples = [
			("2018-10-08_08-18-59_00700", 180.0, 240.0),
			("2018-10-08_08-18-59_03700", 150.0, 220.0),
			("2018-10-08_08-27-03_03730", 95.0, 150.0),
		]
		princeton_prefix = "day/none/dense_fog/dry/clean/highway/non_twilight/open_road"
		princeton_mors = []
		for sample_name, lower, upper in princeton_samples:
			sample_id = f"{princeton_prefix}/{sample_name}"
			with self.subTest(dataset="princeton_stf", sample_id=sample_name):
				profile = estimate_mor_profile_from_sample(
					{
						"rgb": read_princeton_rgb(
							PRINCETON_FIXTURE_DIR / "princeton_stf+png+rgb" / f"{sample_id}.png"
						),
						"sparse_depth": read_princeton_sparse_depth(
							PRINCETON_FIXTURE_DIR / "princeton_stf+sparse_depth" / f"{sample_id}.bin"
						),
						"intrinsics": princeton_k,
						"camera_extrinsics": princeton_t,
					}
				)
				mor = float(profile["mor_visibility_m"])
				self.assertGreater(mor, lower)
				self.assertLess(mor, upper)
				princeton_mors.append(mor)

		self.assertGreater(min(princeton_mors), max(muses_mors))


if __name__ == "__main__":  # pragma: no cover - test module convenience
	unittest.main()
