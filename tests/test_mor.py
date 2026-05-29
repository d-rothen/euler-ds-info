from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from euler_ds_info.mor import _estimate_beta_from_depth_response, estimate_mor_from_sample


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

	def test_estimate_mor_from_sample_prefers_dcp_profile(self) -> None:
		with patch(
			"euler_ds_info.mor.estimate_mor_profile_from_sample",
			return_value={"mor_dcp_m": 72.0, "mor_contrast_m": 180.0},
		):
			self.assertEqual(estimate_mor_from_sample({}), 72.0)


if __name__ == "__main__":  # pragma: no cover - test module convenience
	unittest.main()
