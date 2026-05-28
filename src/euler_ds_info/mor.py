from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np


def _to_numpy(value: Any) -> Optional[np.ndarray]:
	if value is None:
		return None
	if isinstance(value, np.ndarray):
		return value
	if hasattr(value, "detach"):
		try:
			return value.detach().cpu().numpy()  # type: ignore[no-any-return]
		except Exception:
			pass
	if hasattr(value, "cpu") and hasattr(value, "numpy"):
		try:
			return value.cpu().numpy()  # type: ignore[no-any-return]
		except Exception:
			pass
	try:
		return np.asarray(value)
	except Exception:
		return None


def _extract_array(candidate: Any) -> Optional[np.ndarray]:
	if candidate is None:
		return None
	if isinstance(candidate, dict):
		for key in ("points", "xyz", "pointcloud", "data", "values", "value", "array"):
			if key in candidate:
				return _extract_array(candidate[key])
		if len(candidate) == 1:
			return _extract_array(next(iter(candidate.values())))
		return None
	return _to_numpy(candidate)


def _extract_matrix(candidate: Any) -> Optional[np.ndarray]:
	array = _extract_array(candidate)
	if array is None:
		return None
	array = np.asarray(array, dtype=np.float64)
	if array.ndim != 2:
		return None
	if array.shape == (3, 3):
		return array
	if array.shape == (4, 4):
		return array
	if array.shape == (3, 4):
		return np.vstack([array, np.array([0.0, 0.0, 0.0, 1.0])])
	return None


def _normalize_image(rgb: Any) -> Optional[np.ndarray]:
	array = _extract_array(rgb)
	if array is None:
		return None

	image = np.asarray(array, dtype=np.float64)
	if image.ndim == 2:
		image = image[..., None]
	elif image.ndim == 3:
		if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
			image = np.moveaxis(image, 0, -1)
		elif image.shape[-1] not in (1, 3, 4) and image.shape[0] <= 4:
			image = np.moveaxis(image, 0, -1)
	else:
		return None

	if image.size == 0:
		return None

	if np.isfinite(image).any() and np.nanmax(image) > 1.5:
		image = image / 255.0
	return np.clip(image, 0.0, 1.0)


def _infer_rgb_shape(rgb: Any) -> Optional[Tuple[int, int]]:
	image = _normalize_image(rgb)
	if image is None or image.ndim < 2:
		return None
	return int(image.shape[0]), int(image.shape[1])


def _infer_sample_id(sample: dict[str, Any], index: int) -> str:
	for key in ("full_id", "id"):
		value = sample.get(key)
		if isinstance(value, str) and value:
			return value
		if value is not None and not isinstance(value, (dict, list)):
			return str(value)
	return str(index)


def _select_hierarchical_value(value: Any, sample: dict[str, Any]) -> Any:
	if not isinstance(value, dict) or not value:
		return value

	sample_ids = []
	for key in ("full_id", "id"):
		current = sample.get(key)
		if isinstance(current, str) and current:
			sample_ids.append(current)
		elif current is not None:
			sample_ids.append(str(current))

	for sample_id in sample_ids:
		if sample_id in value:
			return value[sample_id]

	best_key: Optional[str] = None
	best_score = -1
	for key in value.keys():
		if not isinstance(key, str):
			continue
		for sample_id in sample_ids:
			if sample_id.startswith(key) or key.startswith(sample_id):
				score = len(key)
				if score > best_score:
					best_score = score
					best_key = key

	if best_key is not None:
		return value[best_key]

	if len(value) == 1:
		return next(iter(value.values()))

	# Deterministic fallback for ambiguous hierarchical calibration values.
	return value[sorted(value.keys())[0]]


def _extract_points(sparse_depth: Any) -> Optional[np.ndarray]:
	array = _extract_array(sparse_depth)
	if array is None:
		return None
	array = np.asarray(array, dtype=np.float64)
	if array.size == 0:
		return None

	if array.ndim == 1:
		if array.shape[0] >= 3:
			return array[:3].reshape(1, 3)
		return None

	if array.ndim >= 2:
		if array.shape[-1] >= 3:
			pts = array.reshape(-1, array.shape[-1])[:, :3]
		elif array.shape[0] >= 3:
			pts = array[:3, :].T
		else:
			return None

		mask = np.isfinite(pts).all(axis=1)
		pts = pts[mask]
		return pts if pts.size > 0 else None

	return None


def _extract_points_with_features(sparse_depth: Any) -> Optional[np.ndarray]:
	array = _extract_array(sparse_depth)
	if array is None:
		return None

	points = np.asarray(array, dtype=np.float64)
	if points.size == 0:
		return None

	if points.ndim == 1:
		if points.shape[0] < 3:
			return None
		points = points.reshape(1, -1)
	elif points.ndim >= 2 and points.shape[0] >= 3 and points.shape[-1] < 3:
		points = points[:3, :].T
	else:
		points = points.reshape(-1, points.shape[-1])

	if points.shape[1] < 3:
		return None

	mask = np.isfinite(points[:, :3]).all(axis=1)
	points = points[mask]
	return points if points.size > 0 else None


def _transform_to_camera(points: np.ndarray, extrinsics: Optional[np.ndarray]) -> np.ndarray:
	if extrinsics is None:
		return points

	matrix = np.asarray(extrinsics, dtype=np.float64)
	if matrix.shape != (4, 4):
		return points

	homogeneous = np.concatenate([points[:, :3], np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
	transformed = homogeneous @ matrix.T
	return transformed[:, :3]


def _project_points(
	points: np.ndarray,
	intrinsics: Optional[np.ndarray],
	rgb_shape: Optional[Tuple[int, int]],
) -> dict[str, np.ndarray]:
	if points.size == 0:
		empty = np.empty((0,), dtype=np.float64)
		return {"depth": empty, "u": empty, "v": empty, "mask": empty.astype(bool)}

	valid = np.isfinite(points).all(axis=1)
	valid &= points[:, 2] > 0
	points = points[valid]
	if points.size == 0:
		empty = np.empty((0,), dtype=np.float64)
		return {"depth": empty, "u": empty, "v": empty, "mask": empty.astype(bool)}

	depth = points[:, 2]
	if intrinsics is None:
		return {
			"depth": depth,
			"u": np.full(depth.shape, np.nan, dtype=np.float64),
			"v": np.full(depth.shape, np.nan, dtype=np.float64),
			"mask": np.ones(depth.shape, dtype=bool)
		}

	matrix = np.asarray(intrinsics, dtype=np.float64)
	if matrix.shape != (3, 3):
		return {
			"depth": depth,
			"u": np.full(depth.shape, np.nan, dtype=np.float64),
			"v": np.full(depth.shape, np.nan, dtype=np.float64),
			"mask": np.ones(depth.shape, dtype=bool)
		}

	z = depth
	x = points[:, 0]
	y = points[:, 1]
	fx = matrix[0, 0]
	fy = matrix[1, 1]
	cx = matrix[0, 2]
	cy = matrix[1, 2]

	with np.errstate(divide="ignore", invalid="ignore"):
		u = fx * (x / z) + cx
		v = fy * (y / z) + cy

	mask = np.isfinite(u) & np.isfinite(v)
	if rgb_shape is not None:
		height, width = rgb_shape
		mask &= (u >= 0) & (v >= 0) & (u < width) & (v < height)

	return {"depth": z[mask], "u": u[mask], "v": v[mask], "mask": mask}


def _pick_channel_axis(image: np.ndarray) -> np.ndarray:
	if image.ndim == 2:
		return image[..., None]
	return image


def _estimate_atmospheric_light(image: np.ndarray) -> np.ndarray:
	pixels = image.reshape(-1, image.shape[-1])
	if pixels.size == 0:
		return np.ones((image.shape[-1],), dtype=np.float64)
	estimate = np.nanquantile(pixels, 0.995, axis=0)
	estimate = np.where(np.isfinite(estimate) & (estimate > 0), estimate, 1.0)
	return estimate


def _compute_dark_channel(image: np.ndarray, radius: int = 3) -> np.ndarray:
	image = _pick_channel_axis(image)
	channel_min = np.min(image, axis=-1)
	padded = np.pad(channel_min, radius, mode="edge")
	windows = []
	for dy in range(2 * radius + 1):
		for dx in range(2 * radius + 1):
			windows.append(padded[dy : dy + channel_min.shape[0], dx : dx + channel_min.shape[1]])
	return np.minimum.reduce(windows)


def _compute_edge_map(luminance: np.ndarray) -> np.ndarray:
	gy, gx = np.gradient(luminance)
	return np.sqrt(gx * gx + gy * gy)


def _sample_map(values: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int = 2) -> np.ndarray:
	if values.size == 0 or u.size == 0:
		return np.empty((0,), dtype=np.float64)

	height, width = values.shape[:2]
	uu = np.rint(u).astype(int)
	vv = np.rint(v).astype(int)
	result = np.empty((uu.shape[0],), dtype=np.float64)

	for index, (x, y) in enumerate(zip(uu, vv)):
		x0 = max(0, x - radius)
		y0 = max(0, y - radius)
		x1 = min(width, x + radius + 1)
		y1 = min(height, y + radius + 1)
		patch = values[y0:y1, x0:x1]
		result[index] = float(np.nanmean(patch)) if patch.size else np.nan

	return result


def _sample_min_map(values: np.ndarray, u: np.ndarray, v: np.ndarray, radius: int = 2) -> np.ndarray:
	if values.size == 0 or u.size == 0:
		return np.empty((0,), dtype=np.float64)

	height, width = values.shape[:2]
	uu = np.rint(u).astype(int)
	vv = np.rint(v).astype(int)
	result = np.empty((uu.shape[0],), dtype=np.float64)

	for index, (x, y) in enumerate(zip(uu, vv)):
		x0 = max(0, x - radius)
		y0 = max(0, y - radius)
		x1 = min(width, x + radius + 1)
		y1 = min(height, y + radius + 1)
		patch = values[y0:y1, x0:x1]
		result[index] = float(np.nanmin(patch)) if patch.size else np.nan

	return result


def _fit_linear(depths: np.ndarray, values: np.ndarray) -> Optional[Tuple[float, float, np.ndarray]]:
	mask = np.isfinite(depths) & np.isfinite(values)
	depths = depths[mask]
	values = values[mask]
	if depths.size < 3:
		return None

	order = np.argsort(depths)
	depths = depths[order]
	values = values[order]

	bin_count = min(10, max(3, int(np.sqrt(depths.size))))
	edges = np.quantile(depths, np.linspace(0.0, 1.0, bin_count + 1))
	bin_depths: list[float] = []
	bin_values: list[float] = []

	for index in range(bin_count):
		left = edges[index]
		right = edges[index + 1]
		if index == bin_count - 1:
			bin_mask = (depths >= left) & (depths <= right)
		else:
			bin_mask = (depths >= left) & (depths < right)
		if int(bin_mask.sum()) == 0:
			continue
		bin_depths.append(float(np.median(depths[bin_mask])))
		bin_values.append(float(np.median(values[bin_mask])))

	if len(bin_depths) < 2:
		return None

	slope, intercept = np.polyfit(np.asarray(bin_depths), np.asarray(bin_values), 1)
	residuals = values - (slope * depths + intercept)
	return float(slope), float(intercept), residuals


def _estimate_mor(beta: float) -> Optional[float]:
	if not np.isfinite(beta) or beta <= 0:
		return None
	return float(3.912 / max(beta, 1e-6))


def _ensure_positive(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
	return np.clip(values, floor, None)


def _normalized_confidence(sample_confidence: Optional[np.ndarray], projected_count: int, valid_mask: np.ndarray) -> np.ndarray:
	if sample_confidence is None:
		return np.ones((projected_count,), dtype=np.float64)

	conf = np.asarray(sample_confidence, dtype=np.float64).reshape(-1)
	conf = conf[:projected_count]
	if conf.size == 0:
		return np.ones((projected_count,), dtype=np.float64)

	conf = np.where(np.isfinite(conf), conf, np.nan)
	if not np.isfinite(conf).any():
		return np.ones((projected_count,), dtype=np.float64)

	if np.nanmax(conf) <= 1.5 and np.nanmin(conf) >= 0.0:
		conf = np.clip(conf, 0.0, 1.0)
	else:
		scale = np.nanquantile(conf, 0.9)
		if not np.isfinite(scale) or scale <= 0:
			scale = np.nanmax(conf)
		if not np.isfinite(scale) or scale <= 0:
			return np.ones((projected_count,), dtype=np.float64)
		conf = np.clip(conf / scale, 0.0, 1.0)

	conf[~valid_mask] = 0.0
	return conf


def estimate_mor_profile_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
	rgb = sample.get("rgb")
	sparse_depth = sample.get("sparse_depth")
	intrinsics = _select_hierarchical_value(sample.get("intrinsics"), sample)
	extrinsics = _select_hierarchical_value(sample.get("camera_extrinsics"), sample)

	image = _normalize_image(rgb)
	rgb_shape = _infer_rgb_shape(rgb)
	points = _extract_points_with_features(sparse_depth)

	num_lidar_points_total = int(points.shape[0]) if points is not None else 0
	if points is None or points.size == 0:
		return {
			"mor_dcp_m": None,
			"mor_contrast_m": None,
			"beta_dcp": None,
			"beta_contrast": None,
			"num_lidar_points_total": num_lidar_points_total,
			"num_projected_points": 0,
			"num_valid_points": 0,
			"valid_fraction": 0.0,
			"median_depth_valid_m": None,
			"fit_mad": None,
			"fog_score_lidar": 0.0
		}

	intrinsics_matrix = _extract_matrix(intrinsics)
	extrinsics_matrix = _extract_matrix(extrinsics)
	camera_points = _transform_to_camera(points, extrinsics_matrix)
	projected = _project_points(camera_points, intrinsics_matrix, rgb_shape)

	projected_depths = np.asarray(projected["depth"], dtype=np.float64)
	projected_count = int(projected_depths.size)
	if projected_count == 0:
		return {
			"mor_dcp_m": None,
			"mor_contrast_m": None,
			"beta_dcp": None,
			"beta_contrast": None,
			"num_lidar_points_total": num_lidar_points_total,
			"num_projected_points": 0,
			"num_valid_points": 0,
			"valid_fraction": 0.0,
			"median_depth_valid_m": None,
			"fit_mad": None,
			"fog_score_lidar": 0.0
		}

	u = np.asarray(projected["u"], dtype=np.float64)
	v = np.asarray(projected["v"], dtype=np.float64)

	if image is None:
		depths = _ensure_positive(projected_depths)
		mor_from_depths = estimate_mor_from_depths(depths)
		beta = 3.912 / mor_from_depths if mor_from_depths and mor_from_depths > 0 else None
		valid_fraction = 1.0
		median_depth = float(np.median(depths)) if depths.size else None
		return {
			"mor_dcp_m": mor_from_depths,
			"mor_contrast_m": mor_from_depths,
			"beta_dcp": beta,
			"beta_contrast": beta,
			"num_lidar_points_total": num_lidar_points_total,
			"num_projected_points": projected_count,
			"num_valid_points": projected_count,
			"valid_fraction": valid_fraction,
			"median_depth_valid_m": median_depth,
			"fit_mad": 0.0,
			"fog_score_lidar": float(np.clip(1.0 - np.exp(-100.0 / max(mor_from_depths or 100.0, 1e-6)), 0.0, 1.0))
		}

	luminance = np.dot(image[..., :3], np.array([0.2126, 0.7152, 0.0722], dtype=np.float64))
	edge_map = _compute_edge_map(luminance)
	dark_channel = _compute_dark_channel(image, radius=3)
	atmospheric_light = _estimate_atmospheric_light(image)
	atmospheric_light = np.maximum(atmospheric_light, 1e-3)
	atmospheric_scale = float(np.nanmean(atmospheric_light))
	if not np.isfinite(atmospheric_scale) or atmospheric_scale <= 0:
		atmospheric_scale = 1.0
	dark_channel_normalized = dark_channel / atmospheric_scale
	dark_channel_normalized = np.clip(dark_channel_normalized, 0.0, 1.0)

	local_dark = _sample_min_map(dark_channel_normalized, u, v, radius=1)
	local_edge = _sample_map(edge_map, u, v, radius=1)
	local_contrast = _sample_map(luminance, u, v, radius=2)
	local_contrast_sq = _sample_map(luminance * luminance, u, v, radius=2)

	contrast_std = np.sqrt(np.maximum(local_contrast_sq - local_contrast * local_contrast, 0.0))
	contrast = contrast_std / np.maximum(np.abs(local_contrast), 1e-3)
	contrast = np.clip(contrast, 1e-4, None)

	confidence_values = None
	if points.shape[1] >= 4:
		confidence_values = np.asarray(points[:, 3], dtype=np.float64)
		if confidence_values.shape[0] != projected_count:
			confidence_values = confidence_values[:projected_count]

	confidence = _normalized_confidence(confidence_values, projected_count, np.ones(projected_count, dtype=bool))
	edge_threshold = float(np.nanquantile(local_edge, 0.8)) if np.isfinite(local_edge).any() else 0.0
	if not np.isfinite(edge_threshold) or edge_threshold <= 0:
		edge_threshold = float(np.nanmean(local_edge)) if np.isfinite(local_edge).any() else 0.0
	if not np.isfinite(edge_threshold) or edge_threshold <= 0:
		edge_threshold = 1.0

	depths = _ensure_positive(projected_depths)
	depth_median = float(np.nanmedian(depths)) if depths.size else 0.0
	depth_mad = float(np.nanmedian(np.abs(depths - depth_median))) if depths.size else 0.0
	if not np.isfinite(depth_mad) or depth_mad <= 1e-6:
		depth_mad = float(np.nanstd(depths)) if depths.size else 0.0
	if not np.isfinite(depth_mad) or depth_mad <= 1e-6:
		depth_mad = max(1.0, 0.1 * depth_median)

	depth_consistency = np.abs(depths - depth_median) <= (4.0 * max(depth_mad, 1e-6))
	edge_consistency = local_edge <= edge_threshold
	contrast_consistency = np.isfinite(contrast) & (contrast > 0)
	confidence_consistency = confidence >= 0.15

	valid_mask = depth_consistency & edge_consistency & contrast_consistency & confidence_consistency & np.isfinite(local_dark)
	num_valid_points = int(valid_mask.sum())
	valid_depths = depths[valid_mask]
	valid_fraction = float(num_valid_points / projected_count) if projected_count else 0.0
	median_depth_valid_m = float(np.median(valid_depths)) if valid_depths.size else None

	valid_local_dark = local_dark[valid_mask]
	valid_contrast = contrast[valid_mask]
	valid_depths = _ensure_positive(valid_depths)

	dcp_values = _ensure_positive(valid_local_dark, floor=1e-4)
	dcp_transmission = np.clip(1.0 - 0.95 * dcp_values, 0.02, 0.99)
	dcp_response = -np.log(dcp_transmission)
	dcp_fit = _fit_linear(valid_depths, dcp_response)
	if dcp_fit is None:
		fallback_mor = estimate_mor_from_depths(valid_depths)
		beta_dcp = 3.912 / fallback_mor if fallback_mor and fallback_mor > 0 else None
		mor_dcp = fallback_mor
		dcp_residual_mad = None
	else:
		dcp_slope, dcp_intercept, dcp_residuals = dcp_fit
		beta_dcp = max(float(dcp_slope), 1e-6)
		mor_dcp = _estimate_mor(beta_dcp)
		dcp_residual_mad = float(np.median(np.abs(dcp_residuals))) if dcp_residuals.size else None

	log_contrast = np.log(_ensure_positive(valid_contrast, floor=1e-4))
	contrast_fit = _fit_linear(valid_depths, log_contrast)
	if contrast_fit is None:
		fallback_mor = estimate_mor_from_depths(valid_depths)
		beta_contrast = 3.912 / fallback_mor if fallback_mor and fallback_mor > 0 else None
		mor_contrast = fallback_mor
		contrast_residual_mad = None
	else:
		contrast_slope, contrast_intercept, contrast_residuals = contrast_fit
		beta_contrast = max(float(-contrast_slope), 1e-6)
		mor_contrast = _estimate_mor(beta_contrast)
		contrast_residual_mad = float(np.median(np.abs(contrast_residuals))) if contrast_residuals.size else None

	fit_residual_candidates = [value for value in (dcp_residual_mad, contrast_residual_mad) if value is not None and np.isfinite(value)]
	if fit_residual_candidates:
		fit_mad = float(np.median(np.asarray(fit_residual_candidates, dtype=np.float64)))
	else:
		fit_mad = None

	score_components = []
	if beta_dcp is not None:
		score_components.append(float(np.clip(1.0 - np.exp(-beta_dcp / 0.012), 0.0, 1.0)))
	if beta_contrast is not None:
		score_components.append(float(np.clip(1.0 - np.exp(-beta_contrast / 0.012), 0.0, 1.0)))
	if mor_dcp is not None:
		score_components.append(float(np.clip(1.0 - np.exp(-80.0 / max(mor_dcp, 1e-6)), 0.0, 1.0)))
	if mor_contrast is not None:
		score_components.append(float(np.clip(1.0 - np.exp(-80.0 / max(mor_contrast, 1e-6)), 0.0, 1.0)))

	if score_components:
		fog_evidence = float(np.mean(score_components))
	else:
		fog_evidence = 0.0

	confidence_penalty = float(np.clip(valid_fraction, 0.0, 1.0))
	if fit_mad is not None and np.isfinite(fit_mad):
		confidence_penalty *= float(np.exp(-fit_mad))
	fog_score_lidar = float(np.clip(fog_evidence * (0.4 + 0.6 * confidence_penalty), 0.0, 1.0))

	return {
		"mor_dcp_m": float(mor_dcp) if mor_dcp is not None else None,
		"mor_contrast_m": float(mor_contrast) if mor_contrast is not None else None,
		"beta_dcp": float(beta_dcp) if beta_dcp is not None else None,
		"beta_contrast": float(beta_contrast) if beta_contrast is not None else None,
		"num_lidar_points_total": num_lidar_points_total,
		"num_projected_points": projected_count,
		"num_valid_points": num_valid_points,
		"valid_fraction": valid_fraction,
		"median_depth_valid_m": median_depth_valid_m,
		"fit_mad": fit_mad,
		"fog_score_lidar": fog_score_lidar
	}


def estimate_mor_from_depths(depths: Any) -> Optional[float]:
	array = _extract_array(depths)
	if array is None:
		return None

	values = np.asarray(array, dtype=np.float64).reshape(-1)
	values = values[np.isfinite(values) & (values > 0)]
	if values.size == 0:
		return None

	if values.size < 5:
		return float(np.quantile(values, 0.9))

	tail = values[values >= np.quantile(values, 0.5)]
	if tail.size < 5:
		tail = values

	low = float(np.quantile(tail, 0.05))
	high = float(np.quantile(tail, 0.95))
	if not np.isfinite(low) or not np.isfinite(high) or high <= low:
		return float(np.quantile(values, 0.9))

	bins = min(24, max(8, int(np.sqrt(tail.size))))
	hist, edges = np.histogram(tail, bins=bins, range=(low, high), density=True)
	centers = 0.5 * (edges[:-1] + edges[1:])
	mask = hist > 0

	if int(mask.sum()) >= 3:
		slope, _intercept = np.polyfit(centers[mask], np.log(hist[mask]), 1)
		beta = max(float(-slope), 1e-6)
		mor = 3.912 / beta
		if np.isfinite(mor) and mor > 0:
			return float(mor)

	return float(np.quantile(values, 0.9))


def summarize_mor_values(mor_values: Any) -> dict[str, dict[str, Optional[float]]]:
	array = _extract_array(mor_values)
	if array is None:
		return {
			"mean": {"MOR": None},
			"median": {"MOR": None},
			"p90": {"MOR": None}
		}

	values = np.asarray(array, dtype=np.float64).reshape(-1)
	values = values[np.isfinite(values) & (values > 0)]
	if values.size == 0:
		return {
			"mean": {"MOR": None},
			"median": {"MOR": None},
			"p90": {"MOR": None}
		}

	return {
		"mean": {"MOR": float(np.mean(values))},
		"median": {"MOR": float(np.median(values))},
		"p90": {"MOR": float(np.quantile(values, 0.9))}
	}


def summarize_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
	numeric_fields = [
		"mor_dcp_m",
		"mor_contrast_m",
		"beta_dcp",
		"beta_contrast",
		"num_lidar_points_total",
		"num_projected_points",
		"num_valid_points",
		"valid_fraction",
		"median_depth_valid_m",
		"fit_mad",
		"fog_score_lidar"
	]

	summary: dict[str, Any] = {
		"sample_count": len(profiles),
		"valid_count": sum(1 for profile in profiles if int(profile.get("num_valid_points") or 0) > 0)
	}

	for stat_name, reducer in (
		("mean", np.mean),
		("median", np.median),
		("p90", lambda values: np.quantile(values, 0.9))
	):
		stat_block: dict[str, Optional[float]] = {}
		for field in numeric_fields:
			values = [
				float(profile[field])
				for profile in profiles
				if field in profile and profile[field] is not None and np.isfinite(float(profile[field]))
			]
			stat_block[field] = float(reducer(np.asarray(values, dtype=np.float64))) if values else None
		summary[stat_name] = stat_block

	return summary


def estimate_mor_from_sample(sample: dict[str, Any]) -> Optional[float]:
	profile = estimate_mor_profile_from_sample(sample)
	for field in ("mor_contrast_m", "mor_dcp_m"):
		value = profile.get(field)
		if isinstance(value, (int, float)) and np.isfinite(float(value)) and float(value) > 0:
			return float(value)
	return None
