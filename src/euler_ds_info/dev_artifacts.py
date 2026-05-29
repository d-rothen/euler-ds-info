from __future__ import annotations

import argparse
import json
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .cli import run

DEFAULT_OUTPUT_DIR = Path(".outputs")
DEFAULT_OUTPUT_FILENAME = "euler-ds-info.sample.json"


def _make_rgb_sample(seed: int, haze_strength: float) -> np.ndarray:
	rng = np.random.default_rng(seed)
	height, width = 32, 32
	yy, xx = np.meshgrid(
		np.linspace(0.0, 1.0, height, dtype=np.float64),
		np.linspace(0.0, 1.0, width, dtype=np.float64),
		indexing="ij"
	)
	base = np.stack([xx, yy, 1.0 - xx], axis=-1)
	haze = np.array([0.76, 0.81, 0.86], dtype=np.float64)
	image = base * (1.0 - 0.6 * haze_strength) + haze * (0.6 * haze_strength)
	image += rng.normal(0.0, 0.01 + 0.01 * haze_strength, size=image.shape)
	return np.clip(image, 0.0, 1.0)


def _make_sparse_depth(
	z_start: float,
	z_stop: float,
	count: int,
	fx: float = 40.0,
	fy: float = 40.0,
	cx: float = 16.0,
	cy: float = 16.0,
) -> np.ndarray:
	u = np.linspace(2.0, 29.0, count, dtype=np.float64)
	v = np.linspace(3.0, 28.0, count, dtype=np.float64)[::-1]
	z = np.linspace(z_start, z_stop, count, dtype=np.float64)
	x = (u - cx) * z / fx
	y = (v - cy) * z / fy
	confidence = np.linspace(0.55, 0.98, count, dtype=np.float64)
	return np.stack([x, y, z, confidence], axis=-1)


def _build_sample(sample_id: str, seed: int, haze_strength: float, z_start: float, z_stop: float) -> dict[str, Any]:
	return {
		"full_id": sample_id,
		"id": sample_id,
		"rgb": _make_rgb_sample(seed=seed, haze_strength=haze_strength),
		"sparse_depth": _make_sparse_depth(z_start=z_start, z_stop=z_stop, count=96),
		"intrinsics": np.array(
			[
				[40.0, 0.0, 16.0],
				[0.0, 40.0, 16.0],
				[0.0, 0.0, 1.0]
			],
			dtype=np.float64
		),
		"camera_extrinsics": np.array(
			[
				[1.0, 0.0, 0.0, 0.0],
				[0.0, 1.0, 0.0, 0.0],
				[0.0, 0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0, 1.0]
			],
			dtype=np.float64
		)
	}


SAMPLE_RUN_DOCUMENT: dict[str, Any] = {
	"mode": "estimate-mor",
	"modalities": {
		"rgb": "mock://rgb",
		"sparse_depth": "mock://sparse_depth"
	},
	"hierarchical_modalities": {
		"intrinsics": "mock://intrinsics",
		"camera_extrinsics": "mock://camera_extrinsics"
	}
}

SAMPLE_DATASET = [
	_build_sample("sample-clear", seed=1, haze_strength=0.12, z_start=8.0, z_stop=24.0),
	_build_sample("sample-hazy", seed=2, haze_strength=0.58, z_start=16.0, z_stop=52.0),
	_build_sample("sample-dim", seed=3, haze_strength=0.32, z_start=6.0, z_stop=18.0)
]


class _FakeModality:
	def __init__(self, path: str, **_kwargs: Any) -> None:
		self.path = path


class _FakeMultiModalDataset:
	def __init__(self, modalities: dict[str, _FakeModality], hierarchical_modalities: dict[str, _FakeModality] | None = None) -> None:
		self.modalities = modalities
		self.hierarchical_modalities = hierarchical_modalities or {}
		self._samples = SAMPLE_DATASET

	def __len__(self) -> int:
		return len(self._samples)

	def __getitem__(self, index: int) -> dict[str, Any]:
		return self._samples[index]


@contextmanager
def _mock_euler_loading() -> Iterator[None]:
	module = types.SimpleNamespace(Modality=_FakeModality, MultiModalDataset=_FakeMultiModalDataset)
	previous = sys.modules.get("euler_loading")
	sys.modules["euler_loading"] = module
	try:
		yield
	finally:
		if previous is None:
			sys.modules.pop("euler_loading", None)
		else:
			sys.modules["euler_loading"] = previous


def build_sample_mor_output() -> dict[str, Any]:
	with _mock_euler_loading():
		return run(SAMPLE_RUN_DOCUMENT)


def write_sample_mor_output(output_dir: Path | str = DEFAULT_OUTPUT_DIR, filename: str = DEFAULT_OUTPUT_FILENAME) -> Path:
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	target = output_path / filename
	document = build_sample_mor_output()
	target.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return target


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="euler-ds-info-dev-artifacts")
	parser.add_argument(
		"--output-dir",
		default=str(DEFAULT_OUTPUT_DIR),
		help="Directory to write the sample artifact JSON into."
	)
	parser.add_argument(
		"--filename",
		default=DEFAULT_OUTPUT_FILENAME,
		help="Name of the JSON artifact file."
	)
	return parser


def main(argv: list[str] | None = None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	target = write_sample_mor_output(args.output_dir, args.filename)
	print(str(target))
	return 0


if __name__ == "__main__":  # pragma: no cover - helper entrypoint
	raise SystemExit(main())
