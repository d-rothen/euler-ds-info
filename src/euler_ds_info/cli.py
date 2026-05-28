from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Iterable
from typing import Any, Optional

from .mor import estimate_mor_profile_from_sample, summarize_profiles


def _load_input_document(raw_input: str) -> dict[str, Any]:
	if raw_input.strip() == "-":
		payload = sys.stdin.read()
	else:
		payload = raw_input

	document = json.loads(payload)
	if not isinstance(document, dict) or isinstance(document, list):
		raise ValueError("Input JSON must be an object")
	return document


def _normalize_mode(value: Any) -> str:
	if not isinstance(value, str) or not value.strip():
		raise ValueError("mode is required")
	return value.strip()


def _normalize_modalities(value: Any) -> dict[str, str]:
	if not isinstance(value, dict) or isinstance(value, list):
		raise ValueError("modalities must be an object")

	modalities: dict[str, str] = {}
	for key, entry in value.items():
		if not isinstance(key, str) or not key.strip():
			continue
		if not isinstance(entry, str) or not entry.strip():
			raise ValueError(f"modalities.{key} must be a non-empty string")
		modalities[key.strip()] = entry.strip()
	return modalities


def _normalize_hierarchical_modalities(value: Any) -> dict[str, str]:
	if value is None:
		return {}
	if not isinstance(value, dict) or isinstance(value, list):
		raise ValueError("hierarchical_modalities must be an object")

	modalities: dict[str, str] = {}
	for key, entry in value.items():
		if not isinstance(key, str) or not key.strip():
			continue
		if not isinstance(entry, str) or not entry.strip():
			raise ValueError(f"hierarchical_modalities.{key} must be a non-empty string")
		modalities[key.strip()] = entry.strip()
	return modalities


def _normalize_optional_positive_int(value: Any, label: str) -> Optional[int]:
	if value is None or value == "":
		return None
	try:
		parsed = int(value)
	except (TypeError, ValueError) as exc:
		raise ValueError(f"{label} must be a positive integer") from exc
	if parsed <= 0:
		raise ValueError(f"{label} must be a positive integer")
	return parsed


def _resolve_worker_count(value: Any) -> int:
	workers = _normalize_optional_positive_int(value, "workers")
	if workers is not None:
		return workers

	env_workers = _normalize_optional_positive_int(os.environ.get("EULER_DS_INFO_WORKERS"), "EULER_DS_INFO_WORKERS")
	if env_workers is not None:
		return env_workers

	slurm_workers = _normalize_optional_positive_int(os.environ.get("SLURM_CPUS_PER_TASK"), "SLURM_CPUS_PER_TASK")
	if slurm_workers is not None:
		return slurm_workers

	cpu_count = os.cpu_count() or 1
	return max(1, min(cpu_count, 4))


def _build_output(
	mode: str,
	modalities: dict[str, str],
	per_file_info: dict[str, dict[str, Any]],
) -> dict[str, Any]:
	ordered_profiles = list(per_file_info.values())

	return {
		"artifact_name": "euler-ds-info",
		"mode": mode,
		"modalities": modalities,
		"per_file_info": per_file_info,
		"aggregate": summarize_profiles(ordered_profiles),
		"stats": {
			"sample_count": len(per_file_info),
			"valid_count": sum(1 for entry in ordered_profiles if int(entry.get("num_valid_points") or 0) > 0)
		}
	}


def _emit_progress(message: str) -> None:
	print(message, file=sys.stderr, flush=True)


def _estimate_profile_task(item: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any] | None]:
	index, sample = item
	profile = estimate_mor_profile_from_sample(sample)
	if not profile:
		return index, _infer_sample_id(sample, index), None
	return index, _infer_sample_id(sample, index), profile


def run(document: dict[str, Any]) -> dict[str, Any]:
	mode = _normalize_mode(document.get("mode"))
	modalities = _normalize_modalities(document.get("modalities"))
	hierarchical_modalities = _normalize_hierarchical_modalities(document.get("hierarchical_modalities"))
	workers = _resolve_worker_count(document.get("workers"))

	if mode != "estimate-mor":
		raise ValueError(f"Unsupported mode: {mode}")

	try:
		from euler_loading import Modality, MultiModalDataset
	except ImportError as exc:  # pragma: no cover - depends on installed environment
		raise ImportError(
			"euler-loading is required to run MOR estimation"
		) from exc

	dataset = MultiModalDataset(
		modalities={name: Modality(path) for name, path in modalities.items()},
		hierarchical_modalities={
			name: Modality(path, collapse_single=True)
			for name, path in hierarchical_modalities.items()
		} or None,
	)

	started_at = time.monotonic()
	_emit_progress(
		f"[euler-ds-info] Loaded dataset with {len(dataset)} samples "
		f"({len(modalities)} regular modalities, {len(hierarchical_modalities)} hierarchical modalities, {workers} worker(s))"
	)

	per_file_info: dict[str, dict[str, Any]] = {}
	total_samples = len(dataset)
	batch_size = max(1, workers * 8)

	def iter_batches() -> Iterable[list[tuple[int, dict[str, Any]]]]:
		batch: list[tuple[int, dict[str, Any]]] = []
		for index in range(total_samples):
			sample = dataset[index]
			if not isinstance(sample, dict):
				continue
			batch.append((index, sample))
			if len(batch) >= batch_size:
				yield batch
				batch = []
		if batch:
			yield batch

	processed_count = 0
	with ThreadPoolExecutor(max_workers=workers) as executor:
		for batch in iter_batches():
			if workers == 1:
				results = map(_estimate_profile_task, batch)
			else:
				results = executor.map(_estimate_profile_task, batch, chunksize=1)

			for index, sample_id, profile in results:
				processed_count += 1
				if profile:
					per_file_info[sample_id] = profile
				if processed_count == 1 or processed_count % 100 == 0 or processed_count == total_samples:
					elapsed = time.monotonic() - started_at
					_emit_progress(
						f"[euler-ds-info] Processed {processed_count}/{total_samples} samples "
						f"({len(per_file_info)} profiled, {elapsed:.1f}s elapsed)"
					)

	return _build_output(mode, modalities, per_file_info)


def _infer_sample_id(sample: dict[str, Any], index: int) -> str:
	for key in ("full_id", "id"):
		value = sample.get(key)
		if isinstance(value, str) and value:
			return value
		if value is not None and not isinstance(value, (dict, list)):
			return str(value)
	return str(index)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="euler-ds-info")
	parser.add_argument(
		"--input-json",
		default="-",
		help="JSON document describing the mode and modalities, or '-' to read from stdin."
	)
	parser.add_argument(
		"--pretty",
		action="store_true",
		help="Pretty-print the emitted JSON output."
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=None,
		help="Optional worker count for per-sample profile estimation."
	)
	return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(list(argv) if argv is not None else None)

	try:
		document = _load_input_document(args.input_json)
		if args.workers is not None:
			document["workers"] = args.workers
		output = run(document)
		json.dump(output, sys.stdout, indent=2 if args.pretty else None, sort_keys=args.pretty)
		sys.stdout.write("\n")
		return 0
	except Exception as exc:  # pragma: no cover - CLI failure path
		error = {
			"artifact_name": "euler-ds-info",
			"ok": False,
			"error": {
				"type": exc.__class__.__name__,
				"message": str(exc)
			}
		}
		json.dump(error, sys.stdout, indent=2 if args.pretty else None, sort_keys=args.pretty)
		sys.stdout.write("\n")
		return 1
