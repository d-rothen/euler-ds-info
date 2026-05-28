from __future__ import annotations

import argparse
import json
import sys
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


def run(document: dict[str, Any]) -> dict[str, Any]:
	mode = _normalize_mode(document.get("mode"))
	modalities = _normalize_modalities(document.get("modalities"))

	if mode != "estimate-mor":
		raise ValueError(f"Unsupported mode: {mode}")

	try:
		from euler_loading import Modality, MultiModalDataset
	except ImportError as exc:  # pragma: no cover - depends on installed environment
		raise ImportError(
			"euler-loading is required to run MOR estimation"
		) from exc

	dataset = MultiModalDataset(
		modalities={name: Modality(path) for name, path in modalities.items()}
	)

	per_file_info: dict[str, dict[str, Any]] = {}
	for index in range(len(dataset)):
		sample = dataset[index]
		if not isinstance(sample, dict):
			continue

		profile = estimate_mor_profile_from_sample(sample)
		if not profile:
			continue

		sample_id = _infer_sample_id(sample, index)
		per_file_info[sample_id] = profile

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
	return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
	parser = build_parser()
	args = parser.parse_args(list(argv) if argv is not None else None)

	try:
		document = _load_input_document(args.input_json)
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
