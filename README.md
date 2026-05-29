# euler-ds-info

Dataset-level statistics helpers for Euler-view.

Current mode:

- `estimate-mor` - estimate a lidar-aware MOR profile from RGB, sparse depth, intrinsics, and camera extrinsics loaded through `euler-loading`.

## Installation

```bash
pip install euler-ds-info
```

## Usage

```bash
python -m euler_ds_info --input-json - <<'JSON'
{
  "mode": "estimate-mor",
  "modalities": {
    "rgb": "/data/dataset/train#rgb",
    "sparse_depth": "/data/dataset/train#sparse_depth",
    "intrinsics": "/data/dataset/train#intrinsics",
    "camera_extrinsics": "/data/dataset/train#camera_extrinsics"
  }
}
JSON
```

The command prints a single JSON document with per-file MOR profiles and aggregate summary statistics.

The output includes a top-level `glossary` section. Metric definitions are shared between
`per_file_info` and aggregate reducer outputs, and each metric entry includes display labels,
units, value ranges, interpretation hints, and caveats for downstream visualization.

To generate a sample artifact for inspection, run:

```bash
python -m euler_ds_info.dev_artifacts
```

This writes a realistic sample JSON document to `./.outputs/euler-ds-info.sample.json` by
calling the real MOR pipeline against a mocked in-memory `euler-loading` dataset.

## Performance

- The CLI will use `SLURM_CPUS_PER_TASK` automatically when present.
- You can override the worker count with `--workers N` or `EULER_DS_INFO_WORKERS=N`.
- The loader still runs sequentially; the speedup comes from parallel per-sample profiling.
