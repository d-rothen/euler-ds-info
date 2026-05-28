# euler-ds-info

Dataset-level statistics helpers for Euler-view.

Current mode:

- `estimate-mor` - estimate a lidar-aware MOR profile from RGB, sparse depth, intrinsics, and camera extrinsics loaded through `euler-loading`.

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
