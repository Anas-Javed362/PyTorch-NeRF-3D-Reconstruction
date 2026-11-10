# Dataset: NeRF Synthetic (Blender) Dataset

This project uses the **NeRF Synthetic Blender dataset** introduced in the original
[NeRF paper](https://arxiv.org/abs/2003.08934) (Mildenhall et al., 2020).

---

## Download

The dataset is hosted on Google Drive. You can download individual scenes or the
full dataset (~1.3 GB compressed).

**Official link:**
```
https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
```

**Using the provided download script (recommended):**
```bash
python scripts/download_data.py --scene lego --out_dir data/
```

Supported scenes: `chair`, `drums`, `ficus`, `hotdog`, `lego`, `materials`, `mic`, `ship`

---

## Expected Directory Structure

After downloading and extracting, the data directory should look like this:

```
data/
└── lego/                   # (or any other scene name)
    ├── transforms_train.json
    ├── transforms_val.json
    ├── transforms_test.json
    ├── train/
    │   ├── r_0.png
    │   ├── r_1.png
    │   └── ...
    ├── val/
    │   ├── r_0.png
    │   └── ...
    └── test/
        ├── r_0.png
        └── ...
```

Each `transforms_<split>.json` file contains:
- `camera_angle_x`: horizontal field of view (radians)
- `frames`: list of `{ "file_path": "...", "transform_matrix": [[...]] }` entries
  where `transform_matrix` is the **camera-to-world** 4×4 pose matrix.

Images are RGBA PNG files (800×800 by default). The alpha channel is used to
composite a white background when `white_bkgd: true` is set in the config.

---

## Configuration

Point the config to your data directory:

```yaml
# configs/lego_baseline.yaml
dataset:
  scene: lego
  data_dir: data/lego   # path relative to project root
  img_wh: [100, 100]    # resize resolution (use 800×800 for full quality)
  white_bkgd: true
```

---

## Notes

- **Disk space**: Each full scene is ~200–300 MB; all 8 scenes together ~1.3 GB.
- **Resizing**: The default config uses 100×100 for fast CPU training. Increase
  `img_wh` for higher quality at the cost of longer training time.
- **License**: The Blender synthetic dataset is released under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) by the original authors.
