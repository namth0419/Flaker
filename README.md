# Flaker — Lithography Vision Toolkit with Flake Analyzer & EBL Aligner

A desktop tool (Python + OpenCV + Tkinter) for two common tasks in 2D-material device fabrication:

1. **Flake Analyzer** — batch-detect exfoliated 2D-material flakes in optical microscope images using HSV color thresholding, with geometric isolation checking to flag flakes that are safely separated from surrounding debris.
2. **EBL Aligner** — automatically align microscope images to a stage coordinate system using template-matched fiducial marks, generating `.ssc` alignment files for Raith e-beam lithography.

No machine learning, no GPU, no training data required — everything runs on classical computer vision (HSV thresholding, contour analysis, template matching), so it starts working the moment you calibrate it on your own images.

> ⚠️ This tool was built for a specific lab workflow (grid-based sample naming, a specific `.ssc` alignment file format). See [Known Limitations](#known-limitations) before adopting it as-is — you will likely need to adjust a couple of assumptions for your own setup.

---

## Features

### 🔍 Flake Analyzer
- Interactive scale calibration (click-and-drag on a scale bar, with zoom/pan)
- Interactive HSV color picker for both the target flake and the background, with live mask preview and hue-exclusion ranges (handles hue wrap-around, e.g. red targets near 0°/179°)
- Area (µm²) and aspect-ratio filtering
- **Isolation checking**: rejects flakes that have debris/other material within a configurable clearance radius, using either a circular or shape-following (dilation-based) clearance zone
- Optional strict color re-verification per detected blob, using a mathematically correct **circular mean** for hue (avoids the wrap-around averaging error common with naive HSV means)
- Multiprocessing batch mode with per-core thread management to avoid CPU oversubscription
- Debug image export showing accepted/rejected flakes and why
- Per-file failure logging (a bad image never silently drops results — it's reported)
- JSON config presets (save/load calibration + all parameters)

### 🎯 EBL Aligner
- Template-matching-based fiducial mark detection with sub-pixel accuracy (parabolic peak interpolation)
- Automatic affine transform + rotation angle calculation from three corner marks
- Batch processing of an image folder into `.bmp` + `.ssc` alignment file pairs
- Configurable filename suffix for input matching
- Per-file error isolation (one bad alignment doesn't stop the batch)

---

## Requirements

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [NumPy](https://numpy.org/)
- Tkinter (bundled with most Python installers; on Linux you may need `sudo apt install python3-tk`)

```bash
pip install -r requirements.txt
```

## Installation

```bash
git clone https://github.com/namth0419/Flaker.git
cd Flaker
pip install -r requirements.txt
python align_flake_1_0.py
```

---

## Usage

### Flake Analyzer tab

1. **Browse** to a folder of `.jpg` microscope images (or leave it at the script's own folder).
2. **Calibrate Scale** — drag a line over a scale bar in a sample image and enter its physical length (µm).
3. **Pick Target Flake Color** / **Pick Background Color** — click on representative pixels in the split preview window; adjust the *Tolerance* and *Exclude H* sliders until the mask cleanly separates your target material from everything else.
4. Tune the **Analysis Parameters** panel (area range, aspect ratio, clearance radius, etc. — see [Configuration Reference](#configuration-reference)).
5. **Start Batch Processing.** Progress and per-file results stream into the log panel.

**Outputs** (written to the configured output folder, default `results/`):

| File | Contents |
|---|---|
| `<name>_analyzed.jpg` | Original image annotated with accepted flakes, IDs, and clearance boundary |
| `<name>_debug.jpg` | *(if enabled)* Accepted/rejected flakes with rejection reason overlaid |
| `isolated_flakes_report.csv` | One row per accepted flake: filename, ID, area (µm²), center (px), radius (µm), average BGR color |
| `failed_files_log.txt` | *(if any failures)* Filename + full traceback for images that could not be processed |

### EBL Aligner tab

1. **Browse** to a folder containing images with a shared suffix before the extension (default `_ShiftN`, e.g. `01_02_ShiftN.jpg`).
2. **Run Alignment Pipeline** — an OpenCV window opens on the first image; drag a bounding box around the small alignment crosshair and press Enter. This becomes the fiducial template used for every image in the batch.
3. The tool locates the fiducial in three corners of each image (sub-pixel), computes an affine transform against nominal stage coordinates derived from the filename, and writes `<name>.bmp` + `<name>.ssc` for each input image.

---

## Configuration Reference

`config.json` (Flake Analyzer) is a nested dictionary; unspecified keys fall back to these defaults:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `file_io` | `input_pattern` | `*.jpg` | Glob pattern for input images |
| | `output_folder` | `results` | Relative or absolute output path |
| | `output_suffix` | `_analyzed` | Suffix for annotated output images |
| | `jpeg_quality` | `100` | JPEG write quality |
| | `save_debug_images` | `false` | Export reject/accept debug overlays |
| `calibration` | `px_per_um` | `null` | Set via the Calibrate Scale step |
| `flake_target` / `background` | `hsv_lower`, `hsv_upper` | `null` | Set via the color picker |
| | `exclude_h_range` | `[0, 0]` | Hue sub-range to exclude from the mask |
| `flake_target` | `min_area_um2` / `max_area_um2` | `2.0` / `200.0` | Accepted flake area range |
| | `max_aspect_ratio` | `15.0` | Reject long, thin false positives |
| `isolation_rules` | `clearance_method` | `dilation` | `circle` (radial) or `dilation` (shape-following) |
| | `clearance_radius_um` | `5.0` | Required debris-free clearance around a flake |
| | `halo_buffer_um` | `0.5` | Buffer excluded from the clearance check near the flake edge |
| | `ignore_micro_debris_um2` | `0.5` | Ignore obstacles smaller than this |
| | `noise_tolerance_percent` | `5.0` | % of the clearance ring allowed to contain debris before rejecting |
| `advanced_filters` | `strict_blob_color_verification` | `true` | Re-check each blob's average color against the target bounds |
| | `strict_color_tolerance` | `10` | Tolerance (0–255) for the strict check |
| | `morphology_kernel_um` | `0.5` | Open/close kernel size |
| | `color_check_erosion_um` | `0.2` | Erosion applied before sampling color (avoids edge pixels) |
| `system_settings` | `use_multiprocessing` | `true` | Toggle multi-core batch processing |
| | `max_workers` | `CPU count − 2` | Worker process count |

---

## Known Limitations

- **HSV thresholding, not ML.** Works well for high-contrast materials (e.g. graphene on SiO₂) but is more sensitive to lighting, camera white balance, and substrate variation than deep-learning-based detectors. Re-calibrate per imaging session.
- **Filename convention is lab-specific.** `parse_grid_to_xy` / `get_mark_nominal_uv` assume a grid-encoded filename scheme (e.g. `01_ShiftN.jpg`, `0110_ShiftN.jpg`) tied to a specific sample layout. Adjust these functions if your naming scheme differs.
- **`.ssc` output format is tool-specific.** The `[SLOWSCAN]` section format was written for a particular stage/EBL control software. Confirm compatibility with your own tool before using the generated files directly.
- **No automated test suite yet.** Validate detection accuracy against a manually-annotated sample set before relying on the output for publication-grade data.

Contributions that generalize any of the above are very welcome.

---

## Contributing

Issues and pull requests are welcome — especially around:
- Generalizing the EBL filename/coordinate convention
- Adding support for additional stage/alignment file formats
- Accuracy benchmarking against manually labeled datasets

## License

Released under the [MIT License](LICENSE) — see the `LICENSE` file for details.

## Citation

If this tool is useful in your published research, a link back to this repository (or a citation, once one exists — e.g. via a `CITATION.cff` or a short JOSS/SoftwareX write-up) is appreciated.
