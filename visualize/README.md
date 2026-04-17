# Visualization

ReGenNet now uses the Inter-X-based viewer stack as the primary visualization path.

## Structure

- `visualize/viewer/`
  Main interactive SMPL-X viewer.
- `visualize/viewer/snapshot_viewer.py`
  Manual multi-frame snapshot / teaser viewer for paper-style static layouts.
- `visualize/converters/`
  Convert `results.npy` or processed h5 data into viewer-ready `P1.npz / P2.npz`.
- `visualize/readme/dataset.md`
  Commands for raw dataset GT and processed h5 visualization.
- `visualize/readme/gen.md`
  Commands for model output visualization from `outputs/`.
- `visualize/legacy/`
  Archived old ReGenNet render/export utilities.

## Command Docs

- Dataset / processed data workflows: `visualize/readme/dataset.md`
- Generation / outputs workflows: `visualize/readme/gen.md`

## Notes

- `visualize/viewer/interx_data`, `interx_texts`, and `chi3d_data` are symlinks to the sibling `Inter-X` repo.
- `visualize/viewer/` no longer carries its own `body_models/` copy; it resolves SMPL-X assets from repo-root `body_models/`.
- Older rendering/export scripts remain available under `visualize/legacy/README.md`.
