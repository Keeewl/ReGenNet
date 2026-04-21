# refine_v2/refiner_data

Fast window-level data interface for the first Stage2 refiner.

This module consumes already-built restored-space artifacts:

- `reaction_data.npz`
- `contact_labels_gt.npz`
- `contact_subset/subset_manifest.json`
- `contact_subset/selector_rerun/subset_selector_windows.npz`

The dataset does not recompute SMPL-X vertices or contact distances by default.
It slices existing motion and contact artifacts:

- reaction data -> `actor_motion_window`, `coarse_motion_window`, `gt_motion_window`
- selector artifact -> coarse region contact mask and min-distance curves
- GT label artifact -> GT region contact mask and min-distance curves
- subset manifest / selector windows -> action, bucket, hand, primary region, top-k region metadata

Each sample is one hand-time window. A window is not duplicated into multiple
region samples; it keeps one primary region and top-k region attribution.

Core arrays are cached once when the dataset is constructed. This avoids
repeated `.npz` inflation inside `__getitem__` and keeps the path suitable for
training-time iteration; each DataLoader worker owns its own dataset instance.

`include_xyz=True` is intentionally deferred. The fast path must stay stable
before adding dynamic SMPL-X forward/debug fields.
