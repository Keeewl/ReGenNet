# refine/commands

Current Stage2-lite command set covers the data-entry bridge and lightweight
development utilities.

Files:

- `01_build_reaction_data_interx_raw_meta.sh`
  Builds InterX `reaction_data.npz` directly from a frozen Stage1 model plus raw restored-shape metadata.
- `02_build_reaction_data_interx_meta_package.sh`
  Builds InterX `reaction_data.npz` from a frozen Stage1 model plus a pre-exported restoration package.
- `03_smoke_test_reaction_data.sh`
  Verifies that `ReactionDataDataset` and restored metadata extraction can read a generated pack.
- `04_show_reaction_data_fields.sh`
  Dumps the fields, shapes, and dtypes stored in a generated pack.
- `06_visualize_stage2_pack.sh`
  Converts a Stage2-Lite `refined_pack.npz` into viewer-ready `P1.npz/P2.npz`
  clip folders for `visualize/viewer/data_viewer.py`.

Notes:

- For future Stage2-lite experiments, InterX is the preferred dataset because the new restore-shape chain is defined around restored pair space and SMPL-X metadata.
- The repository currently does not include `dataset/interx/annots/interaction_order.pkl` or the raw `P1.npz/P2.npz` clip folders, so those must be provided externally for a full restored-space build.
