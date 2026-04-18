# reaction_data

`reaction_data` is the new Stage1 -> Stage2-lite bridge package.

It replaces legacy names such as `coarse_cache`, `restored_cache`, and `blueprint_cache` as the new primary data entrypoint. The goal is to make the new `refine/` pipeline self-contained while still reusing the shared Stage1 dataset/model/sampling chain where appropriate.

What lives here:

- `build_reaction_data.py`: runs a frozen Stage1 model once and packs actor motion, coarse reactor motion, GT reactor motion, lengths, indices, and restored-space metadata into `reaction_data.npz`.
- `schema.py`: defines the `reaction_data schema`.
- `restored_space.py`: local restored-pair-space helpers copied/refactored from the archived Stage2 logic.
- `cache_dataset.py` and `collate.py`: minimal Stage2-lite read path for the new pack.

Isolation rule:

- New `refine/` code may reference archived `stage2_old` for design, but it must not runtime import archived Stage2 modules.
