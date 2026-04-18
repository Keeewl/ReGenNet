# reaction_data

`reaction_data` is the new Stage1 -> Stage2-lite bridge package.

It replaces legacy names such as `coarse_cache`, `restored_cache`, and `blueprint_cache` as the new primary data entrypoint. The goal is to make the new `refine/` pipeline self-contained while still reusing the shared Stage1 dataset/model/sampling chain where appropriate.

What lives here:

- `build_reaction_data.py`: runs a frozen Stage1 model once and packs actor motion, coarse reactor motion, GT reactor motion, lengths, indices, and restored-space metadata into `reaction_data.npz`.
- `schema.py`: defines the `reaction_data schema`.
- `restored_space.py`: local restored-pair-space helpers copied/refactored from the archived Stage2 logic.
- `cache_dataset.py` and `collate.py`: minimal Stage2-lite read path for the new pack.

Current status:

- The current build chain has already produced a normal baseline pack at `refine/dataset/train/reaction_data.npz`.
- That build is treated as the current "known good" Stage2-lite data-entry result.
- No restoration warning was observed in that successful build path, so the present `build_reaction_data.py` logic is intentionally left unchanged for now.
- If restored-space handling needs to become a stricter hard requirement later, that cleanup should be done in one dedicated follow-up pass.

Isolation rule:

- New `refine/` code may reference archived `stage2_old` for design, but it must not runtime import archived Stage2 modules.
