# refine

`refine/` is the location for the new independent Stage2-lite implementation.

Rules:

- `refine/` must not runtime import `stage2_old.*`.
- `refine/` must not runtime import `model.contact.*`.
- `refine/` must not runtime import `model.crefine.*`.
- If a useful structure exists in the archive, copy it into `refine/` and convert it to local imports.

Current state:

- This directory is only a scaffold.
- No new Stage2-lite model, loss, training loop, or inference logic is implemented here yet.
- `refine/data/` now contains the new independent `reaction_data` bridge package for Stage1 -> Stage2-lite.
- `refine/commands/` contains the current runnable Stage2-lite data-entry commands, with InterX as the preferred dataset path.
- The current `reaction_data` build chain is considered operational for the present Stage2-lite baseline.
  The repository has already produced a normal build result at `refine/dataset/train/reaction_data.npz`, and the corresponding run completed without restoration warnings.
  `build_reaction_data.py` is therefore left as-is for now; if restored-space constraints need to be hardened later, that should be done in one follow-up pass rather than piecemeal.
- `refine/model/windows.py` is currently the baseline joint-based window selector.
  It uses joint-level hand/target geometry plus simple motion cues to build deterministic contact-critical windows in restored pair space.
  This is not yet the final region-aware or mesh-aware selector; it is the stable baseline to build later Stage2-lite modules on top of.
- `refine/eval/window_audit.py` is a development-time audit helper, not a final paper metric implementation.
  Metrics such as `gt_contact_coverage` and `window_precision_proxy` are currently computed by re-running the same selector on `gt_motion` and comparing proxy GT windows against predicted windows.
  In other words, the current audit is selector-vs-selector proxy analysis, not strict GT contact-label evaluation.
