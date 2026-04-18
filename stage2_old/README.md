# stage2_old

`stage2_old/` is the archived reference for the previous Stage2 implementations.

It is split into three areas:

- `common/`: shared Stage2-only data, restored-space, geometry, eval, and cache tools.
- `proposal/`: the old proposal and contact-refiner pipeline.
- `crefine/`: the old geometry-first diffusion crefine pipeline.

Rules:

- Treat this package as archived reference code.
- New work for Stage2-lite must go under `refine/`.
- Code under `refine/` must not runtime import `stage2_old.*`.
- If a useful structure exists here, copy it into `refine/` and switch it to local imports.

Compatibility:

- Historical entrypoints under `model/`, `train/`, `tools/`, `data/`, and `eval/` are preserved as thin wrappers.
- The wrappers now point into `stage2_old/` so existing old commands keep working.
