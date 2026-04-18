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
