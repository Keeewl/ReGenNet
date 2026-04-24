# exp8b transl relax plan

## Goal

Run one final calibration experiment on top of exp8 without changing the model architecture.

Hypothesis:

- current exp8 contact improvement may still be limited by overly conservative transl residuals
- allowing slightly larger transl correction may improve final hand-target contact closeness
- boundary continuity must still be protected

## Configuration

Base:

- `exp8_interaction_v1_10k`

Changes:

- `transl_delta_scale = 0.3` (from `0.2`)
- `root_delta_scale = 0.2` (unchanged)
- `lambda_boundary_trans = 1.0` (from `2.0`)
- `lambda_contact_geometry = 0.03` (unchanged)

Everything else stays aligned with exp8.

## Expected behavior

Potential gains:

- slightly stronger transl micro-adjustment in contact windows
- improved contact distance
- possible recall gain on borderline contact cases
- slightly more visually plausible closeness

Main risks:

- weaker boundary control
- slight increase in GT-relative overclose aggressiveness
- Stage2 starting to compensate Stage1 placement too much

## Decision criterion

Keep exp8 as the main result unless exp8b clearly improves:

- window-level contact metrics
- full-sequence contact metrics
- visual closeness

without introducing obvious boundary instability
