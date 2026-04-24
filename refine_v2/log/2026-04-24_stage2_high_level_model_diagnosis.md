# Stage2 High-Level Model Diagnosis

Date: 2026-04-24

## Current High-Level Position

At this stage, Stage2 is no longer bottlenecked by:

```text
GT contact labels
selector/window
contact-rich subset
restored-space processing
basic train/eval/visualization pipeline
```

These parts are already strong enough to support Stage2 iteration.

The main bottleneck has moved to:

```text
the refiner model itself
```

More precisely:

```text
how the model represents hand-target interaction
how directly the training objective aligns with contact geometry
how effectively geometry features are consumed by the model
```

## Why The Model Is Now The Main Bottleneck

The current best practical baseline remains:

```text
refiner_v2_exp5_scope_geom_10k
```

This already proves that:

```text
the overall Stage2 framework works
selector/window/subset are not the primary limiting factor
contact quality can be improved beyond coarse
```

However, later experiments show that:

```text
more feature fields alone do not automatically help
proxy contact losses alone do not automatically help
translation/phase tuning is not the main remaining gain source
```

This means the remaining headroom is mainly about:

```text
representation
interaction modeling
task alignment of the model
```

## Current Model Summary

The current Stage2 refiner is:

```text
a window-level residual refiner
```

It takes:

```text
actor_motion_window
coarse_motion_window
hand / primary region / top-k region metadata
coarse contact mask / min-distance
optional geometry feature cache
```

Then it applies:

```text
temporal transformer blocks
condition modulation
group-gated residual scaling
```

And predicts:

```text
pred_motion = coarse_motion + residual_delta
```

This design is fundamentally correct for Stage2 because Stage2 should be:

```text
window-level
local
hand-centric
contact-oriented
residual
```

## Current Model Strengths

### 1. Task-Compatible Structure

The model refines local windows rather than regenerating full sequences.

This matches the Stage2 role:

```text
Stage1 provides coarse reactor motion
Stage2 corrects local contact-relevant errors
```

### 2. Good Scope Control

Through residual scaling and weighted losses, the current model already prefers:

```text
larger hand/arm corrections
smaller body/transl corrections
```

This is important because Stage2 should not become a broad full-body rewrite.

### 3. Fast Iteration

The current pipeline avoids dense differentiable mesh forward inside every
training step, so:

```text
training stays fast
ablation is cheap
iteration speed is practical
```

## Current Model Weaknesses

### 1. Training Target Is Still Not Directly Contact-Geometric

Even with contact-weighted motion losses, the main learning target is still
close to:

```text
make predicted motion closer to GT motion
```

But the real Stage2 target is:

```text
make predicted hand-target contact closer to GT contact
```

These are related, but not equivalent.

This mismatch is now one of the main reasons the model saturates.

### 2. Hand-Target Interaction Is Not Explicit Enough

Current conditioning includes hand / region / top-k / distance information,
but the model still does not explicitly build:

```text
selected hand token
<-> top-k target region token
```

interaction structure in a strong, direct way.

As a result, geometry information can be added without necessarily being used
effectively.

### 3. Contact-Relevant Local Geometry Is Still Averaged

Many current geometry features are still relatively coarse:

```text
centroid-style vectors
aggregated distances
global top-k statistics
```

But actual contact often happens through:

```text
palm
fingertips
local nearest points
```

So the model can still miss the truly decisive local geometry.

### 4. Output Expression Is Still Too Shared

Even though residual scope is gated by joint groups, the core representation is
still mostly shared.

This means:

```text
hand/arm correction ability is not isolated strongly enough
```

## Does The Current Model Use Spatial Attention?

Strictly speaking:

```text
no, not in the full sense
```

The current model mainly has:

```text
temporal attention over frame tokens
condition modulation
group-based residual control
```

It does **not** have explicit joint-level or part-level spatial attention like:

```text
joint tokens per frame
joint-joint spatial self-attention
hand-target cross-attention as a first-class module
space-time transformer blocks
```

So:

```text
temporal attention exists
explicit spatial attention does not
```

## Should Full Spatial Attention Be Added?

The answer is:

```text
not as a full heavy architecture change
```

A full spatial-attention or space-time-transformer redesign would likely be:

```text
too large
too slow to validate
too risky given current time constraints
not clearly necessary for the current Stage2 scope
```

This does not mean spatial structure is unimportant.

Instead, it means the right next step is:

```text
lightweight task-specific spatial interaction
```

not:

```text
full generic spatial transformer
```

## Most Reasonable Model Upgrade Direction

The next model upgrade should keep the exp5 backbone spirit, but add a more
explicit hand-target interaction module.

Recommended direction:

```text
selected-hand interaction tokens
same-side-arm interaction tokens
top-k target region tokens
small cross-attention or gated interaction block
inject this interaction into the existing temporal backbone
```

This is effectively a lightweight, task-specific spatial modeling upgrade.

It is better aligned with Stage2 than a full spatial transformer because it
directly models the interaction that matters most:

```text
reactor hand/arm
vs
actor target contact region
```

## High-Level Conclusion

The next Stage2 model upgrade is justified.

It is both:

```text
aligned with the Stage2 goal
and
consistent with the current empirical bottleneck
```

The main problem is no longer:

```text
windowing
subset selection
translation continuity
```

It is now:

```text
the model does not yet represent hand-target contact interaction explicitly
enough
```

## Final Recommendation

Do not invest in a heavy full spatial-attention redesign.

Instead, if one more meaningful model upgrade is made, it should be:

```text
exp5-style backbone
+ lightweight hand-target interaction module
+ selected-hand / same-side-arm focused representation
+ direct but light contact-aware supervision
```

This is the most reasonable and efficient path if the goal is to approach the
Stage2 ceiling within 1-2 days rather than open-ended iteration.
