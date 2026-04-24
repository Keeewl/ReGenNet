# Stage2 Lightweight Status and Next Update

Date: 2026-04-24

## 1. Current Stage2 Model Is Lightweight

The current Stage2 refiner is still lightweight.

Reasoning:

```text
1. it is a window-level residual refiner, not a full-sequence generator
2. it only runs on selected contact windows, not on every frame equally
3. the backbone is still relatively small and fast
4. training speed is high and GPU memory usage is low
5. the heavy parts in the pipeline are evaluation / visualization / SMPL-X contact checks,
   not the refiner forward itself
```

Therefore, at deployment / sampling time:

```text
Stage1 -> coarse full sequence
Stage2 -> refine only selected windows
stitch -> merge residuals back into the sequence
```

This should not add a large latency increase by itself.

The parts that are expensive are:

```text
1. offline evaluation
2. restored-shape contact measurement
3. SMPL-X based analysis / visualization
```

not the Stage2 model inference itself.

## 2. What the Stage2 Model Actually Is

The Stage2 refine model is currently the `refiner_v2` line:

```text
refine_v2/model/refiner_v2.py
refine_v2/model/condition_encoder.py
refine_v2/model/losses_v2.py
```

The full Stage2 system is:

```text
selector/window/subset
+ refiner_v2
+ residual stitching
+ full-sequence evaluation
```

But the actual trainable model remains:

```text
refiner_v2
```

Current structure remains controlled rather than redundant:

```text
shared temporal backbone
+ contact / hand / region conditioning
+ focused hand/arm residual refinement
```

Even with exp8 interaction, it is still not:

```text
full-body spatial transformer
large autoregressive generator
heavy diffusion-style model
```

So the Stage2 model line remains technically manageable for later code review.

## 3. What the Next Update Should Be

The next update should be treated as:

```text
the final contact-refine update
```

It should not open another large branch.

The recommended direction is:

```text
keep exp8 backbone / selector / subset / eval protocol fixed
```

and do only one meaningful model-side upgrade:

```text
stronger but still lightweight hand-target spatial interaction
```

Specifically:

```text
1. keep the model lightweight
2. do not move to a heavy full spatial transformer
3. do not reopen transl / phase / broad proxy-loss tuning
4. strengthen selected-hand / same-side-arm vs top-k target-region interaction
5. keep final judgment based on full-sequence system evaluation
```

## Final Conclusion

```text
1. current Stage2 is lightweight enough for practical use
2. the trainable Stage2 model is still refiner_v2
3. exp8 already gives a valid system-level Stage2 result
4. the next update should be one final, lightweight, interaction-focused contact-refine upgrade
5. after that, the Stage2 model line should be considered basically fixed
```
