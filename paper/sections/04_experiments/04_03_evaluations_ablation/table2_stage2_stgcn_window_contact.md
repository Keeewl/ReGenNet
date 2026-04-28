## Table2: HiReact stage2 STGCN metric and contact metric

Protocol: 
1. A **subset** of rich contact Inter-X (15-contact-rich subset).
2. STGCN metric in **canonical** space, contact metric in **real-world** space.
3. **Full sequence**, including refined window and stage1 motion.
4. Post-Training (**offline refinement**).

---

<table class="paper-table">
  <thead>
    <tr>
      <th>Method</th>
      <th>FID↓</th>
      <th>Acc.→</th>
      <th>Div.→</th>
      <th>Multimod.→</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Real</td>
      <td>-</td>
      <td>0.985878</td>
      <td>21.559633</td>
      <td>4.758754</td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.474216</u></td>
      <td><u>0.977808</u></td>
      <td><u>21.202896</u></td>
      <td><u>4.836596</u></td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.295525</b></td>
      <td><b>0.978480</b></td>
      <td><b>21.273443</b></td>
      <td><b>4.820340</b></td>
    </tr>
  </tbody>
</table>

---

## Window Candidate Metrics

These metrics are kept here as **window-level candidates** for the Stage2 main
table design. They should be interpreted separately from the full-sequence
STGCN block above.

### Proposal / Window Quality

- `GT-positive Sequence Coverage`: `100%`
  - from `gt_positive_zero_window_ratio = 0.0`
  - meaning: no GT-positive sequence is completely missed by the frozen selector
- `Window Contact Purity`: `0.6857`
  - meaning: about `68.57%` of frames inside predicted windows are true GT-contact frames

### Window Contact Refinement

The following metrics already support `GT / coarse / refined` comparison.

| Metric | GT | Coarse | Refined |
| --- | ---: | ---: | ---: |
| GT-Contact Distance | 0.0158999 | 0.0297788 | 0.0267311 |
| Contact F1 | reference | 0.800338 | 0.821425 |
| Recall | reference | 0.764726 | 0.798100 |
| Top-k Contact F1 | reference | 0.808365 | 0.828897 |

### Notes

- `GT-Contact Distance` is computed only on GT-contact regions / frames.
- `Contact F1`, `Recall`, and `Top-k Contact F1` are window-level contact-quality metrics.
- `GT-positive Sequence Coverage` and `Window Contact Purity` are selector/window quality metrics, not motion-output metrics.

---

## Full-Sequence Contact Candidate Metrics

These metrics are **system-level full-sequence** contact candidates for the
final Stage2 table.

| Metric | GT | Coarse | Refined |
| --- | ---: | ---: | ---: |
| Contact F1 | reference | 0.781628 | 0.794553 |
| Precision | reference | 0.841105 | 0.845477 |
| Recall | reference | 0.730007 | 0.749415 |
| Contact Distance | 0.0162257 | 0.0342606 | 0.0324186 |
| Contact Ratio | 0.489563 | 0.442139 | 0.447734 |
| Avg. Contact Duration | 38.220861 | 33.272267 | 33.523666 |

### Notes

- `Contact Distance` is computed as the mean hand-target minimum distance on GT-contact regions / frames.
- `Contact Ratio` is the fraction of valid frames that are contact frames.
- `Avg. Contact Duration` is the mean duration of contiguous contact segments.
