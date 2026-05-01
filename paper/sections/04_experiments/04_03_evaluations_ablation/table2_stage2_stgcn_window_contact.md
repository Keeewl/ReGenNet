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
      <th>Contact F1↑</th>
      <th>Recall↑</th>
      <th>Contact Distance↓</th>
      <th>Contact Ratio↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Real</td>
      <td>-</td>
      <td>0.985878</td>
      <td>21.559633</td>
      <td>4.758754</td>
      <td>reference</td>
      <td>reference</td>
      <td>0.0162257</td>
      <td>0.489563</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>7.881151</td>
      <td>0.860121</td>
      <td>19.777626</td>
      <td>4.928485</td>
      <td>0.236937</td>
      <td>0.180678</td>
      <td>0.223879</td>
      <td>0.339000</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>6.199835</td>
      <td>0.868863</td>
      <td>19.613888</td>
      <td>4.925842</td>
      <td>0.273077</td>
      <td>0.204079</td>
      <td>0.225612</td>
      <td>0.307119</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>4.426496</td>
      <td>0.915266</td>
      <td>20.175209</td>
      <td>4.823986</td>
      <td>0.343636</td>
      <td>0.260900</td>
      <td>0.209453</td>
      <td>0.316369</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td>1.219376</td>
      <td>0.968393</td>
      <td><u>20.897764</u></td>
      <td><b>4.808750</b></td>
      <td><u>0.672048</u></td>
      <td><u>0.613705</u></td>
      <td><u>0.048985</u></td>
      <td><b>0.449079</b></td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.474216</u></td>
      <td><u>0.977808</u></td>
      <td><u>21.202896</u></td>
      <td>4.836596</td>
      <td><u>0.781628</u></td>
      <td><u>0.730007</u></td>
      <td><u>0.034261</u></td>
      <td>0.442139</td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.295525</b></td>
      <td><b>0.978480</b></td>
      <td><b>21.273443</b></td>
      <td><u>4.820340</u></td>
      <td><b>0.794553</b></td>
      <td><b>0.749415</b></td>
      <td><b>0.032419</b></td>
      <td><u>0.447734</u></td>
    </tr>
  </tbody>
</table>

Notes:

- `HiReact*` denotes the Stage1-only initial reaction result before Stage2 refinement.
- `AGRoL`, `MDM`, `MDM-GRU`, and `ReGenNet` do not include the proposed Stage2 refinement; therefore only one full-sequence row is reported for each baseline.

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
