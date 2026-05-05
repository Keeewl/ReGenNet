## Contact-Rich Subset Construction

To support the Stage II contact evaluation protocol, we construct a contact-rich subset from the full Inter-X training split. The goal of this subset is not to improve selector quality by design, but to define a more appropriate evaluation domain for local contact refinement. We therefore focus only on dataset-level statistics that characterize contact density and action coverage, without using any selector/window quality metrics.

The full Inter-X training split contains `40` action categories and `9110` sequences. From this space, we select `15` action categories with frequent and explicit local physical contacts, including hand-hand, hand-arm, hand-body, support/help, and local face/cheek interaction patterns. The resulting subset contains `2842` sequences and `13190` GT contact segments.

Compared with the full training split, the subset remains substantially smaller in sequence count while preserving most of the contact content. Specifically, the subset contains only `31.2%` of all sequences (`2842 / 9110`), but retains `70.4%` of all GT contact segments (`13190 / 18736`). As a result, the average GT contact density increases from `2.06` to `4.64` GT contact segments per sequence, which is about `2.25×` higher than the full dataset. These statistics indicate that the subset concentrates the interaction patterns that are most relevant to local contact-aware refinement, while still covering diverse hand-centric interaction types.

<table class="paper-table">
  <thead>
    <tr>
      <th>Statistic</th>
      <th>Full Inter-X Train</th>
      <th>15-Action Contact-Rich Subset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Action types</td>
      <td>40</td>
      <td>15</td>
    </tr>
    <tr>
      <td>Sequences</td>
      <td>9110</td>
      <td>2842</td>
    </tr>
    <tr>
      <td>GT contact segments</td>
      <td>18736</td>
      <td>13190</td>
    </tr>
    <tr>
      <td>Sequence retention</td>
      <td>100.0%</td>
      <td>31.2%</td>
    </tr>
    <tr>
      <td>GT contact retention</td>
      <td>100.0%</td>
      <td>70.4%</td>
    </tr>
    <tr>
      <td>GT contact segments / sequence</td>
      <td>2.06</td>
      <td>4.64</td>
    </tr>
    <tr>
      <td>Relative contact density</td>
      <td>1.00×</td>
      <td>2.25×</td>
    </tr>
  </tbody>
</table>

The selected `15` action categories are:

```text
A028 Hand wrestling
A025 Carry on back
A001 Handshake
A009 Sit on leg
A021 Dance
A000 Hug
A008 Pull
A019 Support with hand
A023 Shoulder to shoulder
A035 Help up
A027 Massaging leg
A022 Link arms
A003 Grab
A016 High-five
A034 Kiss on cheek
```

These action types cover several distinct contact patterns:

- hand-hand direct contact: `Handshake`, `High-five`, `Hand wrestling`
- hand-arm / upper-body local contact: `Grab`, `Link arms`, `Shoulder to shoulder`
- hand-body support / carry / help interaction: `Carry on back`, `Support with hand`, `Help up`, `Sit on leg`
- pull / close-range control interaction: `Pull`, `Hug`
- local face / cheek contact: `Kiss on cheek`
- sustained or multi-contact interaction: `Dance`, `Massaging leg`

Together, these statistics support the rationale that the subset is not an arbitrary sample filter, but a contact-focused evaluation domain with substantially higher local contact density and broad coverage of hand-centric interaction types.
