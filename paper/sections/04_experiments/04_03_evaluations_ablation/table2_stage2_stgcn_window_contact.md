## Table2: Shared-Domain Contact Evaluation

Protocol:
1. A **shared fixed evaluation domain** defined by the same 15 contact-rich Inter-X action types for every method.
2. The evaluation sequence set is fixed per split and does **not** depend on each method's `Pred+` selector domain.
3. Only **contact metrics** are reported here; STGCN remains in Table 1.
4. Contact is evaluated on **full sequences** in restored real-world pair space.
5. `HiReact*` denotes Stage1-only initial reaction; `HiReact` denotes Stage1 + Stage2 refinement.

---

<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Train-conditioned</th>
      <th colspan="4">Test-conditioned</th>
    </tr>
    <tr>
      <th>Contact F1↑</th>
      <th>Recall↑</th>
      <th>Contact Distance↓</th>
      <th>Contact Ratio↑</th>
      <th>Contact F1↑</th>
      <th>Recall↑</th>
      <th>Contact Distance↓</th>
      <th>Contact Ratio↑</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>reference</td>
      <td>reference</td>
      <td>0.0162257</td>
      <td>0.489563</td>
      <td>reference</td>
      <td>reference</td>
      <td>reference</td>
      <td>reference</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>0.236937</td>
      <td>0.180678</td>
      <td>0.223879</td>
      <td>0.339000</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>0.273077</td>
      <td>0.204079</td>
      <td>0.225612</td>
      <td>0.307119</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>0.343636</td>
      <td>0.260900</td>
      <td>0.209453</td>
      <td>0.316369</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td><u>0.672048</u></td>
      <td><u>0.613705</u></td>
      <td><u>0.048985</u></td>
      <td><b>0.449079</b></td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
      <td>TBD</td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.781628</u></td>
      <td><u>0.730007</u></td>
      <td><u>0.034261</u></td>
      <td>0.442139</td>
      <td>0.219406</td>
      <td>0.168626</td>
      <td>0.252062</td>
      <td>0.331177</td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.794553</b></td>
      <td><b>0.749415</b></td>
      <td><b>0.032419</b></td>
      <td><u>0.447734</u></td>
      <td>0.220227</td>
      <td>0.170322</td>
      <td>0.251174</td>
      <td>0.334928</td>
    </tr>
  </tbody>
</table>

Notes:

- Train numbers currently reuse the existing deterministic train split evaluation.
- Test numbers shown for `HiReact*` and `HiReact` come from the aligned mainchain test source (`save/cnet_v5/interx_smplx_online_exp1/model000200000.pt`) under the shared fixed-domain protocol.
- `AGRoL`, `MDM`, `MDM-GRU`, and `ReGenNet` test rows are left as `TBD` until the same fixed-domain test pipeline is run for each baseline.
- `Contact Distance` is the mean hand-target minimum distance on GT-contact regions / frames in restored pair space.
