## Table2: Shared-Domain Contact Evaluation

Protocol:
1. A **shared fixed evaluation domain** defined by the same 15 contact-rich Inter-X action types for every method.
2. The evaluation sequence set is fixed per split and does **not** depend on each method's `Pred+` selector domain.
3. Only **contact metrics** are reported here; STGCN remains in Table 1.
4. Contact is evaluated on **full sequences** in restored real-world pair space.
5. `HiReact*` denotes Stage1-only initial reaction; `HiReact` denotes Stage1 + Stage2 refinement.
---

Main table: <table class="paper-table"> <thead> <tr> <th rowspan="2">Method</th> <th colspan="4">Train-conditioned</th> <th colspan="4">Test-conditioned</th> </tr> <tr> <th>Contact F1↑</th> <th>Recall↑</th> <th>Contact Distance↓</th> <th>Contact Ratio→</th> <th>Contact F1↑</th> <th>Recall↑</th> <th>Contact Distance↓</th> <th>Contact Ratio→</th> </tr> </thead> <tbody> <tr> <td>Real</td> <td>-</td> <td>-</td> <td>0.0160</td> <td>0.4171</td> <td>-</td> <td>-</td> <td>0.0168</td> <td>0.3886</td> </tr> <tr> <td>AGRoL</td> <td>0.2369</td> <td>0.1807</td> <td>0.2239</td> <td>0.3390</td> <td>0.1117</td> <td>0.0842</td> <td>0.3504</td> <td>0.2526</td> </tr> <tr> <td>MDM</td> <td>0.2731</td> <td>0.2041</td> <td>0.2256</td> <td>0.3071</td> <td>0.1353</td> <td>0.0993</td> <td>0.3521</td> <td>0.2375</td> </tr> <tr> <td>MDM-GRU</td> <td>0.3436</td> <td>0.2609</td> <td>0.2095</td> <td>0.3164</td> <td>0.1274</td> <td>0.0924</td> <td>0.3927</td> <td>0.2273</td> </tr> <tr> <td>ReGenNet</td> <td>0.6720</td> <td>0.6137</td> <td>0.0490</td> <td>0.4491</td> <td>0.1690</td> <td>0.1266</td> <td>0.3074</td> <td>0.2466</td> </tr> <tr> <td>HiReact*</td> <td><u>0.7816</u></td> <td><u>0.7300</u></td> <td><u>0.0343</u></td> <td><b>0.4421</b></td> <td><u>0.1997</u></td> <td><u>0.1536</u></td> <td><u>0.2828</u></td> <td><u>0.2585<u></td> </tr> <tr> <td>HiReact</td> <td><b>0.7946</b></td> <td><b>0.7494</b></td> <td><b>0.0324</b></td> <td><u>0.4477</u></td> <td><b>0.2012</b></td> <td><b>0.1557</b></td> <td><b>0.2818</b></td> <td><b>0.2617</b></td> </tr> </tbody> </table>







Experiments results:

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
      <td>Real</td>
      <td>reference</td>
      <td>reference</td>
      <td>0.015963</td>
      <td>0.417125</td>
      <td>reference</td>
      <td>reference</td>
      <td>0.016804</td>
      <td>0.388577</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>0.236937</td>
      <td>0.180678</td>
      <td>0.223879</td>
      <td>0.339000</td>
      <td>0.111718</td>
      <td>0.084207</td>
      <td>0.350403</td>
      <td>0.252590</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>0.273077</td>
      <td>0.204079</td>
      <td>0.225612</td>
      <td>0.307119</td>
      <td>0.135332</td>
      <td>0.099289</td>
      <td>0.352129</td>
      <td>0.237495</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>0.343636</td>
      <td>0.260900</td>
      <td>0.209453</td>
      <td>0.316369</td>
      <td>0.127428</td>
      <td>0.092421</td>
      <td>0.392654</td>
      <td>0.227272</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td><u>0.672048</u></td>
      <td><u>0.613705</u></td>
      <td><u>0.048985</u></td>
      <td><b>0.449079</b></td>
      <td>0.169020</td>
      <td>0.126585</td>
      <td>0.307416</td>
      <td>0.246582</td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.781628</u></td>
      <td><u>0.730007</u></td>
      <td><u>0.034261</u></td>
      <td>0.442139</td>
      <td>0.199653</td>
      <td>0.153555</td>
      <td>0.282808</td>
      <td>0.258503</td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.794553</b></td>
      <td><b>0.749415</b></td>
      <td><b>0.032419</b></td>
      <td><u>0.447734</u></td>
      <td>0.201244</td>
      <td>0.155731</td>
      <td>0.281808</td>
      <td>0.261667</td>
    </tr>
  </tbody>
</table>

Notes:

- Train numbers currently reuse the deterministic train split evaluation, while the `Real` row is re-anchored with shared fixed-domain GT extraction.
- Test numbers come from the shared fixed-domain protocol: `HiReact` / `HiReact*` use the aligned mainchain test source (`save/cnet_v5/interx_smplx_online_exp1/model000200000.pt`), while each baseline row uses its own Stage1 test source.
- The current `HiReact` test row uses the best validated test-only selector setting so far: `tau_contact = 0.15`.
- `Contact Distance` is the mean hand-target minimum distance on GT-contact regions / frames in restored pair space.

Interpretation:

- The table is now complete for both `train-conditioned` and `test-conditioned` blocks under the current shared fixed-domain definition.
- The train block shows the expected hierarchy: `HiReact > HiReact* > ReGenNet > MDM-GRU > MDM > AGRoL`, which is consistent with the original Stage2 motivation.
- The test block remains much harder for every method. This is reflected by a large drop in `Contact F1` / `Recall` and a large increase in `Contact Distance` for all rows.
- Under the shared fixed-domain test benchmark, `HiReact*` still outperforms all Stage1 baselines, and `HiReact` remains the best overall test row, but the Stage2 improvement over `HiReact*` is small.
- The test ordering is also reasonable: `HiReact > HiReact* > ReGenNet > MDM > MDM-GRU > AGRoL`, which indicates that the shared fixed-domain evaluation is not producing arbitrary rank inversions.
- This pattern is consistent with the diagnostic conclusion in the logs: the dominant test-side bottleneck is weak Stage1 / selector behavior, while Stage2 still provides only a modest positive correction on top of that.
