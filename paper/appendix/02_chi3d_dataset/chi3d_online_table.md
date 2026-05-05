<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Train-conditioned</th>
      <th colspan="4">Test-conditioned</th>
    </tr>
    <tr>
      <th>FID↓</th><th>Acc.→</th><th>Div.→</th><th>Multimod.→</th>
      <th>FID↓</th><th>Acc.→</th><th>Div.→</th><th>Multimod.→</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gt</td>
      <td>0.00 ± 0.00</td><td>1.00 ± 0.00</td><td>19.400 ± 0.4558</td><td>5.690 ± 0.1252</td>
      <td>0.00 ± 0.00</td><td>0.434 ± 0.0078</td><td>11.698 ± 0.6888</td><td>7.057 ± 0.4563</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td><b>0.322 ± 0.0089</b></td><td><b>1.00 ± 0.00</b></td><td><u>19.217 ± 0.4708</u></td><td><b>5.810 ± 0.1220</b></td>
      <td><u>21.254 ± 8.3158</u></td><td><u>0.381 ± 0.0089</u></td><td><b>9.759 ± 0.9362</b></td><td><u>6.378 ± 0.5756</u></td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.355 ± 0.0071</u></td><td><b>1.00 ± 0.00</b></td><td><b>19.298 ± 0.4796</b></td><td><u>5.825 ± 0.1278</u></td>
      <td><b>16.907 ± 8.7013</b></td><td><b>0.399 ± 0.0068</b></td><td><u>9.722 ± 0.7324</u></td><td><b>6.475 ± 0.4119</b></td>
    </tr>
  </tbody>
</table>