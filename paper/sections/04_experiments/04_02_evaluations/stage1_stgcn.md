<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method (200K)</th>
      <th colspan="4">Train conditioned</th>
      <th colspan="4">Test conditioned</th>
    </tr>
    <tr>
      <th>FID↓</th><th>Acc.↑</th><th>Div.→</th><th>Multimod.→</th>
      <th>FID↓</th><th>Acc.↑</th><th>Div.→</th><th>Multimod.→</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Real</td>
      <td>-</td><td>0.987 ± 0.00</td><td>21.693 ± 0.2708</td><td>12.305 ± 0.0335</td>
      <td>-</td><td>0.726 ± 0.0002</td><td>19.919 ± 0.1911</td><td>13.809 ± 0.0565</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>11.73 ± 0.415</td><td>0.858 ± 0.0002</td><td>19.347 ± 0.1288</td><td>12.5 ± 0.0468</td>
      <td>13.724 ± 0.2459</td><td>0.564 ± 0.0001</td><td>17.952 ± 0.2396</td><td><u>13.412 ± 0.0526</u></td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>7.054 ± 0.4626</td><td>0.871 ± 0.0001</td><td>19.461 ± 0.1709</td><td>12.473 ± 0.048</td>
      <td><u>8.454 ± 0.1578</u></td><td>0.581 ± 0.0002</td><td>18.026 ± 0.2936</td><td>13.326 ± 0.0541</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>5.454 ± 0.2772</td><td>0.911 ± 0.0002</td><td>19.96 ± 0.2379</td><td>12.398 ± 0.0372</td>
      <td>9.805 ± 0.2196</td><td>0.579 ± 0.0001</td><td>18.017 ± 0.2149</td><td>13.304 ± 0.0369</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td><u>1.399 ± 0.0252</u></td><td><u>0.968 ± 0.0001</u></td><td><u>21.111 ± 0.1425</u></td><td><b>12.348 ± 0.0452</b></td>
      <td>8.642 ± 0.28</td><td><u>0.598 ± 0.0002</u></td><td><u>18.339 ± 0.338</u></td><td>13.404 ± 0.0451</td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.941 ± 0.0131</b></td><td><b>0.97 ± 0.0001</b></td><td><b>21.222 ± 0.1602</b></td><td><u>12.375 ± 0.0392</u></td>
      <td><b>4.007 ± 0.0829</b></td><td><b>0.601 ± 0.0002</b></td><td><b>18.838 ± 0.2493</b></td><td><b>13.812 ± 0.0464</b></td>
    </tr>
  </tbody>
</table>