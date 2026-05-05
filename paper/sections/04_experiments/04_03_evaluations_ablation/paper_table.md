Table 1:

<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">Train conditioned</th>
      <th colspan="4">Test conditioned</th>
    </tr>
    <tr>
      <th>FID↓</th>
      <th>Acc.↑</th>
      <th>Div.→</th>
      <th>Multimod.→</th>
      <th>FID↓</th>
      <th>Acc.↑</th>
      <th>Div.→</th>
      <th>Multimod.→</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Real</td>
      <td>-</td>
      <td>0.987 ± 0.00</td>
      <td>21.693 ± 0.2708</td>
      <td>12.305 ± 0.0335</td>
      <td>-</td>
      <td>0.726 ± 0.0002</td>
      <td>19.919 ± 0.1911</td>
      <td>13.809 ± 0.0565</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>11.73 ± 0.415</td>
      <td>0.858 ± 0.0002</td>
      <td>19.347 ± 0.1288</td>
      <td>12.5 ± 0.0468</td>
      <td>13.724 ± 0.2459</td>
      <td>0.564 ± 0.0001</td>
      <td>17.952 ± 0.2396</td>
      <td>13.412 ± 0.0526</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>7.054 ± 0.4626</td>
      <td>0.871 ± 0.0001</td>
      <td>19.461 ± 0.1709</td>
      <td>12.473 ± 0.048</td>
      <td>8.454 ± 0.1578</td>
      <td>0.581 ± 0.0002</td>
      <td>18.026 ± 0.2936</td>
      <td>13.326 ± 0.0541</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>5.454 ± 0.2772</td>
      <td>0.911 ± 0.0002</td>
      <td>19.96 ± 0.2379</td>
      <td>12.398 ± 0.0372</td>
      <td>9.805 ± 0.2196</td>
      <td>0.579 ± 0.0001</td>
      <td>18.017 ± 0.2149</td>
      <td>13.304 ± 0.0369</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td>1.399 ± 0.0252</td>
      <td>0.968 ± 0.0001</td>
      <td><u>21.111 ± 0.1425</u></td>
      <td><b>12.348 ± 0.0452</b></td>
      <td>8.642 ± 0.28</td>
      <td><u>0.598 ± 0.0002</u></td>
      <td>18.339 ± 0.338</td>
      <td>13.404 ± 0.0451</td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.941 ± 0.0131</u></td>
      <td><u>0.97 ± 0.0001</u></td>
      <td><b>21.222 ± 0.1602</b></td>
      <td><u>12.375 ± 0.0392</u></td>
      <td><b>4.007 ± 0.0829</b></td>
      <td><b>0.601 ± 0.0002</b></td>
      <td><b>18.838 ± 0.2493</b></td>
      <td><b>13.812 ± 0.0464</b></td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.627 ± 0.0067</b></td>
      <td><b>0.976 ± 0.0006</b></td>
      <td>20.844 ± 0.0228</td>
      <td>11.818 ± 0.0089</td>
      <td><u>4.355 ± 0.0394</u></td>
      <td>0.591 ± 0.0022</td>
      <td><u>18.689 ± 0.0507</u></td>
      <td><u>13.588 ± 0.0233</u></td>
    </tr>
  </tbody>
</table>


Table 2:

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
      <th>Contact Ratio→</th>
      <th>Contact F1↑</th>
      <th>Recall↑</th>
      <th>Contact Distance↓</th>
      <th>Contact Ratio→</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Real</td>
      <td>-</td>
      <td>-</td>
      <td>0.0160</td>
      <td>0.4171</td>
      <td>-</td>
      <td>-</td>
      <td>0.0168</td>
      <td>0.3886</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>0.2369</td>
      <td>0.1807</td>
      <td>0.2239</td>
      <td>0.3390</td>
      <td>0.1117</td>
      <td>0.0842</td>
      <td>0.3504</td>
      <td>0.2526</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>0.2731</td>
      <td>0.2041</td>
      <td>0.2256</td>
      <td>0.3071</td>
      <td>0.1353</td>
      <td>0.0993</td>
      <td>0.3521</td>
      <td>0.2375</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>0.3436</td>
      <td>0.2609</td>
      <td>0.2095</td>
      <td>0.3164</td>
      <td>0.1274</td>
      <td>0.0924</td>
      <td>0.3927</td>
      <td>0.2273</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td>0.6720</td>
      <td>0.6137</td>
      <td>0.0490</td>
      <td>0.4491</td>
      <td>0.1690</td>
      <td>0.1266</td>
      <td>0.3074</td>
      <td>0.2466</td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.7816</u></td>
      <td><u>0.7300</u></td>
      <td><u>0.0343</u></td>
      <td><b>0.4421</b></td>
      <td><u>0.1997</u></td>
      <td><u>0.1536</u></td>
      <td><u>0.2828</u></td>
      <td><u>0.2585</u></td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.7946</b></td>
      <td><b>0.7494</b></td>
      <td><b>0.0324</b></td>
      <td><u>0.4477</u></td>
      <td><b>0.2012</b></td>
      <td><b>0.1557</b></td>
      <td><b>0.2818</b></td>
      <td><b>0.2617</b></td>
    </tr>
  </tbody>
</table>
