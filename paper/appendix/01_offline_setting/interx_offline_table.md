<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
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
      <td>0.00 ± 0.00</td><td>0.987 ± 0.00</td><td>21.693 ± 0.2708</td><td>12.305 ± 0.0335</td>
      <td>0.00 ± 0.00</td><td>0.726 ± 0.0002</td><td>19.919 ± 0.1911</td><td>13.809 ± 0.0565</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>7.275 ± 0.2868</td><td>0.912 ± 0.0002</td><td>20.024 ± 0.1288</td><td>12.452 ± 0.0375</td>
      <td>12.941 ± 0.2686</td><td>0.566 ± 0.0001</td><td>18.033 ± 0.2050</td><td>13.498 ± 0.0396</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>1.706 ± 0.0520</td><td>0.961 ± 0.0001</td><td>20.919 ± 0.1500</td><td><u>12.371 ± 0.0452</u></td>
      <td>8.882 ± 0.3447</td><td>0.579 ± 0.0001</td><td>18.162 ± 0.2600</td><td>13.449 ± 0.0425</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>1.657 ± 0.0358</td><td><b>0.965 ± 0.0001</b></td><td>20.980 ± 0.1694</td><td><b>12.370 ± 0.0378</b></td>
      <td>6.909 ± 0.1531</td><td><b>0.620 ± 0.0002</b></td><td>18.943 ± 0.2690</td><td>13.653 ± 0.0689</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td><u>1.604 ± 0.0289</u></td><td><u>0.964 ± 0.0001</u></td><td><u>20.987 ± 0.1561</u></td><td>12.374 ± 0.0397</td>
      <td><u>6.328 ± 0.1497</u></td><td>0.608 ± 0.0001</td><td><u>19.075 ± 0.3301</u></td><td><b>13.725 ± 0.0567</b></td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><b>1.296 ± 0.0174</b></td><td><b>0.965 ± 0.0001</b></td><td><b>21.131 ± 0.1729</b></td><td>12.449 ± 0.0401</td>
      <td><b>3.054 ± 0.0370</b></td><td><u>0.615 ± 0.0001</u></td><td><b>19.593 ± 0.2917</b></td><td><u>13.954 ± 0.0581</u></td>
    </tr>
  </tbody>
</table>


<table class="paper-table">
  <thead>
    <tr>
      <th rowspan="2">Method</th>
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
      <td>0.00 ± 0.00</td><td>0.987 ± 0.00</td><td>21.693 ± 0.2708</td><td>12.305 ± 0.0335</td>
      <td>0.00 ± 0.00</td><td>0.726 ± 0.0002</td><td>19.919 ± 0.1911</td><td>13.809 ± 0.0565</td>
    </tr>
    <tr>
      <td>AGRoL</td>
      <td>7.275 ± 0.2868</td><td>0.912 ± 0.0002</td><td>20.024 ± 0.1288</td><td>12.452 ± 0.0375</td>
      <td>12.941 ± 0.2686</td><td>0.566 ± 0.0001</td><td>18.033 ± 0.2050</td><td>13.498 ± 0.0396</td>
    </tr>
    <tr>
      <td>MDM</td>
      <td>1.657 ± 0.0358</td><td><u>0.965 ± 0.0001</u></td><td>20.980 ± 0.1694</td><td><b>12.370 ± 0.0378</b></td>
      <td>6.909 ± 0.1531</td><td><b>0.620 ± 0.0002</b></td><td>18.943 ± 0.2690</td><td>13.653 ± 0.0689</td>
    </tr>
    <tr>
      <td>MDM-GRU</td>
      <td>1.706 ± 0.0520</td><td>0.961 ± 0.0001</td><td>20.919 ± 0.1500</td><td><u>12.371 ± 0.0452</u></td>
      <td>8.882 ± 0.3447</td><td>0.579 ± 0.0001</td><td>18.162 ± 0.2600</td><td>13.449 ± 0.0425</td>
    </tr>
    <tr>
      <td>ReGenNet</td>
      <td>1.604 ± 0.0289</td><td>0.964 ± 0.0001</td><td><u>20.987 ± 0.1561</u></td><td>12.374 ± 0.0397</td>
      <td>6.328 ± 0.1497</td><td>0.608 ± 0.0001</td><td>19.075 ± 0.3301</td><td><b>13.725 ± 0.0567</b></td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>1.296 ± 0.0174</u></td><td><u>0.965 ± 0.0001</u></td><td><b>21.131 ± 0.1729</b></td><td>12.449 ± 0.0401</td>
      <td><b>3.054 ± 0.0370</b></td><td><u>0.615 ± 0.0001</u></td><td><u>19.593 ± 0.2917</u></td><td>13.954 ± 0.0581</td>
    </tr>
    <tr>
      <td>HiReact</td>
      <td><b>0.897 ± 0.0003</b></td><td><b>0.970 ± 0.00</b></td><td>20.829 ± 0.0065</td><td>11.849 ± 0.0005</td>
      <td><u>3.280 ± 0.0092</u></td><td>0.611 ± 0.0001</td><td><b>19.899 ± 0.0306</b></td><td><u>13.916 ± 0.0035</u></td>
    </tr>
  </tbody>
</table>