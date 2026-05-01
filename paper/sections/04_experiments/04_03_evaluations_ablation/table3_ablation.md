## Table3: ablation

两组分别对Stage1-Generation和Stage2-Refinement


### Group1: Reaction Generation

1. Actor condition: Body-Hand
   Modification:
   replace the original `Global + Multi-Part` actor tokenization with `Body + Hand` actor tokens;
   keep the original actor encoder unchanged;
   keep reactor-side body-hand generation, cross-attention, and stream-coordination unchanged.

2. Actor condition: Global only
   Modification:
   replace the original `Global + Multi-Part` actor tokenization with a single `Global` actor token;
   keep the original actor encoder unchanged;
   keep reactor-side body-hand generation, cross-attention, and stream-coordination unchanged.

3. Reactor generation: Single-stream Transformer
   Modification:
   replace the original reactor-side `Body + Hand` dual-stream generation with a single holistic reactor stream;
   remove stream coordination (`ParCoCoord`);
   use a standard transformer block with self-attention, cross-attention, and FFN.


### Group2: Refinement

1. Condition encoder: w/o Contact-aware
   Modification:
   remove contact-aware semantic and coarse contact-state conditioning from the condition encoder;
   keep the local residual refiner backbone unchanged.


2. Condition encoder: w/o Geometry-aware
   Modification:
   remove local geometry-aware cues from the condition encoder, including relative position, relative distance, and distance velocity;
   keep contact-aware conditioning and the local residual refiner backbone unchanged.


3. Condition encoder: w/o Interaction-aware
   Modification:
   remove the hand-target interaction modeling branch inside the condition encoder;
   keep basic contact-aware and geometry-aware conditioning, as well as the local residual refiner backbone, unchanged.


### Ablation Study


main table:








exp:

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
      <td>-</td><td>0.987 ± 0.00</td><td>21.693 ± 0.2708</td><td>12.305 ± 0.0335</td>
      <td>-</td><td>0.726 ± 0.0002</td><td>19.919 ± 0.1911</td><td>13.809 ± 0.0565</td>
    </tr>
    <tr>
      <td>actor_bodyhand</td>
      <td>0.997 ± 0.0159</td><td><b>0.972 ± 0.0001</b></td><td><u>21.231 ± 0.1888</u></td><td><b>12.356 ± 0.0324</b></td>
      <td><u>5.001 ± 0.103</u></td><td><b>0.607 ± 0.0001</b></td><td><u>18.749 ± 0.3194</u></td><td>13.633 ± 0.0536</td>
    </tr>
    <tr>
      <td>actor_global</td>
      <td><b>0.935 ± 0.0104</b></td><td><b>0.972 ± 0.0001</b></td><td><b>21.269 ± 0.1685</b></td><td><u>12.374 ± 0.0353</u></td>
      <td>5.064 ± 0.1188</td><td>0.598 ± 0.0003</td><td><u>18.749 ± 0.2753</u></td><td><u>13.647 ± 0.0321</u></td>
    </tr>
    <tr>
      <td>HiReact*</td>
      <td><u>0.941 ± 0.0131</u></td><td>0.97 ± 0.0001</td><td>21.222 ± 0.1602</td><td>12.375 ± 0.0392</td>
      <td><b>4.007 ± 0.0829</b></td><td><u>0.601 ± 0.0002</u></td><td><b>18.838 ± 0.2493</b></td><td><b>13.812 ± 0.0464</b></td>
    </tr>
  </tbody>
</table>



