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






