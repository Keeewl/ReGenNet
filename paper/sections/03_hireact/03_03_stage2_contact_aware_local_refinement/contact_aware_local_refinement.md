## 3.3 Stage2: Contact-Aware Local Refinement

Stage2 refines the Stage1 initial reaction by focusing on the most interaction-critical parts of the sequence. Given the initial reaction $\tilde{X}$, our goal is to improve local hand-target contact accuracy and interaction geometry without re-generating the full sequence. We therefore formulate Stage2 as a local residual refinement problem over contact-relevant temporal windows. For the $m$-th window, the refiner predicts a residual motion $\Delta X^{(m)}$ and updates the initial reaction as
\[
\hat{X}^{(m)}=\tilde{X}^{(m)}+\Delta X^{(m)},
\]
where $\tilde{X}^{(m)}$ and $\hat{X}^{(m)}$ denote the initial and refined reactor motions within that local window.

To localize the refinement problem, we first construct contact-relevant windows from the Stage1 initial reaction. We derive a coarse contact state from the predicted hand-to-region distances, extract hand-time contact segments, and convert them into fixed-length local windows. Each window is associated with a selected hand $s^{(m)}$, a primary target region $q^{(m)}$, and a top-$K$ target-region set $\mathcal{Q}^{(m)}$. This deterministic proposal mechanism reduces Stage2 from full-sequence re-generation to hand-centric local refinement and provides a unified window-level input format for the subsequent refiner.

Each local window is then equipped with contact-aware conditioning. Specifically, we use three types of cues: hand-region semantic cues, coarse contact state, and local geometry cues. The semantic cues describe which hand and which target regions are involved in the current window. The coarse contact state is represented by a contact mask $M^{(m)}$ and a distance map $D^{(m)}$, which summarize the current Stage1 contact status over regions and frames. The local geometry cues include relative position $\Delta^{(m)}$, relative distance $d^{(m)}$, and distance velocity $v^{(m)}$, which characterize how the selected hand evolves with respect to the target regions over time. These cues are encoded into a unified contact-aware conditioning $Z^{(m)}$.

Based on the local motion inputs and the encoded conditions, Stage2 uses a window-level temporal residual transformer to predict the residual motion. The initial reaction window $\tilde{X}^{(m)}$ serves as the main input sequence, while the actor motion window $Y^{(m)}$ provides the local interaction context. The refiner performs temporal self-attention over the local reactor motion, cross-attention to the actor motion, and condition modulation with $Z^{(m)}$, followed by feed-forward updates. The overall residual prediction can be written as
\[
\Delta X^{(m)} = R\big(Y^{(m)}, \tilde{X}^{(m)}, Z^{(m)}\big).
\]
This design allows Stage2 to preserve the stable global motion structure from Stage1 while selectively correcting the local contact-related errors.

After all selected windows are refined, their outputs are aggregated back into the full sequence through overlap-aware weighted stitching, yielding the final refined reaction $\hat{X}$. Stage2 is trained with a residual reconstruction objective together with contact-aware regularization terms. The overall Stage2 objective is formulated as
\[
\mathcal{L}_{\text{stage2}}
=
\mathcal{L}_{\text{res}}
\lambda_{\text{contact}}\mathcal{L}_{\text{contact}}
\lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}}
\lambda_{\text{geom}}\mathcal{L}_{\text{geom}},
\]
where $\mathcal{L}_{\text{res}}$ supervises the predicted residual motion, $\mathcal{L}_{\text{contact}}$ emphasizes contact-relevant refinement, $\mathcal{L}_{\text{smooth}}$ enforces temporal continuity across local windows, and $\mathcal{L}_{\text{geom}}$ denotes optional geometry-aware regularization for improving contact fidelity.
