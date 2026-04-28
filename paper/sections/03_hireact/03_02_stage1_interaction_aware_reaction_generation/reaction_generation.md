## 3.2 Stage1: Reaction Generation

Stage1 aims to generate an initial reaction sequence $\tilde{X}$ from the actor motion $Y$. We adopt a diffusion-based formulation for conditional reaction generation. Specifically, given the clean reactor motion $X_0$, the forward noising process gradually perturbs it as
\[
q(X_{\tau}\mid X_{\tau-1})=\mathcal{N}\!\left(X_{\tau};\sqrt{1-\beta_{\tau}}\,X_{\tau-1},\beta_{\tau}\mathbf{I}\right),
\]
where $\tau$ denotes the diffusion timestep and $\{\beta_{\tau}\}$ is a predefined variance schedule. Stage1 learns the reverse denoising process conditioned on the actor motion and produces the initial reaction sequence as
\[
\tilde{X}=G(Y).
\]
To support both global interaction consistency and fine-grained local response modeling, HiReact performs structured decomposition on both the conditioning side and the generation side.

On the actor side, we model the input motion as a multi-scale interaction cue provider. Besides the global actor representation $Y_g$, we further decompose the actor motion into $P$ local body-part representations $\{Y_{p_i}\}_{i=1}^{P}$. The global representation captures the overall interaction context, while the local part representations provide body-part-level cues about where and how the interaction is initiated. In our implementation, $P=6$, covering torso-head, lower body, left arm, right arm, left hand, and right hand. This design enables Stage1 to condition the reaction generation on interaction cues at multiple spatial scales.

On the reactor side, we explicitly decompose the target motion into body and hand components, denoted by $X_b$ and $X_h$, and generate their initial predictions $\tilde{X}_b$ and $\tilde{X}_h$ with separate streams. This design preserves hand-centric local responses that are easily smoothed out in holistic full-body generation, while maintaining coherent whole-body reaction dynamics. The structured generation process can be summarized as
\[
(\tilde{X}_b,\tilde{X}_h)=G\big(Y_g,\{Y_{p_i}\}_{i=1}^{P}\big).
\]
The final initial reaction sequence $\tilde{X}$ is obtained by merging the predicted body and hand motions back into the full reactor representation.

To jointly model the dual reactor streams and the multi-scale actor conditions, Stage1 uses an interaction-aware conditional denoising backbone. The body and hand streams first model their own temporal dynamics with stream-specific self-attention. The encoded actor motion is then injected into both streams through actor-conditioned cross-attention, enabling each stream to respond to the global and local interaction cues. We further introduce stream coordination between the body and hand streams to exchange global context during denoising, which improves the consistency between coarse whole-body motion and fine-grained hand responses. For online generation, causal masks are applied to the actor encoder and the denoising backbone to preserve temporal causality. When action text is available, its embedding is added to the diffusion condition together with the timestep embedding.

Stage1 is trained with a diffusion denoising objective together with interaction-aware relation constraints. The denoising loss supervises the reverse diffusion process, while the relation losses regularize the generated reaction with respect to the actor motion in terms of orientation, translation, and relative body-motion consistency. The overall Stage1 objective is written as
\[
\mathcal{L}_{\text{stage1}}
=
\mathcal{L}_{\text{diff}}
\lambda_{\text{ori}}\mathcal{L}_{\text{ori}}
\lambda_{\text{trans}}\mathcal{L}_{\text{trans}}
\lambda_{\text{rel}}\mathcal{L}_{\text{rel}},
\]
where the additional relation terms stabilize interaction semantics and improve the global quality of the generated initial reaction.
