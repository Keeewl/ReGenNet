## Problem Definition

1. Existing methods can often generate globally plausible reaction motions, but they remain inadequate for modeling fine-grained local responses of the reactor, especially hand-centric reactions.
2. High-quality reaction generation depends not only on the actor's global motion pattern, but also on understanding the localized interaction cues conveyed by different actor body parts.
3. Even when the overall reaction is reasonable, local hand-target contact and interaction geometry are still prone to inaccuracies.
4. Therefore, fine-grained reaction generation requires both localized interaction-intent modeling and accurate local contact modeling, which are difficult to achieve simultaneously within a single holistic generator.

## Corresponding Solution

To address this, we adopt a hierarchical two-stage design that decouples globally plausible reaction generation from fine-grained contact refinement.

1. In Stage 1, we construct Global + Multi-Part conditions for the actor to provide multi-scale interaction cues.
2. In Stage 1, we perform body-hand dual reaction generation for the reactor to preserve finer-grained local response details.
3. In Stage 2, we further refine the initial reaction motion through local contact-aware refinement to improve hand-centric interaction accuracy.

## Core Contributions

1. We present HiReact, a hierarchical interaction-aware framework for fine-grained human reaction generation.
2. We propose a multi-scale Stage 1 generator that models actor interaction intent and reactor response in an asymmetric manner.
3. We propose a hand-centric, contact-aware Stage 2 refinement module to improve local interaction accuracy beyond globally plausible reaction generation.
