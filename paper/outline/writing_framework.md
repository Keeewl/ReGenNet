# Writing Framework

## Method Name

- Full name: `HiReact`
- Expansion: `Hierarchical Interaction-aware Reaction Generation`

## Paper Title

- Current title:
  `HiReact: Fine-Grained Human Reaction Generation via Multi-Scale Interaction Modeling and Contact-Aware Two-Stage Refinement`
  `HiReact: Human Reaction Generation via Multi-Scale Modeling and Multi-Stage Refinement`

## Core Positioning

- `HiReact` is a hierarchical two-stage human reaction generation framework.
- Stage1 generates an initial, globally reasonable, interaction-consistent reaction motion.
- Stage2 performs contact-aware refinement on top of this initial reaction motion.
- The role of Stage2 is not to rescue an unreliable Stage1 output.
- The role of Stage2 is to further improve fine-grained local contact realism that is difficult to model directly in full-sequence generation.

## Writing Principle

- Do not describe Stage1 as a weak or rough generator.
- Emphasize that Stage1 provides a reliable initial reaction for subsequent refinement.
- Present Stage2 as a targeted fine-grained refinement stage for local hand-target contact details.
- The overall story should be:
  `initial reaction generation first, then contact-aware local refinement`

## Current One-Sentence Summary

`HiReact is a hierarchical two-stage framework that first generates an interaction-consistent initial human reaction motion with multi-scale interaction modeling, and then enhances fine-grained contact realism through contact-aware local refinement.`
