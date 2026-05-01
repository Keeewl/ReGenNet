# Writing Framework

## Method Name

- Full name: `HiReact`
- Expansion: `Hierarchical Interaction-aware Reaction Generation`

## Paper Title

- Current title:
  `HiReact: Human Reaction Generation via Multi-Scale Modeling and Multi-Stage Refinement`

## Core Positioning

- `HiReact` is a two-stage human reaction generation framework built on multi-scale modeling and multi-stage refinement.
- Stage1 performs multi-scale interaction modeling to generate an initial, globally plausible, interaction-consistent reaction motion.
- Stage2 performs contact-aware local refinement on top of this initial reaction motion.
- The role of Stage2 is to further improve fine-grained local contact realism that is difficult to model directly in full-sequence generation.

## Writing Principle

- Do not describe Stage1 as a weak or rough generator.
- Emphasize that Stage1 provides a reliable initial reaction for subsequent refinement.
- Present Stage2 as a targeted fine-grained refinement stage for local hand-target contact details.
- The overall story should be:
  `multi-scale initial reaction generation first, then multi-stage local contact refinement`

## Current One-Sentence Summary

`HiReact is a human reaction generation framework that first generates an interaction-consistent initial reaction with multi-scale modeling, and then enhances fine-grained contact realism through multi-stage local refinement.`
