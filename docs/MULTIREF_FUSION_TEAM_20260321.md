# Multi-Reference Fusion Iteration Protocol

Updated: 2026-03-21

## Purpose

This document defines how each rollout iteration is executed once the team charter is accepted. It exists alongside `TEAM_MULTIREF_FUSION_ROLLOUT.md` and focuses on operating rules, handoff shape, and recovery state rather than role definitions.

## Iteration Scope

- target branch: `multiref-fusion-rollout`
- active mode: `parallel`
- local interpreter: `.venv/bin/python`
- active change: `rollout-multiref-fusion-indextts-v2`

## Iteration Completion Rule

An iteration is complete only when all of the following are recorded:

1. design input or unchanged-design confirmation
2. implementation delta with owned files
3. review result with findings or explicit no-finding outcome
4. verification result from `.venv`
5. recovery state update in tracked documents

## Required Three-Party Cycle

Every rollout iteration must include all three roles:

1. Design proposes the interface or behavior change and writes down assumptions.
2. Development implements the bounded slice against the written assumptions.
3. Review inspects regressions, behavioral drift, and test coverage before the slice is considered complete.

No implementation slice is done until design, development, and review all have an explicit handoff recorded in either the workstate document or the active OpenSpec change.

## Handoff Contract

Each workstream handoff must state:

- scope completed
- files touched or expected to be touched
- assumptions used
- blockers found
- next required action
- whether ownership can safely pass to another role

## Recovery State

If the session is compacted or interrupted, recovery must come from repository artifacts rather than memory. The lead keeps these facts current:

- active mode
- current assumptions
- current blockers
- next intended step

These recovery facts must exist in the kickoff document and in the active OpenSpec rollout change before and after each substantial implementation pass.

## Locked Defaults

These defaults are locked in unless later review shows a regression:

- timbre default: `spk_cond_emb + speech_conditioning_latent`
- timbre fallback: `speech_conditioning_latent`
- emotion default: `emotion_tensor_anchor_a`
- emotion fallback: `emotion_tensor_sym`

The rollout must keep timbre and emotion variable handling separated.

## Guardrails

- Do not regress the existing single-reference path.
- Do not promote `ref_mel` or `style+ref_mel` as default supported paths because the short-text instability reproduced on CPU.
- Do not merge timbre and emotion experimentation variables into one uncontrolled user-facing preset.
- Do not rely on hidden context for recovery; write recovery facts into tracked documents.
