# Experiment record template

Copy this file for each later milestone and link machine-readable records and
trajectories. Declare the protocol before inspecting confirmatory results.

## Question and protocol

- Milestone, date, and responsible implementation revision:
- Hypothesis and exact mathematical/architectural variant:
- Dependencies and evidence that their gates passed:
- Primary outcome and practical success threshold:
- Comparator policies and identical components:
- Hyperparameter trials, tuning budget, and policy/checkpoint selection rule:
- Fixed episode/interaction/compute budget and stopping rule:
- Pilot training seeds:
- Independent confirmatory training seeds and justification of their count:
- Evaluation episode seeds, held-out layouts, contexts, and trajectory partition:
- Assistance, exploration bonuses, privileged access, or curricula:

## Interfaces and lifetimes

- Exact environment configuration, actions, masks, rewards, and discount:
- Cue-to-decision delay and decision-to-reward delay:
- Policy/critic/executive inputs and causal observation window:
- Frozen encoder checkpoint, layer, preprocessing, schema, and version:
- Feature-cache shapes, checksums, storage, and extraction latency:
- Actor/executive recurrent-state shapes and initialization:
- Termination, truncation, bootstrap masks, and state reset rules:
- Prototype, episodic-memory, replay, and representation-version lifetimes:
- Environment steps, retrieval steps, and parameter-update schedules:
- Rollout length, BPTT window, burn-in, and return horizon:

## Evidence and decision

- Raw trajectory and learning-curve artifact paths and checksums:
- All environment interactions, including diagnostics and unsuccessful runs:
- Parameter counts, latency, wall time, peak memory, and hardware:
- Numerical diagnostics, failures, and any protocol changes:
- Per-training-seed results; episode-level estimates reported separately:
- Uncertainty method and effect size against comparators:
- Continue, revise, or stop; evidence supporting that decision:
- Remaining limitations and the next justified experiment:
