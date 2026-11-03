# pico-WAM: research summary and citation guidance

**Paper:** pico-WAM: Evaluating Counterfactual Prediction and Planning Error in a Compact World Action Model
**Author:** Keivalya Pandya
**Affiliation:** Northeastern University
**Code:** https://github.com/keivalya/pico-wam

**Preprint:** [arXiv:2611.01039](https://arxiv.org/abs/2611.01039), submitted October 31, 2026.
**Full text:** [PDF](https://arxiv.org/pdf/2611.01039).
This is an arXiv preprint, not a claim of journal acceptance.

## What is pico-WAM?

pico-WAM is a compact World Action Model: a shared representation supports a learned
policy, action-conditioned dynamics, and a physical-state readout. An external
planner ranks proposed action sequences. The controlled Push-T experiments supply
physical state at inference and use a constant instruction. They study interactions
between data, predictions, and planning rather than visual-only or language-based control.

## What findings can this work support?

| Research question | Evidence and its scope |
|---|---|
| Does counterfactual training data change search performance? | Across 15 Push-T training seeds, branch data enlarges restart's coverage advantage over warm search by 4.8 percentage points (paired 95% interval 1.5–8.0). Restart wins in both data regimes. |
| Can scoring be improved while retaining learned proposals? | Simulator scoring improves coverage by 12.2 points (9.4–15.0) in the crossed intervention. Substituting simulator states in proposal generation alone shows no detected gain. These are interventions in the tested planner, not optimal-action oracles. |
| Does lower ranking regret establish better manipulation? | No. The surrogate cost includes agent approach; the saved worked example lowers cost without moving the block. Ranking and task completion are distinct measurements. |
| Does the data/ranking finding extend to another task? | A separate door-opening study improves regret and planner success with noisy-expert data. Mixed-data planning succeeds in 88.2% of trials versus 78.6% for demonstration-only planning and 98.6% for BC. This uses a different architecture and fixed data/banks. |

The study does not establish a new state-of-the-art controller, broad architecture
transfer, or general superiority over imitation. Reactive/planner comparisons differ
in observation frequency; the primary search and scoring contrasts hold it fixed.
Confidence intervals describe the specified run-level variation, not uncertainty
over all datasets or tasks.

## Inspect or reproduce the evidence

- [Main results and architecture](../README.md#what-the-current-study-finds)
- [Full reproduction instructions](REPRODUCING.md)
- [Validation scope](VALIDATION.md)
- [Archived numerical reports](../reproduction/analysis/)

The repository supports figure and numerical regeneration, a saved-decision replay,
and a miniature experimental workflow. These checks do not establish independent
full retraining. Original pretrained checkpoints are not included.

## Cite the paper or implementation

For research findings, cite [the paper](https://arxiv.org/abs/2611.01039).
[Download BibTeX](../CITATION.bib) for the paper and the separate software record.
The [machine-readable citation file](../CITATION.cff) makes the paper the preferred
citation while retaining software version metadata.
