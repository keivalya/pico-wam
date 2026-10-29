# Follow one WAM decision

A policy answers “what action should I imitate?” Dynamics answers “what happens if
I take an action?” A state readout exposes that answer in physical coordinates.
An external planner compares several such answers and chooses one action.

## 1. Encode the observation

`PicoWAMPolicy.encode_obs(image, text_ids, state)` combines image keypoints,
instruction features and standardized state into a latent of shape `(batch, 128)`
in the paper model. The instruction GRU processes words; it is not a temporal
memory over camera frames. Push-T's instruction is constant, and its full eight-D
physical state is available at deployment.

## 2. Propose an action

The MLP policy head predicts eight two-dimensional actions. Reactive execution uses
the first four before observing again. The model also supports diffusion/flow
action heads for exploration. The separate Diffusion Policy baseline implements
a full external recipe; selecting the model's small diffusion head is not that baseline.

## 3. Imagine consequences

The dynamics maps `(z_t, a_t)` to `z_next` with a residual MLP and LayerNorm.
Unrolling supplied actions produces `(batch, horizon, 128)` future latents. A
linear readout converts each to standardized state; `denormalize_state` returns
physical coordinates. Optional image decoding is useful for visualization but
is disabled in the controlled study.

The model does not see future observations while imagining a candidate plan.
Future observations are only training targets or simulator checks.

## 4. Understand the training signals

The paper loss combines action MSE, current-state MSE, future-state MSE and latent
consistency. Each term averages over its batch and output dimensions; future losses
also average over the rollout horizon. State standard deviations are floored at
1e-3. Future latent targets are stop-gradient encodings, so prediction gradients
flow through imagined latents and their initial observation encoder.

Only demonstration actions provide imitation targets. Branches are perturbed
simulator continuations with known outcomes, supplying prediction supervision.
Their source episode determines their split, preventing a branch from crossing
from training to a held-out source episode.

Tutorial `scripts.train` is convenient for learning the API. The paper's
`scripts.evidence_study` additionally matches action minibatches, initial parameters,
updates and auxiliary sample counts across five conditions. Those are distinct
training interfaces and their numerical results should not be conflated.

## 5. Choose a plan

The learned planner scores 256 candidate action sequences over 12 imagined steps.
The geometric objective averages corresponding block-corner distances to the target
and adds 0.05 times the agent's distance to the block center beyond its 15-px radius.
Later block distances receive more weight. The planner executes one action and
observes again. It does not call the simulator for candidate selection unless an
explicit exact-simulator diagnostic is selected.

Warm search shifts the previous plan; restart uses a new policy proposal. Both
win/loss comparisons use the same frozen checkpoint. Broadening dynamics data
enlarges the measured restart advantage; restart wins with either dataset.

## 6. Check a prediction against reality

Run `python reproduce.py trace`. It replays the previously selected Fig. 1 example
and compares stored predicted states to simulator outcomes. The branch-trained
model has a lower selected-sequence regret, but neither selected sequence moves
the block over that short horizon: lower true cost reflects agent approach.

This is not proof of episode failure. Approaching the block may be a useful first
step. It demonstrates why short-horizon surrogate regret, prediction error and
closed-loop task success are different measurements.

Run `python reproduce.py sensitivity` to inspect retrospective influence checks on
the primary estimates. These reuse the original training runs, rather than provide
new datasets, new models or independent confirmation.

## 7. Run and inspect a small model

`python -m examples.quickstart` writes a small point-mass dataset, `wam.pt`, training
history, factual/counterfactual prediction diagnostics and a short control report.
The example uses two training epochs and no decoder. It exercises the learning and
evaluation paths; it is not expected to solve the task well.

For a visual decoder experiment, train with `--predict_image` and inspect
`scripts.visualize_rollout`. State renderings and neural image decodings are different
outputs; label them accordingly in any figures you share.

## Questions to explore

- How does error change with rollout horizon or unfamiliar actions?
- Does lower candidate cost come from approaching the object or moving it toward the goal?
- Does a result survive an equal-frequency policy comparison?
- Does changing the dynamics training data alter the warm/restart gap?

Register a new output directory and configuration for changes; do not tune using
the paper's final banks or overwrite archived results.
