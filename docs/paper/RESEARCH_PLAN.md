# pico-WAM: research assessment and execution plan

Reviewed October 14, 2026 and updated October 17, 2026 (America/New_York). This is a research decision document, not a claim of publication readiness.

## Assessment

The existing work is a useful educational implementation, but **not currently competitive for a top-tier research publication**. Rewriting it cannot supply missing novelty. The revised paper is a 10–12-page empirical audit with actual environment renders, archived model outputs, and fresh diagnostics. Its claims deliberately stop short of a new world-model method.

The best possible research direction is **separating the benefit of privileged state supervision from the benefit of transition prediction under partial observability, then measuring whether counterfactual competence improves actual control**. Treat this as a question to falsify, not a result to manufacture.

| Candidate claim | Assessment | Evidence needed |
|---|---|---|
| Small policy + residual latent dynamics + linear state head | Established components; integration/size is not sufficient novelty | A new principle with convincing benefits, or a uniquely useful diagnostic benchmark |
| Expert prediction can hide alternative-action errors | Established problem; direct overlap with AD-WM | A distinct mechanism or protocol that changes conclusions beyond prior results |
| Random exploration improves learned dynamics | Plausible familiar result; archived data confounds coverage with distribution and scale | Fixed state bank, fixed transition budget, common normalization, action-sensitivity tests |
| Imagining futures recovers hidden velocity | Overstated: renderer cannot distinguish opposite velocities; known controls and damping reduce future uncertainty | Matched histories, ambiguous observations, calibrated belief prediction, deployment mechanism |
| Pixel-input WAM improves policy | Interesting preliminary observation, 3 seeds with large variation | BC + current-state auxiliary targets, future-state-only targets, history baseline, equal starts and updates |
| Linear readout reveals emergent physical knowledge | Jointly trained with true-state targets, often true-state inputs | Frozen representation and separately fitted probes; unseen probe split and nuisance controls |
| World model improves planning | Evaluated on Push-T (October 2026): planning near the policy beats the reactive policies, and stronger search exploits model errors | Oracle-dynamics and direct-state planners with the same cost; a learned cost; latency and regret |
| This toy needs a sophisticated dynamics model | Contradicted by new restricted-model audit | Harder dynamics/observation settings and structured baselines |

## What has actually been done

- Reaggregated archived JSON results and inspected the implementation. Original datasets/checkpoints are absent from the initial local checkout; archived numbers are not independent replications.
- Rewrote the manuscript, removed tutorial listings and unsupported mechanistic claims, and preserved unfavorable results.
- Produced genuine simulator frames and panels from archived observation/reconstruction/prediction arrays.
- Ran `scripts/research_audit.py` for five seeds. An affine state+action predictor with a known clipping prior fits the toy transition to numerical precision even from deterministic expert trajectories. Action-free predictors fail under randomized interventions. Opposite-velocity scenes render identically, and sampled `simulate`/`step` checks agree. See `../results/research-audit/README.md` for full protocol and limitations.
- Ran Explorer array **11088783**, training 15 models across five seeds. Mean success was 17.0% for BC, 28.4% with current-state supervision, and 33.6% with additional future-state prediction. Future-minus-current paired differences were +7, +9, +4, -2, +8 percentage points. This is an exploratory pilot with low absolute success, not a convergence or significance claim. Complete per-seed outputs are under `../results/supervision-pilot/`.
- Added `scripts/supervision_ablation.py` and an Explorer Slurm launcher. This pilot compares BC, current-state auxiliary prediction, and current+future state prediction under identical H=5 starts, encoder/policy initialization, expert masks, minibatch order, and update count. It deliberately excludes pixel and latent-consistency losses to isolate the extra future-state term. It is not a reproduction of the archived full WAM.

### Push-T study, October 2026

Stage 4's step beyond the toy was taken with Push-T, trained on LeRobot's 206 human demonstrations; the simulator was rebuilt so that `simulate()` reproduces `step()` exactly. Thirty runs on Explorer, stored in `../results/pusht/`, cover prediction, data coverage, objective and model variants, planning budgets, the supervision pilot, and the changes that make the policies succeed. What they settled:

- Factual accuracy does not imply counterfactual accuracy on Push-T either: trained on human demonstrations, the model predicts the block's response to new pushes worse than assuming it stays put. Simulator branches of the demonstrations fix this. Because branches start from states the humans visited, they separate action coverage from state coverage better than scripted data, though not completely: eight perturbed steps still reach new states.
- Planning with the learned dynamics is the first measured control benefit of the world model. Plan checks against the simulator show that more search, or a warm-started search, exploits model errors.
- The matched pilot again attributes most of the auxiliary gain to current-state supervision; future-state prediction added nothing on Push-T.

A capacity round on top of the 100-epoch model then showed that the action head, one hidden layer of 128 units, was the bottleneck: three layers of 512 raised every policy and nearly closed the gap between the WAM's own policy and mini-VLA, while wider or deeper heads and a wider latent did not help further. Without state input, only the WAM learns to push the T. The head sizes were compared on the test starts.

A dream round then checked dreams against the simulator: with 5-step training windows, dreams stay accurate for about 10 to 15 steps. Without the image decoder, 15-step windows with 16-step branches halve the error at 40 steps; with the decoder they help less and lower planning. Weighting the moving pixels in the image loss sharpens the decoded block and gives the most planning successes so far, and is now the default.

Still open: oracle-dynamics and direct-state planners with the same cost, a history-based baseline, a learned cost, head sizes chosen on validation starts, long training windows that keep the decoder's benefits, and branches that a real robot could collect.

## Nearest prior work and novelty risk

Read these full papers and reproduce the relevant baseline before deciding on a new loss:

- [AD-WM, Qiu et al., 2026](https://arxiv.org/abs/2609.30264): action-recovery regularization and counterfactual MPC are directly relevant. A proposal consisting only of inverse-dynamics supervision or action ranking is at high risk of duplicating this work.
- [MiniWAM, Chen et al., 2026](https://arxiv.org/abs/2610.12194): compact predictive targets and privileged transition supervision.
- [TD-MPC2](https://arxiv.org/abs/2310.16828): strong latent-model/planning reference, not a plug-in apples-to-apples comparator unless its training regime and budgets are aligned.
- [Dreamer](https://arxiv.org/abs/1912.01603): behavior learned through latent imagination; unlike the current reactive policy.
- [Causal Confusion in Imitation Learning](https://arxiv.org/abs/1905.11979): behavior-policy correlations and intervention-based reasoning.
- [Unified World Models](https://roboticsproceedings.org/rss21/p015.html): joint video/action modeling already exists at robotics scale.

The initial assessment uses verified primary abstracts/proceedings and repository evidence. A systematic full-text comparison, including concurrent work after this date, is a prerequisite to asserting priority.

## Experiment program and decision gates

### Stage 1 — Establish the mechanism (approximately one week)

Primary hypothesis H1: adding multistep transition prediction improves closed-loop success beyond current-state supervision under the same observation regime and training budget.

Run the three-condition Explorer pilot first. Then use separate development seeds to choose a fixed training schedule and freeze it before a confirmatory run. Confirmatory comparisons should include:

1. BC with a single frame.
2. BC + current-state reconstruction.
3. BC + current and one-step future-state prediction.
4. BC + current and five-step future-state prediction.
5. Full archived WAM objective, with matched starts.
6. BC with 2–4 frames or a recurrent encoder; then a matched history WAM.

Cross privileged state supervision on/off where the objective permits. A no-privileged-target condition must not silently use state losses. Match encoder capacity, expert labels, selected windows, optimizer updates and test initial states. Report total and active parameter counts and measured GPU time; equal updates do not imply equal compute. Include state-input conditions as a ceiling diagnostic, not evidence for language grounding.

Use at least five independent training/data seeds in the pilot, and choose confirmatory seed count from observed seed-level variability and a predeclared precision target. A reasonable starting confirmatory budget is ten seeds × 200 paired episodes; do not treat 2,000 episodes as 2,000 independent training replications. Use a hierarchical paired bootstrap over training seeds and episodes, report per-seed differences, and distinguish exploratory intervals from confirmatory inference.

**Gate:** if current-state supervision accounts for the improvement, reject the claim that dynamics is necessary. Study the supervision mechanism or move to tasks where temporal inference is required. Do not select the best seed or hide a null result.

### Stage 2 — Separate action coverage from state coverage (one to two weeks)

Hypothesis H2: action diversity reduces counterfactual error after state distribution, transition budget and metric scaling are controlled.

Build a saved state bank and branch each state under a fixed number of actions. Vary branch-action coverage while retaining the same starts and total transitions. Evaluate on a separate frozen state/action bank. Keep a second rollout-based dataset study to measure practical effects of exploration, but do not confuse it with the controlled intervention study.

Baselines: persistence; true zero-action simulator reference; fitted affine state+action with explicit clipping prior; nonlinear state-space MLP; state-only/action-shuffled neural predictor; latent WAM. Fit normalization on a common training reference and report raw position/velocity errors as well as normalized errors. Examine fixed-state action swaps and finite-difference action response. Include constant pushes, braking, action reversal, boundary states, and unseen damping/force parameters.

**Gate:** a novel representation claim must survive direct-state and structured baselines on a task that actually requires learned perception. The fresh affine result means the current fully observed toy is insufficient.

### Stage 3 — Connect prediction to decisions (one to two weeks)

Hypothesis H3: action-selection regret predicts closed-loop performance more reliably than average factual prediction error, across model checkpoints and environment shifts.

Implement one receding-horizon sampling/CEM planner and use it unchanged with oracle dynamics, direct-state predictors and latent dynamics. Fix candidate draws, horizon and model-call budget. Use a cost declared in advance: goal distance plus terminal speed penalty; do not tune the cost per model on test episodes. Measure success, terminal cost, oracle-candidate regret, action ranking near the selected candidates, inference latency and failures near boundaries. Compare a reactive policy and an oracle PD controller as controls.

This hypothesis overlaps AD-WM; novelty requires demonstrating a distinct failure mechanism or a new diagnostic that adds value beyond its reported diagnostics. Reproduce that comparison before proposing an action-ranking loss.

**Gate:** if prediction gains do not improve decisions, publishable value would need a robust explanatory negative result, not a relabeled control claim.

### Stage 4 — Test beyond the toy (two to four weeks, resource dependent)

Select a small, fixed set of contact-rich manipulation tasks from the existing Meta-World integration, such as pushing and pick-and-place, after verifying installed task versions. Include occlusion/history, changes in mass/friction, goal distractors and unseen layouts. Use at least two qualitatively different dynamics/observation regimes. If the paper's claim is about robot deployment, add a real-robot validation with a collaborator; simulation-only evidence must make its transfer argument explicit.

Record data collection cost, task versions, camera calibration, success definitions, assets and licenses. Preserve failed episodes and select qualitative examples by declared rules (fixed seed or representative quantile), not visual appeal. An expensive pretrained VLA comparison is unnecessary unless the claim explicitly concerns that class of model; matched modest baselines are the priority.

**Gate:** top-tier submission requires a clear insight that survives realistic baselines and matters outside the diagnostic toy. More tasks alone do not supply novelty.

## Explorer execution

Host: `login.explorer.northeastern.edu`, user `pandya.kei`.

Isolated directory: a fresh checkout under `/scratch/pandya.kei/`, submitted from its root.

Launcher: `scripts/hpc/supervision_pilot.sbatch`. Five seeds, 200 collected episodes/seed, 15 epochs/condition, 100 paired evaluation episodes/seed, three conditions. GPU-short, one GPU/job, at most two jobs concurrently, 12 GB host RAM, 45-minute wall-time cap/job. This caps the allocation at 3.75 GPU-hours if every task reaches its limit; actual compute is measured in the outputs. No existing jobs or remote projects are modified.

Outputs are unique to the Slurm array ID under `runs/job-<id>/seed-<seed>/`; checkpoints and datasets remain on Explorer. Save per-seed JSON and source hashes locally for the paper. Never rerun `scripts/run_toy.sh` in a directory containing needed results: that existing script deletes its `results` directory.

Before promoting pilot results: verify every task exits successfully; validate per-episode counts and seeds; check same-start/update invariants; inspect loss curves and failures; preserve all outcomes; run the frozen confirmatory protocol on new seeds. Queue submission is not experiment completion.

## Top-tier venues only

“Tier 1” has no universal definition. This shortlist uses established flagship robotics/ML conferences and journals; no workshops or lower-tier fallback venues are proposed. Scope fit is not an acceptance prediction. **None is a sound submission target for the current evidence without further research.**

| Venue | Why it could fit after the gates | Length and timing checked October 14, 2026 | Recommendation |
|---|---|---|---|
| **RSS 2027** | Strong robotics hypothesis, controlled mechanism, meaningful control result | Current official CFP: Stage 1 due **December 4, 2026 AoE**, maximum 6 pages (5 content + 1 references); invited Stage 2 due April 16, 2027, maximum 8 content pages excluding references | Earliest concrete target if a distinct hypothesis and preliminary evidence are ready. The working manuscript must be restructured for Stage 1. |
| **CoRL** | Best topical fit for learned dynamics and representation/control evidence | 2026 guidance permits 8 main-text pages at submission, 9 camera-ready; references/acknowledgments/appendix excluded. It explicitly expects a core robotics problem and robot evidence or convincing simulation transfer. A future-cycle deadline is not verified here. | Preferred mature conference target after realistic experiments; recheck next cycle's CFP. |
| **IEEE Transactions on Robotics (T-RO)** | Substantial, original robotics contribution with deeper validation | Current instructions: first regular submission ≤18 pages; final ≤20; mandatory overlength charges after 12 final pages. A 10–12-page Transactions-format paper is permitted, not guaranteed sufficient. | Best length-compatible journal target, with a substantially stronger robotics study. |
| **ICLR** | Generalizable learning insight about supervision/observability | 2027 main-text limit 9 pages initially, references excluded; submission deadline September 25, 2026 has passed. | Consider a later cycle only if the result generalizes beyond robotics instrumentation. |
| **JMLR** | Broad, mature ML insight with substantial evidence | Author instructions require original work; use its official style and current submission process. No future deadline is asserted. | Long-term option only for a much stronger general learning contribution. |

Primary venue sources: [RSS CFP](https://roboticsconference.org/information/cfp/index.html), [CoRL instructions](https://www.corl.org/contributions/instruction-for-authors), [CoRL scope](https://www.corl.org/contributions/call-for-papers), [T-RO author information](https://www.ieee-ras.org/publications/t-ro/t-ro-information-for-authors/), [ICLR author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines), [ICLR dates](https://iclr.cc/Conferences/2027/CallForPapers), [JMLR author information](https://www.jmlr.org/author-info.html).

The 10–12-page requirement is treated as the entire working PDF, including references. Venue templates have their own accounting and must be followed without shrinking fonts or margins. The current file is a working manuscript, not a venue-compliant anonymous submission.

## Submission decision

Do not submit based on writing quality or additional attractive figures. Proceed only when (1) the closest-paper comparison identifies a real distinction, (2) the key mechanism survives supervision/history controls, (3) gains or negative findings replicate with uncertainty estimates, (4) robot relevance is demonstrated, and (5) artifacts reproduce the claims. Otherwise keep pico-WAM as a useful diagnostic/educational project while revising the research question.
