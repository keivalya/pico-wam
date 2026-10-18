# Building and checking the manuscript

The authoritative manuscript is `pico-wam-paper.tex`. It reports both environments: the point-mass navigation study and the Push-T study. The tutorial, `docs/pico-wam-paper.md`, covers Push-T only and uses the same stored results. Neither is formatted for a conference submission.

From the repository root:

```sh
python docs/figures/make_figures.py      # point-mass figures, from docs/results/<experiment>/
python docs/figures/pusht_figures.py     # Push-T figures, from docs/results/pusht/
python docs/figures/pusht_numbers.py     # every Push-T number in the paper -> docs/paper/pusht-numbers.tex
# the illustration data needs a trained run (wam.pt and branched.npz of one seed):
python docs/figures/export_illustrations.py --run-dir results/pusht/seed0 --out docs/results/pusht/final/figure-data
tectonic --keep-logs docs/paper/pico-wam-paper.tex
```

The paper reads every Push-T number from `pusht-numbers.tex` as a LaTeX macro, for example `\PTbasePlan`, so no Push-T result is typed by hand. Rerun `pusht_numbers.py` after changing anything in `docs/results/pusht/`. Each figure script audits its own output, checking widths, fonts, labels, text size and overlapping text, and prints `figure audit ...: clean` or the problems it found.

## Validation

Point-mass study:

- Selected archived values were recomputed directly from per-experiment summaries; spreads in those files use population SD, not sample SD or confidence intervals.
- The affine/simulator audit was run over five collection seeds, with complete metrics and source hashes.
- The supervision pilot was checked for paired test IDs, equal update counts, complete conditions and local source hashes (Explorer array 11088783, all exit codes 0:0).
- The batched policy evaluator was checked against sequential evaluation on four identical initial-state seeds, and the current-state auxiliary objective was checked to send gradients into the visual encoder and none into the dynamics module.

Push-T study:

- The converted LeRobot data were checked against the simulator (`scripts/pusht_data_checks.py`, stored in `docs/results/pusht/data_checks.json`): replayed block positions, rendered frames against the dataset's frames, the demonstrations' coverage, and the scripted expert's success rate.
- `simulate()` reproduces `step()` exactly from any state, so every counterfactual prediction and plan check is compared with the true outcome.
- Every results folder in `docs/results/pusht/` is mapped to its experiment branch, commit and run in `docs/results/pusht/README.md`. The planner-budget run hit its time limit after its three seeds were tested; its numbers come from the per-seed files, like every other folder.
- Planner settings were chosen on development starts (env seeds 90000 and above), never on test starts. The action-head sizes of the capacity round were compared on the test starts, and the paper says so.
- All experiment branches are merged into `main`. Every option they added is a setting of `scripts/run.sh` whose default reproduces the final configuration, and a one-epoch run with every stage, including the planner-budget sweep, completes end to end.

The pilots use the same full model initialization for each condition, leaving unused modules untouched in the cloning and current-state conditions. Only their policy outputs are meaningful; those unused dynamics checkpoints must not be reported as trained world models. CUDA kernels are not forced into deterministic algorithms: seeds control initialization and minibatch order, not bitwise reproducibility across devices.

The final PDF was compiled without undefined references or overfull boxes, and its pages were rendered and inspected.
