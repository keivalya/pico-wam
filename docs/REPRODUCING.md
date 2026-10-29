# Full experiments: dependency order and resource scope

Run commands from the repository root. Use new output directories. Freeze
code, dependencies and configurations before preparing a study: source or data
changes invalidate its integrity checks. Configurations are the authoritative
record of sample sizes, seeds, losses, budgets and start banks. These are local
timestamped protocols, not independently registered public preregistrations.

## 1. Data and main 15-seed study

Install `huggingface_hub`, `pyarrow` and `av` if you have not installed the root requirements.txt (which includes them).
The downloader pins both LeRobot repositories by full revision in
`scripts/download_pusht.py`; it needs network access and several GB of disk space.

```sh
mkdir -p data/reproduction
python -m scripts.download_pusht --output-path data/reproduction/demos.npz
python -m scripts.branch_demos data/reproduction/demos.npz --output-path data/reproduction/branches.npz --every 4 --length 8 --noise 0.3 --seed 271828
python -m scripts.evidence_study prepare --config experiments/ral_evidence_v2.json --demos data/reproduction/demos.npz --branches data/reproduction/branches.npz --study results/ral-evidence-v2
```

For each seed 10 through 24 and condition `bc`, `current_demo`, `current_branch`,
`wam_demo`, `wam_branch`, run the following with the corresponding values:

```sh
python -m scripts.evidence_study train --study results/ral-evidence-v2 --seed 10 --condition wam_branch --device cuda
python -m scripts.evidence_study evaluate --study results/ral-evidence-v2 --seed 10 --condition wam_branch --stage dev --device cuda
```

After all 75 training/development combinations finish, freeze once. Then evaluate
all 75 combinations on the final bank and build the integrity-checked report:

```sh
python -m scripts.evidence_study freeze --study results/ral-evidence-v2
python -m scripts.evidence_study evaluate --study results/ral-evidence-v2 --seed 10 --condition wam_branch --stage test --device cuda
python -m scripts.evidence_study report --study results/ral-evidence-v2
```

`--device cpu` is available but slower. The five-seed `ral_evidence.json` is historical
development; it is not the final manuscript study. Do not use final results to tune.
Training uses 30,000 Adam updates at batch 64 per model. Main-study work including
registrations/checks used 21.3 V100 GPU-hours; model-only times were 195 s (BC) and
465 s (WAM) per 30,000 updates. Startup, development and simulation add overhead.

## 2. Follow-ups on frozen models

Install `requirements/paper-extras.txt` for Diffusion Policy and the door workflow.
Use a separate virtual environment with `requirements/original-physics.txt` for
gym-pusht 0.1.8: its Pymunk <7 constraint conflicts with the main Pymunk 7 environment.
Activate the appropriate environment before running the corresponding module.

| Manuscript evidence | Implementation / configuration | Required source |
|---|---|---|
| Horizon-matched study | `scripts.evidence_study` / `ral_evidence_f0.json` | Same converted data |
| Common-bank ranking | `scripts.candidate_audit` / `f1_candidate_ranking.json` | Main frozen study |
| Tuned BC | `followup.study` / `s3a_history_bc.json`; tuning via `followup.baselines` | Main frozen study |
| Diffusion Policy | `followup.study` / `s3b_diffusion_policy.json` | Main frozen study; diffusers, einops |
| Dynamics alternatives | `followup.study` / `s4_planners.json` | Main frozen study |
| Same-start imitation contrasts | `followup.s4_imitation` | Completed dynamics and imitation studies |
| Architecture variants | `followup.architecture` / `architecture_v1.json` | Main study/data |
| Original physics check | `followup.original_pusht` / `s5a_original_pusht.json` | Frozen models; gym-pusht 0.1.8 |
| Fresh prediction | `followup.fresh_prediction` / `s5b_fresh_prediction.json` | Main frozen study |
| Proposal/scoring factorial | `followup.tmlr_factorial` / `tmlr_factorial_v1.json` | Main frozen branch checkpoints |

Configurations reside in `experiments/`. The table gives command entry points,
which can differ from implementation modules. `followup.study` loads the baseline
and dynamics implementations; `followup.planners` itself has no command interface.
Run `python -m MODULE --help` for the entry point shown in the table.

For `followup.study` and `followup.architecture`, use `prepare`, training for each
trainable method/seed, development evaluations, `freeze`, final evaluations and
`report`. Architecture requires its own entry point to register its methods.
For tuned BC, after preparation run `python -m followup.baselines tune --study S
--index I --device cuda` for all 16 grid indices (0–15), then `select --study S`
before `followup.study train`. Diffusion Policy requires no tuning stage.

Candidate audit uses `prepare --audit S --config C`, `score --audit S --seed N
--condition NAME`, and `report --audit S`; it does not use `--study` or a new
training/freeze stage. Fresh prediction uses `prepare`, `score` and `report` with
`--study`. Original-physics and same-start-imitation studies use `prepare`,
`evaluate` and `report` with `--study`; the latter takes `--index` rather than
`--seed`. They evaluate already frozen checkpoints and do not retrain or refreeze.
Preserve paths expected by dependent configurations or edit the configuration
before preparation. Pair comparisons on the configuration's same seeds and banks.
The same-start imitation pairing is exploratory, not a prospectively chosen test.

Example factorial commands, after the main study is frozen:

```sh
python -m followup.tmlr_factorial prepare --config experiments/tmlr_factorial_v1.json --study results/tmlr-factorial-v1
python -m followup.tmlr_factorial evaluate --study results/tmlr-factorial-v1 --seed 10 --evaluation LL --stage dev --device cpu
python -m followup.tmlr_factorial common --study results/tmlr-factorial-v1 --seed 10
```

Repeat development for seeds 10–24 and LL/LE/EL/EE, and `common` for all seeds.
Then run `freeze`, all final `evaluate ... --stage test` calls, and `final-report`.
Exact simulation is expensive: the dynamics study used 1,390 allocated CPU-hours,
including reruns. This is not an estimate for the miniature smoke check. Horizon
training used 16.0 GPU-hours, architecture tests 16.9, Diffusion Policy 44.7.
These are study-specific measurements, not a complete project compute total.

## 3. Independent door mechanism study

Use an additional environment with MuJoCo 3.6.0 and official Meta-World source commit
`cdafedfaaa896472fb7e76e2fe103e120b67a9b6` (tag v3.0.0; package metadata says 2.0.0):

```sh
python -m pip install mujoco==3.6.0 'metaworld @ git+https://github.com/Farama-Foundation/Metaworld.git@cdafedfaaa896472fb7e76e2fe103e120b67a9b6'
python -m followup.tmlr_contact prepare --config experiments/tmlr_contact_v1.json --study results/tmlr-contact-v1
python -m followup.tmlr_contact collect --study results/tmlr-contact-v1
python -m followup.tmlr_contact train --study results/tmlr-contact-v1 --seed 40
```

Training selects CUDA when available, otherwise CPU. Train all seeds 40–49,
then `freeze`, `bank`, `evaluate --seed ...` for all ten seeds,
and `report`, always with the same `--study`. Preparation checks arbitrary-state
continuation fidelity. The study has independent state-space models; it does not
transfer the shared pico-WAM architecture. Its configuration fixes 80 expert and
80 noisy-expert episodes, 10,000 updates, 20 bank episodes and 50 final starts.
The miniature checks do not retrain the full numerical experiments. See VALIDATION.md for the release verification scope.
