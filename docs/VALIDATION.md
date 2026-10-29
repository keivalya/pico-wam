# Verification of the standalone release

Software version: 0.1.0. Checked 2026-10-29 with Python 3.11.9 on macOS arm64.

## Completed checks

- Fresh main virtual environment installed from the pinned CPU dependencies; optional
  imitation and door dependencies installed with a successful dependency compatibility check.
- Separate original-physics virtual environment installed successfully with
  gym-pusht 0.1.8 and Pymunk 6.11.1. The main environment uses Pymunk 7.3.1.
- `python -m examples.quickstart`: collection, two-epoch training, prediction
  evaluation and short reactive execution passed. The tiny example was not a
  successful benchmark policy and is not presented as paper evidence.
- `python reproduce.py check`: source verification, all eight figure generators,
  exact agreement of ten regenerated numerical outputs, saved-decision replay,
  retrospective sensitivity and miniature factorial/corruption workflow passed.
- Main environment: 18 protocol, candidate, model, architecture, imitation and
  factorial tests passed. Original-physics environment: two integration tests passed.
- Real MuJoCo/Meta-World miniature door workflow passed data collection, training,
  checkpoint freezing, common-bank evaluation, report generation and corruption rejection.
- A clean staged export, containing no generated outputs or tutorial weights,
  passed `python reproduce.py check` (10.3 s with warm caches).
- Code/config/report private-path and credential-pattern scan passed. Numerical
  reports and their scientific interpretation retain the source study's scope.

Exact tested packages are in `requirements/tested-macos-arm64.txt` and
`requirements/tested-original-macos-arm64.txt`. Original research environment
snapshots remain under `reproduction/environments/`.

## Commands

```sh
python reproduce.py check
python -m unittest tests.test_evidence_study tests.test_candidate_audit tests.test_followup tests.test_baselines tests.test_architecture tests.test_s4_imitation tests.test_tmlr_factorial -q
python -m tests.tmlr_contact_smoke
```

In the separate original-physics environment:

```sh
python -m unittest tests.test_original_pusht -q
```

## Scope

These are internal software and migration checks. They do not constitute external
peer review, a full retraining of the paper, full independent numerical reproduction,
a fresh download/conversion of the entire human dataset, or Linux/CUDA validation.
The original pretrained checkpoints and original datasets are not distributed here.
The clean release is derived from the source revision recorded in
`SOURCE_PROVENANCE.json`; its module layout and new study registrations have their
own hashes. Archived historical freeze hashes cannot be transplanted to changed source.
