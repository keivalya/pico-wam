# Push-T results

Every Push-T number in the paper and the tutorial is computed from these files by
`docs/figures/pusht_numbers.py`, and every Push-T figure by
`docs/figures/pusht_figures.py`. Each folder holds one experiment: three training seeds
(`seed0`, `seed1`, `seed2`), each tested on 100 unseen starts, plus a `summary.json`.

Files in a seed folder:

| File | What it holds |
|---|---|
| `wam_history.json`, `vla_history.json` | training and validation losses per epoch |
| `wam_eval.json` | open-loop prediction errors of the world model, factual and counterfactual |
| `vla_test.json`, `wam_test.json` | closed-loop test of the mini-VLA baseline and of the WAM's action head |
| `wam_mpc_test.json` | closed-loop test of the WAM planning with its world model |
| `wam_mpc_warm_test.json` | the same, with the older planner whose search starts from the last plan |
| `wam_mpc_*_test.json` | the planner-budget variants: `wide`, `cem` and `noprior` |

`data_checks.json` checks the converted LeRobot data against the simulator
(`scripts/pusht_data_checks.py`). `baseline/`, `final/` and the 100-epoch folders also hold the linear-model audit
(`dynamics_audit.json`), the data behind the qualitative figures (`figure-data/`), and
the compute costs on a GPU node (`efficiency/`); `baseline/` also has them on one thread
of a laptop CPU (`efficiency-laptop/`).

## Where each folder comes from

Each experiment was run from a git branch that changes one setting of its parent in
`scripts/run.sh`. All runs used `bash scripts/run.sh` on one GPU of the Explorer cluster.

| Folder | Change from its parent | Parent | Branch | Commit | Run |
|---|---|---|---|---|---|
| `baseline` | the study base: 206 LeRobot demos, 15 epochs, one action per step, planning | | `orx/push-t-study-base-planning-extras` | b5de283 | 2eb1c9a5 |
| `horizon-1` | imagination window H = 1 | `baseline` | `orx/push-t-horizon-h-1` | dccb0b8 | d5d3fd6b |
| `horizon-10` | imagination window H = 10 | `baseline` | `orx/push-t-horizon-h-10` | 15bc7ac | 050de994 |
| `no-latent` | no latent-consistency loss | `baseline` | `orx/push-t-no-latent-consistency` | cf5c76c | 51b04102 |
| `no-image` | no image loss | `baseline` | `orx/push-t-no-image-loss` | f63c5ee | 72048917 |
| `pixel-input` | images and language only, no state input | `baseline` | `orx/push-t-pixel-input` | 5833a7a | b62f963d |
| `demos-50` | the first 50 demos | `baseline` | `orx/push-t-50-demos` | 42c9fc1 | f983b4f9 |
| `demos-100` | the first 100 demos | `baseline` | `orx/push-t-100-demos` | 5adec91 | 13c42b6f |
| `expert-p0` | scripted expert data, no random bursts | `baseline` | `orx/push-t-scripted-expert-no-bursts` | 8940efa | 56ca06d0 |
| `expert-p0.1` | scripted expert, random bursts with p = 0.1 | `baseline` | `orx/push-t-scripted-expert-bursts-0-1` | 89e8fd7 | a887713e |
| `expert-p0.3` | scripted expert, random bursts with p = 0.3 | `baseline` | `orx/push-t-scripted-expert-bursts-0-3` | 01a45ad | bf99dfa7 |
| `mixed-p0.3` | human demos plus scripted bursts with p = 0.3 | `baseline` | `orx/push-t-human-scripted-bursts` | c237ba0 | 8a3aa982 |
| `diffusion-head` | diffusion action head | `baseline` | `orx/push-t-diffusion-head` | 687eae3 | c4adafcd |
| `wider-model` | latent width 256 | `baseline` | `orx/push-t-wider-model-d-model-256` | 2bb71a8 | 5b8e1274 |
| `epochs-100` | 100 epochs | `baseline` | `orx/push-t-100-epochs` | ce3a8d9 | 02a0929a |
| `planner-budget` | four planner settings on the same models | `baseline` | `orx/push-t-planner-budget` | db6fa17 | fd60bb49 |
| `supervision-pilot` | matched-supervision pilot, five seeds | `baseline` | `orx/push-t-supervision-pilot` | 2b89e79 | 148b7e11 |
| `action-chunks` | predict 8 actions, execute 4 | `baseline` | `orx/push-t-action-chunks` | 0522b45 | 003babe9 |
| `chunks-k16` | predict 16 actions, execute 8 | `action-chunks` | `orx/push-t-chunks-k-16-execute-8` | 2b3fd69 | d7fc0780 |
| `chunks-temporal-ensembling` | average overlapping chunks | `action-chunks` | `orx/push-t-temporal-ensembling` | 8463176 | fab64790 |
| `chunks-dmodel-64` | latent width 64 | `action-chunks` | `orx/push-t-chunks-d-model-64` | 2b3aa8c | 68826332 |
| `chunks-plain-cnn` | plain CNN image encoder instead of keypoints | `action-chunks` | `orx/push-t-chunks-plain-cnn-encoder` | 0e9a0ec | 48d74899 |
| `chunks-no-decoder` | no image decoder | `action-chunks` | `orx/push-t-chunks-no-image-decoder` | 7586bf7 | 5fdbef14 |
| `chunks-epochs-100` | 100 epochs | `action-chunks` | `orx/push-t-chunks-100-epochs` | a3e3a07 | 0bed5d45 |
| `chunks-branched` | world model also learns from simulator branches of the demos | `action-chunks` | `orx/push-t-chunks-simulator-branched-data` | 40c756f | ef4c299c |
| `chunks-restart-planner` | each plan search restarts from the policy's plan | `action-chunks` | `orx/push-t-chunks-planner-restarts-from-the-policy` | 580cf45 | e944bedd |
| `branched-restart-planner` | branched data and the restarting planner | `chunks-branched` | `orx/push-t-branched-data-planner-restarts-from-the-p` | 4946f33 | a9587dbf |
| `branched-restart-100-epochs` | 100 epochs | `branched-restart-planner` | `orx/push-t-branched-data-planner-restarts-100-epochs` | 0aa97c2 | 0378c41c |
| `final` | action head with 3 hidden layers of 512: the final configuration, and the defaults of `scripts/run.sh` | `branched-restart-100-epochs` | `orx/push-t-bigger-policy-head-3x512-mlp` | cbd8ead | 09fe70a3 |
| `head-3x1024` | action head 3 x 1024 | `final` | `orx/push-t-policy-head-3x1024` | 7f79088 | 3ebd6c3a |
| `head-6x512` | action head 6 x 512 | `final` | `orx/push-t-policy-head-6x512` | 6e6178c | 363ef8e6 |
| `flow-head-3x512` | flow-matching action head, 3 x 512 | `branched-restart-100-epochs` | `orx/push-t-flow-matching-policy-head-3x512` | 05e036e | 2b61d201 |
| `latent-256` | latent width 256, dynamics 512 wide, one-layer head | `branched-restart-100-epochs` | `orx/push-t-scale-up-2x-wider-latent` | 2e70aa7 | 12b3b471 |
| `head-3x512-latent-256` | latent width 256 | `final` | `orx/push-t-3x512-head-2x-wider-latent` | c0dc7ae | a336b859 |
| `head-3x512-no-decoder` | no image decoder | `final` | `orx/push-t-3x512-head-no-image-decoder` | 27f3acc | d75345a4 |
| `pixel-input-head-3x512` | images and language only, no state input | `final` | `orx/push-t-pixel-input-3x512-head` | 7fe8633 | 5f3cf816 |

The `planner-budget` job hit its 6-hour limit after all three seeds had been tested,
while computing the extras, so it has no `summary.json`; its numbers are read from the
seed files like every other folder. The supervision pilot's seed folders are named
`seed-0` to `seed-4` and hold `ablation/results.json`.

`final/figure-data/` also holds the data behind the illustrations, recorded from the final
model's first seed by `docs/figures/export_illustrations.py`: a branched demonstration
(`branches.npz`), one planning step with all its candidates (`planner.npz`), and a
successful planning episode (`episode.npz`).

All the experiment branches are merged into `main`, where each experiment is the same
`bash scripts/run.sh` with a few settings changed; the tutorial lists the command for every
folder. Run ids are the first 8 characters of the cluster run ids.
