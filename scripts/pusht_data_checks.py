"""Checks of the Push-T data and simulator that the paper quotes.

On a converted LeRobot dataset (scripts/download_pusht.py) and the env:

    replay       from each recorded state and action, how far is the
                 simulator's next block position from the recorded one?
                 (median and 99th percentile, pixels) and how far does a
                 moving block actually travel per step?
    render       mean absolute pixel difference between the env's image of a
                 recorded state and the dataset's frame (0-255)
    demos        the human demonstrations' best coverage and successes
    expert       the scripted expert's success rate and episode length on
                 30 starts, without and with random bursts (p = 0.1)

    python -m scripts.pusht_data_checks --dataset data/pusht.npz --output docs/results/pusht/data_checks.json
"""

import argparse
import json

import numpy as np

from envs.pusht_env import PushTEnv, coverage, from_state


def replay(d, env, every=7):
    S, A, NS, ids = d["states"], d["actions"], d["next_states"], d["episode_ids"]
    err, move = [], []
    for i in range(0, len(S), every):
        pred = from_state(env.simulate(S[i], A[i : i + 1])[0])
        true, now = from_state(NS[i]), from_state(S[i])
        err.append(np.linalg.norm(pred[4:6] - true[4:6]))
        step = np.linalg.norm(true[4:6] - now[4:6])
        if step > 0.5:
            move.append(step)
    err, move = np.asarray(err), np.asarray(move)
    return {"block_error_px_p50": float(np.median(err)), "block_error_px_p99": float(np.percentile(err, 99)),
            "moving_block_step_px_median": float(np.median(move)), "n_checked": int(len(err))}


def render(d, env, every=50):
    diffs = [np.abs(env._render(d["states"][i]).astype(float) - d["images"][i]).mean() for i in range(0, len(d["states"]), every)]
    return {"mean_abs_pixel_difference": float(np.mean(diffs)), "max": float(np.max(diffs)), "n_frames": len(diffs)}


def demos(d):
    ids = d["episode_ids"]
    best = [max(coverage(from_state(s)[4:7]) for s in d["next_states"][ids == e]) for e in np.unique(ids)]
    best = np.asarray(best)
    return {"episodes": int(len(best)), "best_coverage_mean": float(best.mean()), "successes": int((best > 0.95).sum())}


def expert(p, n=30, max_steps=300, seed=0):
    env, rng = PushTEnv(seed=seed), np.random.default_rng(seed)
    wins, steps = [], []
    for ep in range(n):
        env.reset(seed=1000 + ep)
        burst, done, t = 0, False, 0
        while not done and t < max_steps:
            if burst == 0 and rng.random() < p:
                burst = int(rng.integers(1, 4))
            if burst:
                a = rng.uniform(-1, 1, 2)
                burst -= 1
            else:
                a = env.expert_action(noise=0.05 if p > 0 else 0.0)
            _, _, _, done, info = env.step(a)
            t += 1
        wins.append(info["success"])
        steps.append(t)
    return {"random_action_prob": p, "episodes": n, "success_rate": float(np.mean(wins)), "mean_steps": float(np.mean(steps))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    d = dict(np.load(args.dataset, allow_pickle=True))
    env = PushTEnv(image_size=d["images"].shape[1])
    clipped = int(np.any(np.abs(d["actions"]) >= 1.0 - 1e-6, axis=1).sum())   # targets beyond 100 px on an axis
    out = {"dataset": args.dataset, "source": str(d.get("source", "")), "clipped_actions": clipped,
           "transitions": int(len(d["actions"])),
           "replay": replay(d, env), "render": render(d, env), "demos": demos(d),
           "expert": [expert(0.0), expert(0.1)]}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
