"""Branch the Push-T demonstrations with the simulator: "what if I had pushed differently?"

    python -m scripts.branch_demos data/pusht.npz --output-path data/pusht_branched.npz

Human demonstrations only show what *good* pushes do. A world model trained
on them predicts the block well along the human's actions, but a planner
asks about other actions too, and there the model has never seen what
happens. On Push-T, under the human's actions plus a little noise, its
predicted block pose 10 steps ahead is no better than assuming the block
does not move at all, and the planner picks exactly those actions whose
outcome the model gets wrong.

This script fills that gap with the exact simulator (envs/pusht_env.py):

    1. take a state the human visited (every --every steps of every demo)
    2. replay the human's next --length actions with random noise added
       (std --noise, so the agent pushes a bit harder, softer or sideways)
    3. let the simulator say what really happens, and draw every new state
       with the env's renderer

Each branch becomes a short extra episode marked is_expert = 0: the policy
never imitates these actions, only the world model learns from them. Each
branch also remembers the demo it started from (source_episodes), so
scripts/train.py puts it on the same side of the train/val split and the
evaluation still uses only unseen human demos.

The output holds the original demos unchanged, followed by the branches.
"""

import argparse

import numpy as np

from envs.pusht_env import PushTEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="A Push-T dataset, e.g. from scripts/download_pusht.py")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--every", type=int, default=4, help="Start a branch at every N-th demo state.")
    parser.add_argument("--length", type=int, default=8, help="Actions per branch.")
    parser.add_argument("--noise", type=float, default=0.3, help="Std of the noise added to the human's actions.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = dict(np.load(args.input, allow_pickle=True))
    episodes, L = data["episode_ids"], args.length
    env = PushTEnv(image_size=data["images"].shape[1])
    rng = np.random.default_rng(args.seed)

    new = {k: [] for k in ["images", "states", "actions", "next_images", "next_states", "episode_ids",
                           "source_episodes", "timesteps", "text_ids", "texts"]}
    next_id = int(episodes.max()) + 1
    for i in range(0, len(episodes), args.every):
        if i + L > len(episodes) or episodes[i + L - 1] != episodes[i]:
            continue  # the demo ends before the branch would

        # the human's next L actions, pushed a little differently
        actions = np.clip(data["actions"][i : i + L] + args.noise * rng.standard_normal((L, 2)), -1.0, 1.0)
        actions = actions.astype(np.float32)
        states = np.concatenate([data["states"][i : i + 1], env.simulate(data["states"][i], actions)])  # (L+1, 8)
        images = np.stack([env._render(s) for s in states])

        new["images"].append(images[:-1])
        new["states"].append(states[:-1])
        new["actions"].append(actions)
        new["next_images"].append(images[1:])
        new["next_states"].append(states[1:])
        new["episode_ids"].append(np.full(L, next_id))
        new["source_episodes"].append(np.full(L, episodes[i]))
        new["timesteps"].append(data["timesteps"][i] + np.arange(L))
        new["text_ids"].append(np.repeat(data["text_ids"][i : i + 1], L, axis=0))
        new["texts"].append(np.repeat(data["texts"][i : i + 1], L))
        next_id += 1

    n_demo, n_branch = len(episodes), next_id - int(episodes.max()) - 1
    out = dict(data)
    for k, parts in new.items():
        before = episodes if k == "source_episodes" else data[k]  # a demo is its own source
        out[k] = np.concatenate([before] + parts).astype(before.dtype)
    out["is_expert"] = np.concatenate([data["is_expert"], np.zeros(n_branch * L, np.float32)])
    out["branch"] = np.asarray(f"every={args.every} length={L} noise={args.noise} seed={args.seed}")
    np.savez_compressed(args.output_path, **out)

    moved = np.abs(np.concatenate(new["next_states"])[:, 4:6] - np.concatenate(new["states"])[:, 4:6]).max(1) > 1e-4
    print(f"Saved demos + simulator branches to: {args.output_path}")
    print(f"  {n_demo} demo transitions, {n_branch} branches x {L} = {n_branch * L} branch transitions "
          f"(noise {args.noise}; the block moves in {moved.mean():.0%} of them)")


if __name__ == "__main__":
    main()
