"""Merge pico-WAM datasets of the same env into one.

    python -m scripts.merge_datasets data/pusht.npz data/pusht_expert.npz --output-path data/pusht_mixed.npz

Episodes keep their order and get new, consecutive ids; the vocabulary is
rebuilt from all instructions. Every transition keeps its is_expert flag, so
the policy still imitates only demonstrated actions while the world model
learns from all of them. A `source` array records which input file each
transition came from.
"""

import argparse

import numpy as np

from scripts.collect_data import tokenize_texts

PER_TRANSITION = ["images", "states", "actions", "next_images", "next_states", "is_expert", "timesteps", "texts"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    parts = [dict(np.load(p, allow_pickle=True)) for p in args.inputs]
    envs = {str(p["env"]) for p in parts}
    labels = {tuple(str(l) for l in p["state_labels"]) for p in parts}
    if len(envs) != 1 or len(labels) != 1:
        raise ValueError(f"Inputs differ in env or state layout: {envs}")

    merged = {k: np.concatenate([p[k] for p in parts]) for k in PER_TRANSITION}
    episode_ids, offset = [], 0
    for p in parts:
        _, ids = np.unique(p["episode_ids"], return_inverse=True)  # 0..n-1, in order
        episode_ids.append(ids + offset)
        offset += ids.max() + 1
    merged["episode_ids"] = np.concatenate(episode_ids).astype(np.int64)
    merged["source"] = np.concatenate([np.full(len(p["actions"]), i) for i, p in enumerate(parts)])
    merged["text_ids"], merged["vocab"] = tokenize_texts(merged["texts"])
    merged["state_labels"] = parts[0]["state_labels"]
    merged["env"] = parts[0]["env"]
    merged["inputs"] = np.asarray(args.inputs)

    np.savez_compressed(args.output_path, **merged)
    for i, p in enumerate(parts):
        print(f"  {args.inputs[i]}: {len(np.unique(p['episode_ids']))} episodes, {len(p['actions'])} transitions, "
              f"expert fraction {p['is_expert'].mean():.2f}")
    print("Saved merged dataset to:", args.output_path, f"({offset} episodes, {len(merged['actions'])} transitions)")


if __name__ == "__main__":
    main()
