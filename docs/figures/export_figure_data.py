"""Export the data behind the paper's qualitative figures.

Loads one run's trained checkpoints (one directory per seed, each holding
wam.pt and the run's <env>.npz, as written by scripts/run.sh) and saves
exactly what the figures show. The figures are then drawn from those files
only (docs/figures/make_figures.py), so every pixel and every curve can be
traced back to a model output.

    python docs/figures/export_figure_data.py --run-dir results/pusht --name baseline \\
        --out docs/results/pusht/figure-data

Episode views, what-if pushes and the latent map come from seed 0; keypoint
tracking scores and what-if errors cover every seed. Keypoint scores of
several runs (e.g. baseline and pixel-input) collect in one keypoint_stats.json.
Everything is computed on held-out episodes (the checkpoint's val_episodes).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from scripts.evaluate import load_model_and_data, get_episode, encode_episode, make_simulator  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "figure-data")
STRIP_FRAMES = 8
WHATIF_HORIZON = 15


def to_uint8(x):
    """(..., 3, H, W) float in [0, 1] -> (..., H, W, 3) uint8"""
    x = x.clamp(0, 1).cpu().numpy()
    return (np.moveaxis(x, -3, -1) * 255).round().astype(np.uint8)


def load(seed_dir):
    data_path = [p for p in glob.glob(os.path.join(seed_dir, "*.npz"))][0]
    return load_model_and_data(os.path.join(seed_dir, "wam.pt"), data_path)


def positions(states, env_name):
    """What the figures track: the agent, and on Push-T also the block (unit square, y up)."""
    out = {"agent": states[..., 0:2]}
    if env_name == "pusht":
        out["block"] = states[..., 4:6]
    return out


@torch.no_grad()
def export_episode_views(model, ckpt, data):
    """Imagination strip and belief traces for the first held-out episode."""
    episode = ckpt["val_episodes"][0]
    ep = get_episode(data, episode)
    T = ep["actions"].shape[0]

    z_real = encode_episode(model, ep)
    imag = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T, ep["actions"][None])
    dream = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T)
    recon = model.decode(z_real)

    cols = np.unique(np.linspace(0, T, STRIP_FRAMES).round().astype(int))
    if "image" in recon:  # the strip shows decoded frames: only for a WAM with an image decoder
        np.savez_compressed(
            os.path.join(OUT, "episode_strip.npz"),
            timesteps=cols,
            real=to_uint8(ep["images"][cols]),
            recon=to_uint8(recon["image"][cols]),
            imagined=to_uint8(imag["image"][0, cols]),
            dreamed=to_uint8(dream["image"][0, cols]),
            instruction=np.asarray(ep["text"]),
            episode=episode,
        )
    np.savez_compressed(
        os.path.join(OUT, "episode_beliefs.npz"),
        true=ep["states"].cpu().numpy(),
        now=recon["state"].cpu().numpy(),
        imagined=imag["state"][0].cpu().numpy(),
        dreamed=dream["state"][0].cpu().numpy(),
        labels=np.asarray(ckpt["state_labels"]),
        episode=episode,
    )


def push_plans(horizon, scale=1.0):
    """No-op plus constant pushes of size `scale` in 8 compass directions: (9, horizon, 2)."""
    angles = np.deg2rad(np.arange(0, 360, 45))
    dirs = np.concatenate([np.zeros((1, 2)), scale * np.stack([np.cos(angles), np.sin(angles)], 1)])
    return np.repeat(dirs[:, None, :], horizon, axis=1).astype(np.float32), ["no-op"] + [f"{a}°" for a in range(0, 360, 45)]


@torch.no_grad()
def imagine_plans(model, ep, t, plans):
    n = len(plans)
    return model.imagine(
        ep["images"][t : t + 1].expand(n, -1, -1, -1),
        ep["text_ids"].expand(n, -1),
        ep["states"][t : t + 1].expand(n, -1),
        plans.shape[1],
        torch.from_numpy(plans),
    )


@torch.no_grad()
def export_what_if(models, env_name):
    """Constant-push counterfactuals: one example (seed 0) + endpoint errors for all seeds."""
    sim, _, scale = make_simulator(env_name)
    plans, names = push_plans(WHATIF_HORIZON, scale)

    model, ckpt, data = models[0]
    ep = get_episode(data, ckpt["val_episodes"][0])
    s0 = ep["states"][0].cpu().numpy()
    out = imagine_plans(model, ep, 0, plans)
    truth = np.stack([np.concatenate([s0[None], sim.simulate(s0, p)]) for p in plans])
    example = dict(
        start_image=to_uint8(ep["images"][0]),
        start_state=s0,
        imagined=out["state"].cpu().numpy(),
        true=truth,
        names=np.asarray(names),
    )
    if "image" in out:
        example["final_frames"] = to_uint8(out["image"][:, -1])

    # endpoint error (Euclidean distance after the push, unit square) for every
    # held-out episode start of every seed: the agent, and on Push-T the block
    tracked = list(positions(np.zeros((1, 8)), env_name))
    errors = {k: np.zeros((len(models), len(plans))) for k in tracked}
    ignore = {k: np.zeros((len(models), len(plans))) for k in tracked}  # reference: the no-op endpoint for every push
    counts = []
    for i, (model, ckpt, data) in enumerate(models):
        per_ep = {k: [] for k in tracked}
        per_ep_ignore = {k: [] for k in tracked}
        for e in ckpt["val_episodes"]:
            ep = get_episode(data, e)
            s0 = ep["states"][0].cpu().numpy()
            pred = positions(imagine_plans(model, ep, 0, plans)["state"][:, -1].cpu().numpy(), env_name)
            true = positions(np.stack([sim.simulate(s0, p)[-1] for p in plans]), env_name)
            for k in tracked:
                per_ep[k].append(np.linalg.norm(pred[k] - true[k], axis=1))
                per_ep_ignore[k].append(np.linalg.norm(true[k][0][None] - true[k], axis=1))
        for k in tracked:
            errors[k][i] = np.mean(per_ep[k], axis=0)
            ignore[k][i] = np.mean(per_ep_ignore[k], axis=0)
        counts.append(len(per_ep["agent"]))

    extra = {}
    for k in tracked:
        prefix = "" if k == "agent" else "block_"
        extra[prefix + "endpoint_error"], extra[prefix + "ignore_push_error"] = errors[k], ignore[k]
    np.savez_compressed(os.path.join(OUT, "what_if.npz"), n_episodes=np.asarray(counts), **extra, **example)


@torch.no_grad()
def export_latent_map(model, ckpt, data):
    """PCA of held-out latents (seed 0) and correlation of components with state variables."""
    zs, ss, prog = [], [], []
    for e in ckpt["val_episodes"]:
        ep = get_episode(data, e)
        z = encode_episode(model, ep).cpu().numpy()
        zs.append(z)
        ss.append(ep["states"].cpu().numpy())
        prog.append(np.linspace(0, 1, len(z)))
    Z, S, P = np.concatenate(zs), np.concatenate(ss), np.concatenate(prog)

    mean = Z.mean(0)
    _, sing, Vt = np.linalg.svd(Z - mean, full_matrices=False)
    explained = sing ** 2 / (sing ** 2).sum()
    pcs = (Z - mean) @ Vt[:8].T
    corr = np.array([[np.corrcoef(pcs[:, i], S[:, j])[0, 1] for j in range(S.shape[1])] for i in range(8)])

    ep = get_episode(data, ckpt["val_episodes"][0])
    T = ep["actions"].shape[0]
    z_real = encode_episode(model, ep).cpu().numpy()
    z_imag = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], T, ep["actions"][None])["z"][0].cpu().numpy()

    np.savez_compressed(
        os.path.join(OUT, "latent_map.npz"),
        xy=pcs[:, :2],
        progress=P,
        explained=explained[:8],
        corr=corr,
        labels=np.asarray(ckpt["state_labels"]),
        real_traj=((z_real - mean) @ Vt[:2].T),
        imagined_traj=((z_imag - mean) @ Vt[:2].T),
    )


@torch.no_grad()
def keypoint_stats(model, ckpt, data, env_name):
    """
    How well do the spatial-softmax keypoints track the agent (and on Push-T the block)?

    For each keypoint, the squared correlation between its x (y) coordinate and
    the object's x (y) position over all held-out frames; the score is the best
    keypoint's mean of the two. Also returns the keypoints' mean spatial spread.
    """
    kps, states = [], []
    for e in ckpt["val_episodes"]:
        ep = get_episode(data, e)
        kps.append(model.img_encoder.keypoints(ep["images"]).cpu().numpy())   # (T+1, K, 2), x right, y down
        states.append(ep["states"].cpu().numpy())                              # positions y up
    K = np.concatenate(kps)
    scores = {}
    for name, P in positions(np.concatenate(states), env_name).items():
        r2x = np.array([np.corrcoef(K[:, k, 0], P[:, 0])[0, 1] ** 2 for k in range(K.shape[1])])
        r2y = np.array([np.corrcoef(K[:, k, 1], P[:, 1])[0, 1] ** 2 for k in range(K.shape[1])])
        scores[name] = float(np.nan_to_num((r2x + r2y) / 2).max())
    return scores, float(K.std(axis=0).mean())


@torch.no_grad()
def export_keypoints(runs, env_name):
    """Example frames (seed 0) and tracking scores (all seeds) for each run."""
    path = os.path.join(OUT, "keypoint_stats.json")
    stats = json.load(open(path)) if os.path.exists(path) else {}
    for name, models in runs.items():
        model, ckpt, data = models[0]
        ep = get_episode(data, ckpt["val_episodes"][0])
        T = ep["images"].shape[0]
        cols = np.unique(np.linspace(0, T - 1, STRIP_FRAMES).round().astype(int))
        kp = model.img_encoder.keypoints(ep["images"][cols]).cpu().numpy()
        np.savez_compressed(
            os.path.join(OUT, f"keypoints_{name}.npz"),
            frames=to_uint8(ep["images"][cols]),
            keypoints=kp,
            timesteps=cols,
        )
        scores = [keypoint_stats(*m, env_name) for m in models]
        stats[name] = {
            "tracking_r2": [s[0]["agent"] for s in scores],
            "spread": [s[1] for s in scores],
        }
        if env_name == "pusht":
            stats[name]["block_tracking_r2"] = [s[0]["block"] for s in scores]
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="A run's results directory with seed*/wam.pt")
    parser.add_argument("--name", required=True, help="Name of the run in keypoint_stats.json (e.g. baseline)")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--keypoints-only", action="store_true", help="Only export keypoints (e.g. for a pixel-input run)")
    args = parser.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)
    models = [load(d) for d in sorted(glob.glob(os.path.join(args.run_dir, "seed*")))]
    env_name = models[0][1]["env"]

    if not args.keypoints_only:
        export_episode_views(*models[0])
        export_what_if(models, env_name)
        export_latent_map(*models[0])
    stats = export_keypoints({args.name: models}, env_name)

    for f in sorted(os.listdir(OUT)):
        print("wrote", os.path.join(OUT, f))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
