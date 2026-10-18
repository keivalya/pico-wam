"""Offline evaluation of a trained pico-WAM world model on unseen episodes.

Questions this answers (all on episodes the model never trained on):

1. Does the latent know the current world?        state readout R^2 at k = 0
2. Can it imagine the future?                      state / image error vs imagination step k
3. Is imagination better than "nothing changes"?   compared against a copy-current baseline
4. Which physical quantities live in the latent?   R^2 per state variable
5. Does it understand *actions*?                   pusht and toy envs: imagine
   the future under random action sequences the expert never took, and
   compare against the true simulator ("counterfactual error"). A model that
   ignores actions scores like the "no-op" baseline, which predicts the
   future the agent would have had without acting.

The imagination here is open-loop: encode o_t once, then roll the latent
dynamics forward with the actions that were really taken, never looking at
o_t+1 ... o_t+k.

Usage:
    python -m scripts.evaluate --checkpoint checkpoints/wam.pt --dataset-path data/pusht.npz
"""

import argparse
import json
import os

import numpy as np
import torch

from models.wam_policy import PicoWAMPolicy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--dataset-path", type=str, default=None, help="Defaults to the training dataset.")
    parser.add_argument("--max-horizon", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None, help="Defaults to <checkpoint>_eval.json")
    return parser.parse_args()


# ----------------------------------------------------------------- loading


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model_and_data(checkpoint, dataset_path=None, device="cpu"):
    device = resolve_device(device)
    model, ckpt = PicoWAMPolicy.from_checkpoint(checkpoint, device=device)
    data = dict(np.load(dataset_path or ckpt["dataset_path"], allow_pickle=True))
    return model, ckpt, data


def to_image_tensor(images, device):
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float in [0, 1]"""
    return torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255.0).to(device)


def get_episode(data, episode, device="cpu"):
    """
    All T transitions of one episode as T+1 observations and T actions:
        images  : (T+1, 3, H, W)
        states  : (T+1, state_dim)
        actions : (T, action_dim)
        text_ids: (1, T_text)
    """
    idx = np.nonzero(data["episode_ids"] == episode)[0]
    images = np.concatenate([data["images"][idx], data["next_images"][idx[-1:]]], axis=0)
    states = np.concatenate([data["states"][idx], data["next_states"][idx[-1:]]], axis=0)

    return {
        "images": to_image_tensor(images, device),
        "states": torch.from_numpy(states).float().to(device),
        "actions": torch.from_numpy(data["actions"][idx]).float().to(device),
        "text_ids": torch.from_numpy(data["text_ids"][idx[:1]]).long().to(device),
        "text": str(data["texts"][idx[0]]) if "texts" in data else "",
    }


# --------------------------------------------------------------- latents


@torch.no_grad()
def encode_episode(model, ep):
    """Real latents z_0 ... z_T, each encoded from its own observation. (T+1, D)"""
    n = ep["images"].shape[0]
    return model.encode_obs(ep["images"], ep["text_ids"].expand(n, -1), ep["states"])


@torch.no_grad()
def imagine_from_every_start(model, ep, horizon):
    """
    For every start t, imagine k = 1..horizon steps with the real actions.

    returns z_imag: (T, horizon, D) where z_imag[t, k-1] is the latent the model
    imagines for time t+k having only seen o_t. Entries past the end of the
    episode are NaN.
    """
    z_real = encode_episode(model, ep)
    T, A = ep["actions"].shape
    D = z_real.shape[-1]

    acts = torch.zeros(T, horizon, A, device=z_real.device)
    valid = torch.zeros(T, horizon, dtype=torch.bool, device=z_real.device)
    for t in range(T):
        k = min(horizon, T - t)
        acts[t, :k] = ep["actions"][t : t + k]
        valid[t, :k] = True

    z_imag = model.dynamics.rollout(z_real[:T], acts)
    z_imag = torch.where(valid[..., None], z_imag, torch.full_like(z_imag, float("nan")))
    return z_real, z_imag.view(T, horizon, D)


# --------------------------------------------------------------- metrics


def r2_per_dim(pred, true):
    """Coefficient of determination per column; NaN for (near-)constant columns."""
    ss_res = ((pred - true) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r2[ss_tot < 1e-8 * len(true)] = np.nan
    return r2


@torch.no_grad()
def evaluate(model, data, episodes, max_horizon=15):
    """
    Collect, for every unseen (start t, step k) pair:
        true state s_t+k, imagined readout, copy baseline s_t
        image MSE of the imagined decode and of the copy baseline image_t
    """
    K = max_horizon
    per_k = {k: {"true": [], "pred": [], "copy": [], "img": [], "img_copy": []} for k in range(K + 1)}

    for e in episodes:
        ep = get_episode(data, e, device=model.state_mean.device)
        z_real, z_imag = imagine_from_every_start(model, ep, K)
        T = ep["actions"].shape[0]

        dec_real = model.decode(z_real)
        dec_imag = model.decode(torch.nan_to_num(z_imag))

        for t in range(T + 1):
            # k = 0: read out the real latent (reconstruction)
            per_k[0]["true"].append(ep["states"][t])
            per_k[0]["copy"].append(ep["states"][t])
            if "state" in dec_real:
                per_k[0]["pred"].append(dec_real["state"][t])
            if "image" in dec_real:
                per_k[0]["img"].append(((dec_real["image"][t] - ep["images"][t]) ** 2).mean())
                per_k[0]["img_copy"].append(torch.zeros(()))

        for t in range(T):
            for k in range(1, min(K, T - t) + 1):
                per_k[k]["true"].append(ep["states"][t + k])
                per_k[k]["copy"].append(ep["states"][t])
                if "state" in dec_imag:
                    per_k[k]["pred"].append(dec_imag["state"][t, k - 1])
                if "image" in dec_imag:
                    per_k[k]["img"].append(((dec_imag["image"][t, k - 1] - ep["images"][t + k]) ** 2).mean())
                    per_k[k]["img_copy"].append(((ep["images"][t] - ep["images"][t + k]) ** 2).mean())

    std = model.state_std.cpu().numpy()
    results = {"horizon": [], "n": [], "state_nmse": [], "state_nmse_copy": [],
               "image_mse": [], "image_mse_copy": [], "state_r2": [],
               "state_nmse_per_dim": [], "state_nmse_copy_per_dim": []}

    for k in range(K + 1):
        d = per_k[k]
        if not d["true"]:
            break
        true = torch.stack(d["true"]).cpu().numpy()
        copy = torch.stack(d["copy"]).cpu().numpy()

        results["horizon"].append(k)
        results["n"].append(len(true))
        results["state_nmse_copy"].append(float((((copy - true) / std) ** 2).mean()))
        results["state_nmse_copy_per_dim"].append((((copy - true) / std) ** 2).mean(0).tolist())

        if d["pred"]:
            pred = torch.stack(d["pred"]).cpu().numpy()
            results["state_nmse"].append(float((((pred - true) / std) ** 2).mean()))
            results["state_nmse_per_dim"].append((((pred - true) / std) ** 2).mean(0).tolist())
            results["state_r2"].append([None if np.isnan(r) else float(r) for r in r2_per_dim(pred, true)])
        if d["img"]:
            results["image_mse"].append(float(torch.stack(d["img"]).mean()))
            results["image_mse_copy"].append(float(torch.stack(d["img_copy"]).mean()))

    return results


def make_simulator(env_name):
    """
    The true simulator of an env, the state dimensions its actions move, and
    the size of the random actions to test it with:
        pusht: everything (agent position/velocity, block pose); random
               targets up to 40 px from the agent, about twice the human
               demonstrations' typical 19 px
        toy:   agent_x, agent_y, vel_x, vel_y (the goal never moves); any force
    """
    if env_name == "pusht":
        from envs.pusht_env import PushTEnv, STEP
        return PushTEnv(), list(range(8)), 40.0 / STEP
    if env_name == "toy":
        from envs.toy_point_env import ToyPointEnv
        return ToyPointEnv(), [0, 1, 2, 3], 1.0
    return None, None, None


@torch.no_grad()
def evaluate_counterfactuals(model, data, episodes, env_name="toy", horizon=10, n_plans=8, every=5, seed=0):
    """
    For starts every `every` steps of each unseen episode, draw `n_plans`
    random action sequences, imagine with each, and compare the imagined
    state with the true simulator.

    returns per-k normalized MSE of:
        wam   : imagined vs true future under the random actions
        noop  : true future under zero actions vs true future under the random
                actions (what you'd score by ignoring the actions)
    """
    sim, dims, scale = make_simulator(env_name)
    rng = np.random.default_rng(seed)
    std = model.state_std.cpu().numpy()[dims]
    dev = model.state_mean.device

    err_wam = np.zeros(horizon)
    err_noop = np.zeros(horizon)
    dim_wam = np.zeros((horizon, len(dims)))    # the same errors, per state variable
    dim_noop = np.zeros((horizon, len(dims)))
    count = 0

    for e in episodes:
        ep = get_episode(data, e, device=dev)
        T = ep["actions"].shape[0]
        for t in range(0, T, every):
            plans = rng.uniform(-scale, scale, size=(n_plans, horizon, 2)).astype(np.float32)
            s0 = ep["states"][t].cpu().numpy()

            truth = np.stack([sim.simulate(s0, p) for p in plans])[..., dims]           # (P, H, S)
            noop = sim.simulate(s0, np.zeros((horizon, 2), np.float32))[None, :, dims]  # (1, H, S)

            out = model.imagine(
                ep["images"][t : t + 1].expand(n_plans, -1, -1, -1),
                ep["text_ids"].expand(n_plans, -1),
                ep["states"][t : t + 1].expand(n_plans, -1),
                horizon,
                torch.from_numpy(plans).to(dev),
            )
            pred = out["state"][:, 1:].cpu().numpy()[..., dims]                          # (P, H, S)

            err_wam += (((pred - truth) / std) ** 2).mean(axis=(0, 2))
            err_noop += (((noop - truth) / std) ** 2).mean(axis=(0, 2))
            dim_wam += (((pred - truth) / std) ** 2).mean(axis=0)
            dim_noop += (((noop - truth) / std) ** 2).mean(axis=0)
            count += 1

    return {
        "horizon": list(range(1, horizon + 1)),
        "wam_nmse": (err_wam / count).tolist(),
        "noop_nmse": (err_noop / count).tolist(),
        "wam_nmse_per_dim": (dim_wam / count).tolist(),
        "noop_nmse_per_dim": (dim_noop / count).tolist(),
        "dims": dims,
        "n_starts": count,
        "n_plans": n_plans,
        "action_scale": scale,
    }


@torch.no_grad()
def evaluate_dreams(model, data, episodes, horizon=40, every=20, seed=0):
    """
    Push-T: is a dream right? From a start every `every` steps of each unseen episode,
    the policy acts inside the model's imagination for `horizon` steps; the simulator
    then plays the same actions from the same state. Medians over starts, in pixels:
    how far the imagined agent and T (mean corner distance) are from the real ones.
    """
    from envs.pusht_env import PushTEnv, T_OUTLINE, from_state, to_world

    env = PushTEnv(image_size=model.config["image_size"])
    torch.manual_seed(seed)
    agent, block = [], []
    for e in episodes:
        ep = get_episode(data, e, device=model.state_mean.device)
        for t0 in range(0, ep["actions"].shape[0] - horizon, every):
            d = model.imagine(ep["images"][t0:t0 + 1], ep["text_ids"], ep["states"][t0:t0 + 1], horizon)
            dream = d["state"][0, 1:].cpu().numpy()
            real = env.simulate(ep["states"][t0].cpu().numpy(), d["actions"][0].cpu().numpy())
            P, R = [from_state(x) for x in dream], [from_state(x) for x in real]
            agent.append([float(np.linalg.norm(p[:2] - r[:2])) for p, r in zip(P, R)])
            block.append([float(np.linalg.norm(to_world(T_OUTLINE, p[4:7]) - to_world(T_OUTLINE, r[4:7]), axis=1).mean())
                          for p, r in zip(P, R)])
    agent, block = np.asarray(agent), np.asarray(block)
    return {"horizon": list(range(1, horizon + 1)), "agent_px": np.median(agent, 0).tolist(),
            "block_px": np.median(block, 0).tolist(), "n_starts": int(len(agent))}


@torch.no_grad()
def evaluate_moving_pixels(model, data, episodes, threshold=0.1):
    """Decoder error on the pixels that move: where a frame differs from the mean image (the T and the agent)."""
    mean = model.image_decoder.mean_image
    num, den = 0.0, 0.0
    for e in episodes:
        ep = get_episode(data, e, device=mean.device)
        recon = model.decode(encode_episode(model, ep)[:, None])["image"][:, 0]
        mask = ((ep["images"] - mean).abs().amax(dim=1, keepdim=True) > threshold).float()
        num += float((((recon - ep["images"]) ** 2) * mask).sum())
        den += float(mask.sum()) * 3
    return num / max(den, 1.0)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    model, ckpt, data = load_model_and_data(args.checkpoint, args.dataset_path, args.device)

    if not model.is_wam:
        raise SystemExit("This checkpoint is a VLA (no world model); nothing to evaluate here.")

    episodes = ckpt["val_episodes"] or sorted(set(data["episode_ids"].tolist()))
    results = evaluate(model, data, episodes, args.max_horizon)
    results["state_labels"] = ckpt["state_labels"]
    results["episodes"] = episodes
    results["checkpoint"] = args.checkpoint

    labels = ckpt["state_labels"]
    print(f"Unseen episodes: {len(episodes)}")
    print(f"{'k':>3} {'n':>6} {'state nMSE':>11} {'copy':>8} {'image MSE':>10} {'copy':>8}")
    for i, k in enumerate(results["horizon"]):
        s = results["state_nmse"][i] if results["state_nmse"] else float("nan")
        im = results["image_mse"][i] if results["image_mse"] else float("nan")
        imc = results["image_mse_copy"][i] if results["image_mse_copy"] else float("nan")
        print(f"{k:>3} {results['n'][i]:>6} {s:>11.4f} {results['state_nmse_copy'][i]:>8.4f} {im:>10.5f} {imc:>8.5f}")

    if results["state_r2"]:
        print("\nState readout R^2 (k=0: from the real latent, k>0: from the imagined latent)")
        show = [k for k in [0, 1, 5, 10, args.max_horizon] if k < len(results["state_r2"])]
        print(f"{'variable':>14} " + " ".join(f"{'k=' + str(k):>7}" for k in sorted(set(show))))
        for j, name in enumerate(labels):
            row = [results["state_r2"][k][j] for k in sorted(set(show))]
            print(f"{name:>14} " + " ".join(f"{'—':>7}" if r is None else f"{r:>7.3f}" for r in row))

    if ckpt.get("env") in ("pusht", "toy") and model.predict_state:
        cf = evaluate_counterfactuals(model, data, episodes, env_name=ckpt["env"])
        results["counterfactual"] = cf
        print(f"\nCounterfactual imagination ({cf['n_starts']} starts x {cf['n_plans']} random plans)")
        print(f"{'k':>3} {'WAM nMSE':>9} {'no-op':>8}")
        for k, w, n in zip(cf["horizon"], cf["wam_nmse"], cf["noop_nmse"]):
            print(f"{k:>3} {w:>9.4f} {n:>8.4f}")

    if ckpt.get("env") == "pusht" and model.predict_state:
        dr = evaluate_dreams(model, data, episodes)
        results["dream"] = dr
        print(f"\nDreams checked in the simulator ({dr['n_starts']} starts): median error in pixels")
        for k in [1, 5, 10, 20, 40]:
            print(f"  k={k:>2}  agent {dr['agent_px'][k - 1]:5.1f}  T {dr['block_px'][k - 1]:5.1f}")
    if model.predict_image:
        results["image_mse_moving"] = evaluate_moving_pixels(model, data, episodes)
        print(f"\nDecoded frames, error on the moving pixels only: {results['image_mse_moving']:.5f}")

    output = args.output or os.path.splitext(args.checkpoint)[0] + "_eval.json"
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved:", output)


if __name__ == "__main__":
    main()
