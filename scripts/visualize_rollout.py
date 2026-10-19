"""Look inside the WAM's head: what does its latent world model believe?

Writes a set of figures for one unseen episode (and the whole unseen set):

    1_imagination.png   real frames vs. what the WAM reconstructs, imagines
                        (open-loop, real actions) and dreams (its own policy
                        acting inside its imagination)
    2_beliefs.png       every state variable over time: truth vs. what the
                        latent says now vs. what the WAM imagined from t=0
    3_latent_map.png    2D PCA map of the latent space: real trajectories vs.
                        imagined trajectories, and which latent directions
                        encode which physical quantities
    4_what_if.png       counterfactual imagination: same start, different
                        actions -> where does the WAM think the agent goes?
                        (pusht and toy envs: checked against the true simulator)
    5_horizon.png       how imagination error grows with the number of
                        imagined steps, against a "nothing changes" baseline
    6_keypoints.png     what the spatial-softmax encoder is tracking
                        (only with --image_encoder keypoints)
    dream.mp4           real vs. imagined video (--video)
    dream.gif           the same video as a GIF, as in the README (--gif)

Usage:
    python -m scripts.visualize_rollout --checkpoint checkpoints/wam.pt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.evaluate import (
    load_model_and_data,
    get_episode,
    encode_episode,
    evaluate,
    make_simulator,
    resolve_device,
)


# Okabe-Ito: colorblind-safe
C_TRUE = "#000000"
C_RECON = "#0072B2"
C_IMAG = "#D55E00"
C_DREAM = "#009E73"
C_COPY = "#999999"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--episode", type=int, default=None, help="Unseen episode id (default: first)")
    parser.add_argument("--horizon", type=int, default=None, help="Imagination horizon (default: episode length)")
    parser.add_argument("--frames", type=int, default=8, help="Columns in the imagination strip")
    parser.add_argument("--eval-horizon", type=int, default=15)
    parser.add_argument("--out-dir", type=str, default=None, help="Default: outputs/<checkpoint name>")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--gif", action="store_true", help="Also save the dream video as a GIF.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def to_np_img(x):
    """(3, H, W) in [0, 1] -> (H, W, 3) uint8"""
    return (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    print("Saved:", path)


# ------------------------------------------------------------ 1. imagination


@torch.no_grad()
def plot_imagination(model, ep, horizon, n_frames, out_dir):
    T = ep["actions"].shape[0]
    H = min(horizon, T)
    z_real = encode_episode(model, ep)

    # open-loop with the actions that were really taken
    imag = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], H, ep["actions"][None, :H])
    # dream: the policy chooses actions inside its own imagination
    dream = model.imagine(ep["images"][:1], ep["text_ids"], ep["states"][:1], H)
    recon = model.decode(z_real)

    cols = np.unique(np.linspace(0, H, n_frames).round().astype(int))
    rows = [("real", ep["images"])]
    if "image" in recon:
        rows += [
            ("reconstructed,\ndecoded z_t", recon["image"]),
            ("imagined from t=0,\nreal actions", imag["image"][0]),
            ("dreamed from t=0,\nown policy", dream["image"][0]),
        ]

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.1 * len(cols) + 1.2, 1.15 * len(rows)), squeeze=False)
    for r, (name, frames) in enumerate(rows):
        for c, t in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(to_np_img(frames[t]))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"t = {t}")
            if c == 0:
                ax.set_ylabel(name, rotation=0, ha="right", va="center")
    fig.suptitle(f'Imagination — "{ep["text"]}"', y=1.02)
    save(fig, out_dir, "1_imagination.png")

    return z_real, imag, dream


# ---------------------------------------------------------------- 2. beliefs


@torch.no_grad()
def plot_beliefs(model, ep, z_real, imag, dream, labels, out_dir):
    if not model.predict_state:
        return

    true = ep["states"].cpu().numpy()
    now = model.decode(z_real)["state"].cpu().numpy()
    im = imag["state"][0].cpu().numpy()
    dr = dream["state"][0].cpu().numpy()

    n = true.shape[1]
    # don't draw 39 Meta-World panels: keep the variables that actually move
    show = [j for j in range(n) if true[:, j].std() > 1e-6][:12]
    ncols = min(3, len(show))
    nrows = int(np.ceil(len(show) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 1.9 * nrows), squeeze=False, sharex=True)
    for i, j in enumerate(show):
        ax = axes.flat[i]
        ax.plot(true[:, j], color=C_TRUE, lw=1.6, label="true")
        ax.plot(now[:, j], color=C_RECON, lw=1.2, label="read from z_t, now")
        ax.plot(np.arange(len(im)), im[:, j], color=C_IMAG, lw=1.2, ls="--", label="imagined from t=0, real actions")
        ax.plot(np.arange(len(dr)), dr[:, j], color=C_DREAM, lw=1.2, ls=":", label="dreamed from t=0, own policy")
        ax.set_ylabel(labels[j])
    for ax in axes.flat[len(show):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("timestep")
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out_dir, "2_beliefs.png")


# ------------------------------------------------------------- 3. latent map


@torch.no_grad()
def plot_latent_map(model, data, episodes, ep, z_real, imag, labels, out_dir):
    zs, ss, ts = [], [], []
    for e in episodes:
        epi = get_episode(data, e, device=model.state_mean.device)
        z = encode_episode(model, epi).cpu().numpy()
        zs.append(z)
        ss.append(epi["states"].cpu().numpy())
        ts.append(np.linspace(0, 1, len(z)))
    Z, S, P = np.concatenate(zs), np.concatenate(ss), np.concatenate(ts)

    # PCA via SVD
    mean = Z.mean(axis=0)
    U, sing, Vt = np.linalg.svd(Z - mean, full_matrices=False)
    explained = sing ** 2 / (sing ** 2).sum()
    proj = lambda z: (z - mean) @ Vt[:2].T

    n_pc = min(8, Vt.shape[0])
    pcs = (Z - mean) @ Vt[:n_pc].T
    moving = [j for j in range(S.shape[1]) if S[:, j].std() > 1e-6][:16]
    corr = np.array([[np.corrcoef(pcs[:, i], S[:, j])[0, 1] for j in moving] for i in range(n_pc)])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.1, 1]})

    xy = proj(Z)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=P, cmap="viridis", s=4, alpha=0.5, linewidths=0)
    fig.colorbar(sc, ax=ax, label="episode progress, 0 = start, 1 = end", fraction=0.046)

    real_xy = proj(z_real.cpu().numpy())
    imag_xy = proj(imag["z"][0].cpu().numpy())
    ax.plot(real_xy[:, 0], real_xy[:, 1], color=C_TRUE, lw=1.6, label="real latent trajectory")
    ax.plot(imag_xy[:, 0], imag_xy[:, 1], color=C_IMAG, lw=1.6, ls="--", label="imagined from t=0")
    ax.scatter(*real_xy[0], color=C_TRUE, s=30, zorder=5, marker="o")
    ax.scatter(*real_xy[-1], color=C_TRUE, s=40, zorder=5, marker="*")
    ax.set_xlabel(f"PC1, {explained[0]:.0%} of variance")
    ax.set_ylabel(f"PC2, {explained[1]:.0%} of variance")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Latent space of unseen episodes")

    im = ax2.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(moving)))
    ax2.set_xticklabels([labels[j] for j in moving], rotation=60, ha="right")
    ax2.set_yticks(range(n_pc))
    ax2.set_yticklabels([f"PC{i + 1}, {explained[i]:.0%}" for i in range(n_pc)])
    for i in range(n_pc):
        for j in range(len(moving)):
            if abs(corr[i, j]) > 0.5:
                ax2.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(corr[i, j]) > 0.75 else "black")
    fig.colorbar(im, ax=ax2, label="Pearson correlation", fraction=0.046)
    ax2.set_title("What do the main latent directions encode?")
    fig.tight_layout()
    save(fig, out_dir, "3_latent_map.png")


# ---------------------------------------------------------------- 4. what if


@torch.no_grad()
def plot_what_if(model, ep, labels, env_name, out_dir, horizon=15):
    if not model.predict_state:
        return

    A = ep["actions"].shape[1]
    sim, _, scale = make_simulator(env_name)
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    plans = [np.zeros(A)]
    for a in angles:
        v = np.zeros(A)
        v[0], v[1] = (scale or 1.0) * np.cos(a), (scale or 1.0) * np.sin(a)
        plans.append(v)
    names = ["no-op"] + [f"{int(np.degrees(a))}°" for a in angles]

    dev = model.state_mean.device
    acts = torch.tensor(np.stack(plans), dtype=torch.float32, device=dev)[:, None].expand(-1, horizon, -1)
    n = len(plans)
    out = model.imagine(
        ep["images"][:1].expand(n, -1, -1, -1),
        ep["text_ids"].expand(n, -1),
        ep["states"][:1].expand(n, -1),
        horizon,
        acts,
    )
    pred = out["state"].cpu().numpy()  # (n, H+1, S)

    truth = None
    if sim is not None:
        s0 = ep["states"][0].cpu().numpy()
        truth = np.stack([np.concatenate([s0[None], sim.simulate(s0, acts[i].cpu().numpy())]) for i in range(n)])

    cmap = plt.get_cmap("twilight")
    has_img = "image" in out
    fig, axes = plt.subplots(1, 2 if has_img else 1, figsize=(10 if has_img else 5, 4.6), squeeze=False)
    ax = axes[0, 0]

    if env_name in ("pusht", "toy"):
        # both envs report positions in the unit square with y up, like the image
        ax.imshow(to_np_img(ep["images"][0]), extent=[0, 1, 0, 1], alpha=0.35)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    for i in range(n):
        color = "black" if i == 0 else cmap(i / n)
        ax.plot(pred[i, :, 0], pred[i, :, 1], color=color, lw=1.6, ls="--")
        ax.scatter(pred[i, -1, 0], pred[i, -1, 1], color=color, s=18, zorder=5)
        if truth is not None:
            ax.plot(truth[i, :, 0], truth[i, :, 1], color=color, lw=0.9, alpha=0.7)
    ax.plot([], [], color="gray", ls="--", label="WAM imagination")
    if truth is not None:
        ax.plot([], [], color="gray", lw=0.9, label="true simulator")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"What if I pushed in each direction for {horizon} steps?")

    if has_img:
        axes[0, 1].axis("off")
        grid = fig.add_gridspec(3, 6, left=0.55, right=0.98, top=0.88, bottom=0.08)
        order = [4, 3, 2, 5, 0, 1, 6, 7, 8]  # 3x3 compass layout (indices into plans)
        for cell, i in enumerate(order):
            sub = fig.add_subplot(grid[cell // 3, 2 * (cell % 3): 2 * (cell % 3) + 2])
            sub.imshow(to_np_img(out["image"][i, -1]))
            sub.set_xticks([]); sub.set_yticks([])
            sub.set_title(names[i], fontsize=7, pad=2)
        fig.text(0.765, 0.93, f"Imagined frame after {horizon} steps", ha="center")
    save(fig, out_dir, "4_what_if.png")


# ----------------------------------------------------------------- 5. horizon


def plot_horizon(results, labels, out_dir):
    k = np.array(results["horizon"])
    panels = [("state_nmse", "state_nmse_copy", "state error, normalized MSE")]
    if results["image_mse"]:
        panels.append(("image_mse", "image_mse_copy", "image error, pixel MSE"))
    has_r2 = bool(results["state_r2"])

    ncols = len(panels) + int(has_r2)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 2.9), squeeze=False)

    for ax, (m, c, ylabel) in zip(axes[0], panels):
        if results[m]:
            ax.plot(k, results[m], color=C_IMAG, marker="o", ms=3, label="WAM imagination")
        ax.plot(k, results[c], color=C_COPY, marker="s", ms=3, label='"nothing changes" baseline')
        ax.set_xlabel("imagined steps k")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    if has_r2:
        ax = axes[0, -1]
        r2 = np.array([[np.nan if v is None else v for v in row] for row in results["state_r2"]])
        moving = [j for j in range(r2.shape[1]) if not np.all(np.isnan(r2[:, j]))][:10]
        cmap = plt.get_cmap("tab10")
        for i, j in enumerate(moving):
            ax.plot(k, np.clip(r2[:, j], -0.2, 1), color=cmap(i), lw=1.2, label=labels[j])
        ax.set_ylim(-0.2, 1.02)
        ax.set_xlabel("imagined steps k")
        ax.set_ylabel("readout R², 1 = perfect")
        ax.legend(fontsize=6, ncol=2)

    fig.suptitle(f"Imagination quality on {len(results['episodes'])} unseen episodes", y=1.03)
    fig.tight_layout()
    save(fig, out_dir, "5_horizon.png")


# --------------------------------------------------------------- 6. keypoints


@torch.no_grad()
def plot_keypoints(model, ep, n_frames, out_dir):
    if not hasattr(model.img_encoder, "keypoints"):
        return
    T = ep["images"].shape[0]
    cols = np.unique(np.linspace(0, T - 1, n_frames).round().astype(int))
    kp = model.img_encoder.keypoints(ep["images"][cols]).cpu().numpy()  # (F, K, 2) in [-1, 1]
    size = ep["images"].shape[-1]
    K = kp.shape[1]
    cmap = plt.get_cmap("tab20")

    fig, axes = plt.subplots(1, len(cols), figsize=(1.3 * len(cols), 1.6), squeeze=False)
    for c, t in enumerate(cols):
        ax = axes[0, c]
        ax.imshow(to_np_img(ep["images"][t]))
        px = (kp[c, :, 0] + 1) / 2 * (size - 1)
        py = (kp[c, :, 1] + 1) / 2 * (size - 1)
        ax.scatter(px, py, c=[cmap(i % 20) for i in range(K)], s=10, marker="x", linewidths=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {t}")
    fig.suptitle(f"{K} spatial-softmax keypoints, one color per keypoint", y=1.08)
    save(fig, out_dir, "6_keypoints.png")


# ------------------------------------------------------------------- video


LOOK_EVERY = 4   # carry out 4 steps of the chosen plan, then look at the world again


@torch.no_grad()
def dream_for_real(model, ep, max_steps=300, every=LOOK_EVERY):
    """
    Push-T: the planner (scripts/plan.py) picks a 12-step plan inside the model's
    imagination; the first `every` steps of it are played in the simulator, and the
    next plan starts from what really happened, until the T covers the target or
    `max_steps` pass. Returns the imagined and the real states, each (T + 1, 8).
    """
    from envs.pusht_env import SUCCESS_COVERAGE, PushTEnv, coverage, from_state
    from scripts.plan import LatentPlanner

    env, planner = PushTEnv(image_size=model.config["image_size"]), LatentPlanner(model)
    dev = ep["states"].device
    s = ep["states"][0].cpu().numpy()
    dreamed, real = [s], [s]
    while len(real) <= max_steps and coverage(from_state(s)[4:7]) <= SUCCESS_COVERAGE:
        img = torch.from_numpy(env._render(s)).permute(2, 0, 1).float()[None].div(255.0).to(dev)
        state = torch.from_numpy(s).float()[None].to(dev)
        planner.act(img, ep["text_ids"], state)
        plan = planner.last["plan"][:every]
        imagined = model.imagine(img, ep["text_ids"], state, len(plan), actions=plan[None])["state"][0, 1:]
        dreamed += list(imagined.cpu().numpy())
        sim = env.simulate(s, plan.cpu().numpy())
        real += list(sim)
        s = sim[-1]
    return np.asarray(dreamed), np.asarray(real)


@torch.no_grad()
def save_dream_video(model, ep, imag, dream, out_dir, gif=False, env_name="toy"):
    """
    Push-T, four panels: the real episode | imagined from its first frame with the
    person's actions | dreamed plan, the plan the model picks inside its imagination,
    looking at the world again every LOOK_EVERY steps | the same plan played in the
    simulator. Frames are drawn from the states the model imagines, which its linear
    readout gives to within a few pixels; the image decoder's pictures are blurrier.
    Other envs: real | imagined | dreamed, as decoded images.
    """
    import imageio.v2 as imageio
    import cv2

    def label(img, text):
        img = cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
        return cv2.putText(img.copy(), text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

    if env_name == "pusht" and "state" in imag:
        from envs.pusht_env import draw

        T = imag["state"].shape[1] - 1
        dreamed, real = dream_for_real(model, ep)
        states, imagined = ep["states"].cpu().numpy(), imag["state"][0].cpu().numpy()
        n = max(T, len(real) - 1) + 8   # hold the last frames for a moment
        frames = [
            np.concatenate([
                label(draw(states[min(t, T)]), f"real t={min(t, T)}" + (", done" if t > T else "")),
                label(draw(imagined[min(t, T)]), "imagined"),
                label(draw(dreamed[min(t, len(dreamed) - 1)]), f"dreamed plan t={min(t, len(dreamed) - 1)}"),
                label(draw(real[min(t, len(real) - 1)]), "dream, for real"),
            ], axis=1)
            for t in range(n + 1)
        ]
    elif "image" in imag:
        frames = [
            np.concatenate([
                label(to_np_img(ep["images"][t]), f"real t={t}"),
                label(to_np_img(imag["image"][0, t]), "imagined"),  # under the real actions
                label(to_np_img(dream["image"][0, t]), "dreamed"),   # its own policy acting
            ], axis=1)
            for t in range(imag["image"].shape[1])
        ]
    else:
        return
    path = os.path.join(out_dir, "dream.mp4")
    with imageio.get_writer(path, fps=12) as w:
        for f in frames:
            w.append_data(f)
    print("Saved:", path)
    if gif:
        path = os.path.join(out_dir, "dream.gif")
        imageio.mimsave(path, frames, duration=1000 / 12, loop=0)
        print("Saved:", path)


def main():
    args = parse_args()
    args.device = resolve_device(args.device)
    model, ckpt, data = load_model_and_data(args.checkpoint, args.dataset_path, args.device)
    if not model.is_wam:
        raise SystemExit("This checkpoint is a VLA (no world model); nothing to visualize.")

    labels = ckpt["state_labels"]
    episodes = ckpt["val_episodes"] or sorted(set(data["episode_ids"].tolist()))
    episode = args.episode if args.episode is not None else episodes[0]

    out_dir = args.out_dir or os.path.join("outputs", os.path.splitext(os.path.basename(args.checkpoint))[0])
    os.makedirs(out_dir, exist_ok=True)

    ep = get_episode(data, episode, device=args.device)
    horizon = args.horizon or ep["actions"].shape[0]
    print(f"Episode {episode}: {ep['actions'].shape[0]} steps, \"{ep['text']}\"")

    z_real, imag, dream = plot_imagination(model, ep, horizon, args.frames, out_dir)
    plot_beliefs(model, ep, z_real, imag, dream, labels, out_dir)
    plot_latent_map(model, data, episodes, ep, z_real, imag, labels, out_dir)
    plot_what_if(model, ep, labels, ckpt["env"], out_dir)
    results = evaluate(model, data, episodes, args.eval_horizon)
    results["episodes"] = episodes
    plot_horizon(results, labels, out_dir)
    plot_keypoints(model, ep, args.frames, out_dir)
    if args.video or args.gif:
        save_dream_video(model, ep, imag, dream, out_dir, gif=args.gif, env_name=ckpt["env"])


if __name__ == "__main__":
    main()
