"""Data for the Push-T illustrations, from one trained run of scripts/run.sh.

docs/figures/pusht_figures.py draws them; this script only records states,
so the pictures can be redrawn without the checkpoint:

    branches.npz   one human demonstration and the simulator branches started
                   along a stretch of it (scripts/branch_demos.py)
    planner.npz    one typical planning step of that episode: the 256 candidate
                   plans, the states the world model imagines for each, their
                   costs, and what the simulator says the chosen plan and the
                   policy's plan do
    episode.npz    the first successful planning episode on the unseen starts:
                   the state and the coverage of the target at every step

    python docs/figures/export_illustrations.py --run-dir results/pusht/seed0 \
        --out docs/results/pusht/final/figure-data
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from envs.pusht_env import PushTEnv, coverage, distance_to_t, from_state  # noqa: E402
from models.wam_policy import PicoWAMPolicy  # noqa: E402
from scripts.plan import LatentPlanner  # noqa: E402
from utils.tokenizer import SimpleTokenizer  # noqa: E402


def export_branches(path, out, window=32):
    """The demo whose block moves most in a 32-step stretch, and the branches started there."""
    d = np.load(path, allow_pickle=True)
    ids, src, ts = d["episode_ids"], d["source_episodes"], d["timesteps"]
    demos = np.unique(ids[ids == src])
    best = None
    for e in demos:
        S = d["states"][ids == e]
        if len(S) < window + 8:
            continue
        step = np.linalg.norm(np.diff(S[:, 4:6], axis=0), axis=1)
        moved = np.convolve(step, np.ones(window), "valid")
        a = int(np.argmax(moved))
        if best is None or moved[a] > best[0]:
            best = (moved[a], e, a)
    _, e, a = best
    demo = np.concatenate([d["states"][ids == e], d["next_states"][ids == e][-1:]])
    branches, starts = [], []
    for b in np.unique(ids[(src == e) & (ids != e)]):
        rows = ids == b
        t0 = int(ts[rows][0])
        if a <= t0 < a + window:
            branches.append(np.concatenate([d["states"][rows][:1], d["next_states"][rows]]))
            starts.append(t0)
    order = np.argsort(starts)
    np.savez_compressed(os.path.join(out, "branches.npz"), demo=demo, window=np.array([a, a + window]),
                        branches=np.stack(branches)[order], starts=np.array(starts)[order], episode=e)
    print(f"branches: demo {e}, steps {a}-{a + window}, {len(branches)} branches")


RECORDER = torch.Generator().manual_seed(1)   # its own noise, so recording never changes the episode


@torch.no_grad()
def candidates(planner, img_t, text_ids, state_t):
    """One planning step, as LatentPlanner.act does it, keeping every candidate."""
    model, H, N = planner.model, planner.horizon, planner.samples
    z0 = model.encode_obs(img_t, text_ids, state_t)
    prior = model.imagine(img_t, text_ids, state_t, H)["actions"][0]
    eps = torch.randn((N, H, prior.shape[-1]), generator=RECORDER, device=prior.device)
    seqs = (prior + planner.noise * eps).clamp(-1.0, 1.0)
    seqs[0] = seqs[1] = prior
    states = model.denormalize_state(model.state_readout(model.dynamics.rollout(z0.expand(N, -1), seqs)))
    costs = planner.cost(states)
    return seqs.cpu().numpy(), states.cpu().numpy(), costs.cpu().numpy()


def plan_error(a, b):
    """Mean distance in pixels between the T's corners in two states."""
    from envs.pusht_env import T_OUTLINE, to_world
    return float(np.linalg.norm(to_world(T_OUTLINE, from_state(a)[4:7]) - to_world(T_OUTLINE, from_state(b)[4:7]), axis=1).mean())


def run_episodes(model, tokenizer, out, seed, max_steps=300, tries=10, every=10):
    """Plan on unseen starts until one succeeds; from that episode keep a typical planning step:
    the agent at the T, and the chosen plan's imagined end pose as far from the real one as the median."""
    env, sim = PushTEnv(image_size=model.config["image_size"]), PushTEnv(image_size=model.config["image_size"])
    for ep in range(tries):
        planner = LatentPlanner(model)
        img, state, info = env.reset(seed=seed + ep)
        text_ids = torch.tensor([tokenizer.encode(info["instruction"])], dtype=torch.long)
        states, covs, done, t, steps = [state], [info["coverage"]], False, 0, []
        while not done and t < max_steps:
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0).div(255.0)
            state_t = torch.from_numpy(np.asarray(state)).float().unsqueeze(0)
            phys = from_state(state)
            if t % every == 0 and distance_to_t(phys[:2], phys[4:7]) < 30:
                seqs, imagined, costs = candidates(planner, img_t, text_ids, state_t)
                chosen = int(np.argmin(costs))
                real = sim.simulate(state, seqs[chosen])
                steps.append(dict(state=state, plans=seqs, imagined=imagined, costs=costs, chosen=chosen, policy=1,
                                  real_chosen=real, real_policy=sim.simulate(state, seqs[1]), seed=seed + ep, step=t,
                                  error=plan_error(imagined[chosen][-1], real[-1])))
            action = planner.act(img_t, text_ids, state_t)[0].numpy()
            img, state, _, done, info = env.step(action)
            states.append(state)
            covs.append(info["coverage"])
            t += 1
        print(f"episode {seed + ep}: {'success' if done else 'no success'} after {t} steps, best coverage {max(covs):.2f}")
        if done and steps:
            errors = np.array([c["error"] for c in steps])
            pick = steps[int(np.argmin(np.abs(errors - np.median(errors))))]
            np.savez_compressed(os.path.join(out, "planner.npz"), **pick)
            print(f"planner: step {pick['step']} of {len(steps)} checked, plan error {pick['error']:.1f} px "
                  f"(median {np.median(errors):.1f}), chosen cost {pick['costs'][pick['chosen']]:.1f} vs policy {pick['costs'][1]:.1f}")
            np.savez_compressed(os.path.join(out, "episode.npz"), states=np.stack(states), coverage=np.array(covs),
                                seed=seed + ep)
            return
    raise SystemExit("no successful planning episode in the tries")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="A seed directory with wam.pt and branched.npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=10000, help="First unseen start, as scripts/run.sh tests seed 0")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(1)

    export_branches(os.path.join(args.run_dir, "branched.npz"), args.out)
    model, ckpt = PicoWAMPolicy.from_checkpoint(os.path.join(args.run_dir, "wam.pt"), device="cpu")
    run_episodes(model, SimpleTokenizer(vocab=ckpt["vocab"]), args.out, args.seed)


if __name__ == "__main__":
    main()
