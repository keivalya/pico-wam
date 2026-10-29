"""How much of Push-T can a linear model predict? A baseline for the WAM's world model.

For every seed of a run (results/<run>/seed*/pusht.npz), fits two least-squares
models of the one-step state change on that seed's training episodes, in the
same 8-dim state the WAM reads out:

    state only        s_t+1 - s_t = W s_t + b
    state + action    s_t+1 - s_t = W [s_t, a_t] + b

and scores them exactly as scripts/evaluate.py scores the WAM, on the same
unseen episodes and the same random action plans:

    factual          roll out k = 1..15 steps with the recorded actions
    counterfactual   from every 5th unseen state, 8 random action plans of
                     10 steps, checked against the simulator

Errors are normalized MSE with the training-set standard deviations, so they
are comparable with the WAM's numbers in wam_eval.json. Agent and block
errors are also reported in pixels. Two environment checks complete the
audit: two states that differ only in the agent's velocity render
identically, and simulate() matches step() exactly.

    python -m scripts.pusht_dynamics_audit --run-dir results/pusht
"""

import argparse
import glob
import json
import os

import numpy as np

from envs.pusht_env import PushTEnv, STATE_LABELS, from_state
from envs.pusht_expert import pose_error
from scripts.evaluate import make_simulator
from scripts.train import split_episodes

HORIZON, CF_HORIZON, CF_PLANS, CF_EVERY = 15, 10, 8, 5


def fit(states, actions, next_states, with_action):
    x = np.concatenate([states, actions], 1) if with_action else states
    x = np.concatenate([x, np.ones((len(x), 1))], 1)
    w, *_ = np.linalg.lstsq(x, next_states - states, rcond=None)
    return w


def step(w, s, a, with_action):
    x = np.concatenate([s, a], -1) if with_action else s
    x = np.concatenate([x, np.ones(x.shape[:-1] + (1,))], -1)
    return s + x @ w


def pixel_errors(pred, true):
    """Agent position error and block pose error (mean T-corner distance), in pixels."""
    p, t = from_state(pred), from_state(true)
    return float(np.linalg.norm(p[:2] - t[:2])), pose_error(p[4:7], t[4:7])


def audit_seed(path, seed):
    d = np.load(path, allow_pickle=True)
    ids, S, A, NS = d["episode_ids"], d["states"].astype(np.float64), d["actions"].astype(np.float64), d["next_states"].astype(np.float64)
    train_eps, val_eps = split_episodes(ids, 0.1, seed)
    tr = np.isin(ids, train_eps)
    std = np.maximum(S[tr].std(0), 1e-3)   # as PicoWAMPolicy.set_state_stats
    models = {name: fit(S[tr], A[tr], NS[tr], wa) for name, wa in [("state_only", False), ("state_action", True)]}

    out = {"train_transitions": int(tr.sum()), "val_episodes": [int(e) for e in val_eps]}

    # factual: every start, recorded actions
    fact = {name: np.zeros(HORIZON) for name in models}
    count = np.zeros(HORIZON)
    for e in val_eps:
        idx = np.nonzero(ids == e)[0]
        states = np.concatenate([S[idx], NS[idx[-1:]]])
        T = len(idx)
        for name, w in models.items():
            wa = name == "state_action"
            s = states[:T].copy()
            for k in range(HORIZON):
                n = T - k                          # starts that still have a true state k+1 steps ahead
                if n <= 0:
                    break
                s = step(w, s[:n], A[idx[k:k + n]], wa)
                err = (((s - states[k + 1:k + 1 + n]) / std) ** 2).mean(1)
                fact[name][k] += err.sum()
                if name == "state_only":
                    count[k] += n
    out["factual_nmse"] = {name: (v / np.maximum(count, 1)).tolist() for name, v in fact.items()}

    # counterfactual: same starts, plans and random generator as scripts/evaluate.py
    sim, dims, scale = make_simulator("pusht")
    rng = np.random.default_rng(0)
    cf = {name: np.zeros((CF_HORIZON, 8)) for name in models}     # per step and state variable
    noop = np.zeros((CF_HORIZON, 8))
    px = {name: {"agent": [], "block": []} for name in models}
    n_starts = 0
    for e in val_eps:
        idx = np.nonzero(ids == e)[0]
        for t in range(0, len(idx), CF_EVERY):
            plans = rng.uniform(-scale, scale, size=(CF_PLANS, CF_HORIZON, 2)).astype(np.float32)
            s0 = S[idx[t]]
            truth = np.stack([sim.simulate(s0, p) for p in plans])                       # (P, H, 8)
            still = sim.simulate(s0, np.zeros((CF_HORIZON, 2), np.float32))[None]
            noop += (((still - truth) / std) ** 2).mean(axis=0)
            for name, w in models.items():
                s = np.repeat(s0[None], CF_PLANS, 0)
                preds = []
                for k in range(CF_HORIZON):
                    s = step(w, s, plans[:, k], name == "state_action")
                    preds.append(s)
                preds = np.stack(preds, 1)
                cf[name] += (((preds - truth) / std) ** 2).mean(axis=0)
                for p in range(CF_PLANS):
                    ag, bl = pixel_errors(preds[p, -1], truth[p, -1])
                    px[name]["agent"].append(ag)
                    px[name]["block"].append(bl)
            n_starts += 1
    out["counterfactual_nmse"] = {name: (v / n_starts).mean(1).tolist() for name, v in cf.items()}
    out["counterfactual_nmse_per_dim"] = {name: (v / n_starts).tolist() for name, v in cf.items()}
    out["counterfactual_noop_nmse"] = (noop / n_starts).mean(1).tolist()
    out["counterfactual_noop_nmse_per_dim"] = (noop / n_starts).tolist()
    out["counterfactual_px_at_10"] = {name: {k: float(np.mean(v)) for k, v in d_.items()} for name, d_ in px.items()}
    out["n_counterfactual_starts"] = n_starts
    return out


def env_checks():
    env = PushTEnv(image_size=64)
    rng = np.random.default_rng(7)
    _, s, _ = env.reset(seed=7)
    a, b = s.copy(), s.copy()
    a[2:4], b[2:4] = [0.4, -0.2], [-0.4, 0.2]   # same scene, opposite agent velocities
    same_image = bool(np.array_equal(env._render(a), env._render(b)))
    act = np.array([0.3, -0.7], np.float32)
    na, nb = env.simulate(a, act[None])[0], env.simulate(b, act[None])[0]
    apart = float(np.linalg.norm(from_state(na)[:2] - from_state(nb)[:2]))

    worst = 0.0
    for i in range(20):
        _, s0, _ = env.reset(seed=1000 + i)
        acts = rng.uniform(-1.2, 1.2, size=(20, 2)).astype(np.float32)   # includes out-of-range actions
        stepped = np.stack([env.step(x)[1] for x in acts])
        worst = max(worst, float(np.abs(stepped - env.simulate(s0, acts)).max()))
    return {"opposite_velocities_render_identically": same_image,
            "next_agent_positions_apart_px": apart,
            "simulate_vs_step_max_abs_difference": worst}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="A run's results directory with seed*/pusht.npz")
    parser.add_argument("--output", default=None, help="Default: <run-dir>/dynamics_audit.json")
    args = parser.parse_args()

    result = {"state_labels": STATE_LABELS, "seeds": {}, "checks": env_checks()}
    for d in sorted(glob.glob(os.path.join(args.run_dir, "seed*"))):
        seed = int(os.path.basename(d)[4:])
        result["seeds"][seed] = audit_seed(os.path.join(d, "pusht.npz"), seed)
        r = result["seeds"][seed]
        print(f"seed {seed}: factual nMSE k=1/15  state-only {r['factual_nmse']['state_only'][0]:.4f}/{r['factual_nmse']['state_only'][-1]:.4f}"
              f"  state+action {r['factual_nmse']['state_action'][0]:.4f}/{r['factual_nmse']['state_action'][-1]:.4f}"
              f" | counterfactual nMSE k=10  state+action {r['counterfactual_nmse']['state_action'][-1]:.4f}"
              f"  no-op {r['counterfactual_noop_nmse'][-1]:.4f}"
              f" | k=10 px: agent {r['counterfactual_px_at_10']['state_action']['agent']:.1f}, block {r['counterfactual_px_at_10']['state_action']['block']:.1f}")
    print("checks:", json.dumps(result["checks"]))
    output = args.output or os.path.join(args.run_dir, "dynamics_audit.json")
    with open(output, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved:", output)


if __name__ == "__main__":
    main()
