"""Aggregate a multi-seed run of scripts/run_toy.sh into one table + JSON.

Reads, for every seed directory under --root (e.g. results/seed0, results/seed1, ...):
    wam_eval.json   from scripts/evaluate.py
    wam_test.json   from scripts/test.py (WAM)
    vla_test.json   from scripts/test.py (mini-VLA baseline, optional)

and prints mean ± std across seeds of the headline numbers.

Usage:
    python -m scripts.summarize_runs --root results
"""

import argparse
import glob
import json
import os

import numpy as np


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def mean_std(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values), "values": values}


def fmt(ms, digits=4):
    if ms is None:
        return "—"
    return f"{ms['mean']:.{digits}f} ± {ms['std']:.{digits}f} (n={ms['n']})"


def at(lst, k):
    return lst[k] if lst is not None and k < len(lst) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="results")
    parser.add_argument("--output", type=str, default=None, help="Default: <root>/summary.json")
    args = parser.parse_args()

    seed_dirs = sorted(glob.glob(os.path.join(args.root, "seed*")))
    evals = [load(os.path.join(d, "wam_eval.json")) for d in seed_dirs]
    wam_tests = [load(os.path.join(d, "wam_test.json")) for d in seed_dirs]
    vla_tests = [load(os.path.join(d, "vla_test.json")) for d in seed_dirs]
    mpc_tests = [load(os.path.join(d, "wam_mpc_test.json")) for d in seed_dirs]   # Push-T: planning with the WAM

    summary = {"seeds": [os.path.basename(d) for d in seed_dirs]}

    summary["wam_success"] = mean_std([t["success_rate"] for t in wam_tests if t])
    summary["vla_success"] = mean_std([t["success_rate"] for t in vla_tests if t])
    summary["wam_steps"] = mean_std([t["mean_steps"] for t in wam_tests if t])
    summary["vla_steps"] = mean_std([t["mean_steps"] for t in vla_tests if t])
    # Push-T only: best coverage of the goal per episode, averaged
    warm_tests = [load(os.path.join(d, "wam_mpc_warm_test.json")) for d in seed_dirs]  # the old warm-start planner
    coverage = {"wam_coverage": wam_tests, "vla_coverage": vla_tests, "wam_mpc_coverage": mpc_tests,
                "wam_mpc_warm_coverage": warm_tests}
    for name in ["cem", "wide", "noprior"]:   # planner budgets (PLAN_SWEEP=1)
        coverage[f"wam_mpc_{name}_coverage"] = [load(os.path.join(d, f"wam_mpc_{name}_test.json")) for d in seed_dirs]
    for key, tests in coverage.items():
        values = mean_std([t.get("mean_max_coverage") for t in tests if t])
        if values is not None:
            summary[key] = values
    # planning: block-pose improvement (px) the model expected for its chosen plans vs the simulator's
    for name, tests in [("mpc", mpc_tests), ("mpc_warm", warm_tests)]:
        for k in ["expected_gain_px", "real_gain_px", "plan_error_px"]:
            values = mean_std([t["plan_checks"][k] for t in tests if t and "plan_checks" in t])
            if values is not None:
                summary[f"{name}_{k}"] = values

    for k in [1, 5, 10, 15]:
        summary[f"state_nmse_k{k}"] = mean_std([at(e["state_nmse"], k) for e in evals if e and e["state_nmse"]])
        summary[f"state_copy_k{k}"] = mean_std([at(e["state_nmse_copy"], k) for e in evals if e])
        summary[f"image_mse_k{k}"] = mean_std([at(e["image_mse"], k) for e in evals if e and e["image_mse"]])
        summary[f"image_copy_k{k}"] = mean_std([at(e["image_mse_copy"], k) for e in evals if e and e["image_mse_copy"]])

    for k in [1, 5, 10]:
        cfs = [e.get("counterfactual") for e in evals if e]
        summary[f"cf_wam_k{k}"] = mean_std([cf["wam_nmse"][k - 1] for cf in cfs if cf])
        summary[f"cf_noop_k{k}"] = mean_std([cf["noop_nmse"][k - 1] for cf in cfs if cf])
    # the block alone (Push-T state dims 4-7): does the model know what a push does to it?
    cfs = [e.get("counterfactual") for e in evals if e]
    if cfs and cfs[0] and len(cfs[0]["dims"]) == 8:
        summary["cf_block_wam_k10"] = mean_std([float(np.mean(cf["wam_nmse_per_dim"][9][4:8])) for cf in cfs if cf])
        summary["cf_block_noop_k10"] = mean_std([float(np.mean(cf["noop_nmse_per_dim"][9][4:8])) for cf in cfs if cf])

    # mean R^2 of the imagined-latent readout over the moving state variables
    for k in [0, 5, 15]:
        vals = []
        for e in evals:
            if e and e["state_r2"] and k < len(e["state_r2"]):
                r = [x for x in e["state_r2"][k] if x is not None]
                vals.append(float(np.mean(r)) if r else None)
        summary[f"readout_r2_k{k}"] = mean_std(vals)

    print(f"Summary over {len(seed_dirs)} seeds ({', '.join(summary['seeds'])})")
    rows = [
        ("closed-loop success, WAM", "wam_success", 3),
        ("closed-loop success, mini-VLA", "vla_success", 3),
        ("mean steps, WAM", "wam_steps", 1),
        ("mean steps, mini-VLA", "vla_steps", 1),
        ("mean max coverage, WAM", "wam_coverage", 3),
        ("mean max coverage, mini-VLA", "vla_coverage", 3),
        ("mean max coverage, WAM planning", "wam_mpc_coverage", 3),
        ("  planning, warm start", "wam_mpc_warm_coverage", 3),
        ("  plans: expected gain px", "mpc_expected_gain_px", 1), ("  plans: real gain px", "mpc_real_gain_px", 1),
        ("  plans: block error px", "mpc_plan_error_px", 1),
        ("  warm: expected gain px", "mpc_warm_expected_gain_px", 1), ("  warm: real gain px", "mpc_warm_real_gain_px", 1),
        ("  warm: block error px", "mpc_warm_plan_error_px", 1),
        ("  planning, 3 CEM rounds", "wam_mpc_cem_coverage", 3),
        ("  planning, wider noise", "wam_mpc_wide_coverage", 3),
        ("  planning, no policy prior", "wam_mpc_noprior_coverage", 3),
        ("state nMSE k=1", "state_nmse_k1", 4), ("  copy baseline k=1", "state_copy_k1", 4),
        ("state nMSE k=5", "state_nmse_k5", 4), ("  copy baseline k=5", "state_copy_k5", 4),
        ("state nMSE k=15", "state_nmse_k15", 4), ("  copy baseline k=15", "state_copy_k15", 4),
        ("image MSE k=1", "image_mse_k1", 5), ("  copy baseline k=1", "image_copy_k1", 5),
        ("image MSE k=15", "image_mse_k15", 5), ("  copy baseline k=15", "image_copy_k15", 5),
        ("counterfactual nMSE k=1", "cf_wam_k1", 4), ("  no-op baseline k=1", "cf_noop_k1", 4),
        ("counterfactual nMSE k=5", "cf_wam_k5", 4), ("  no-op baseline k=5", "cf_noop_k5", 4),
        ("counterfactual nMSE k=10", "cf_wam_k10", 4), ("  no-op baseline k=10", "cf_noop_k10", 4),
        ("  block only k=10", "cf_block_wam_k10", 4), ("  block, no-op k=10", "cf_block_noop_k10", 4),
        ("readout R2 (mean) k=0", "readout_r2_k0", 3),
        ("readout R2 (mean) k=5", "readout_r2_k5", 3),
        ("readout R2 (mean) k=15", "readout_r2_k15", 3),
    ]
    for name, key, digits in rows:
        if key.endswith(("_coverage", "_px")) and key not in summary:
            continue
        print(f"  {name:<28} {fmt(summary.get(key), digits)}")

    output = args.output or os.path.join(args.root, "summary.json")
    with open(output, "w") as f:
        json.dump(summary, f, indent=2)
    print("SUMMARY_JSON", json.dumps({k: (v["mean"] if isinstance(v, dict) else v) for k, v in summary.items()}))
    print("Saved:", output)


if __name__ == "__main__":
    main()
