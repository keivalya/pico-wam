"""Build the Push-T figures of the pico-WAM paper and tutorial.

Inputs (all committed): docs/results/pusht/<experiment>/, copied from the runs
of scripts/run.sh (see docs/results/pusht/README.md):

    seed*/wam_eval.json, wam_test.json, vla_test.json, wam_mpc*_test.json
    dynamics_audit.json            linear models of the dynamics (scripts/pusht_dynamics_audit.py)
    figure-data/                   model outputs (docs/figures/export_figure_data.py)
    efficiency/efficiency.json     parameters and CPU costs (docs/figures/benchmark_efficiency.py)
    supervision-pilot/summary.json the matched-supervision pilot

Outputs: fig-pusht-<name>.pdf (paper) and .svg (tutorial) next to this script,
at the paper's text width. Every plotted value is read from those files.

    python docs/figures/pusht_figures.py
"""

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import figstyle as fs  # noqa: E402

fs.use_style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = os.environ.get("PUSHT_RESULTS", os.path.join(HERE, "..", "results", "pusht"))
W = fs.TEXT

WAM = fs.PALETTE["red"]
VLA = fs.PALETTE["blue"]
PLAN = fs.PALETTE["green"]
LIN = fs.PALETTE["purple"]
DREAM = fs.PALETTE["purple"]
TRUE = "#000000"
REF = fs.BASELINE

AGENT, BLOCK = [0, 1, 2, 3], [4, 5, 6, 7]
NAMES = {"agent_x": "agent $x$", "agent_y": "agent $y$", "agent_vx": "agent $v_x$", "agent_vy": "agent $v_y$",
         "block_x": "block $x$", "block_y": "block $y$", "block_cos": r"block $\cos\theta$", "block_sin": r"block $\sin\theta$"}


# ---------------------------------------------------------------- loading


def load_runs(name):
    """Per-seed dicts of every JSON in docs/results/pusht/<name>/seed*/ (file stem -> content)."""
    seeds = []
    for d in sorted(glob.glob(os.path.join(RESULTS, name, "seed[0-9]*"))):
        seeds.append({os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))})
    return seeds


def load_json(*parts):
    path = os.path.join(RESULTS, *parts)
    return json.load(open(path)) if os.path.exists(path) else None


def per_seed(seeds, getter):
    return np.asarray([float(getter(s)) for s in seeds])


def curves(seeds, getter):
    rows = [np.asarray(getter(s), dtype=float) for s in seeds]
    n = min(len(r) for r in rows)
    return np.stack([r[:n] for r in rows])


def coverage(seeds, key):
    return per_seed(seeds, lambda s: 100 * s[key]["mean_max_coverage"])


def band(ax, x, runs, color, label, ls="-", marker="o"):
    m, s = runs.mean(0), runs.std(0)
    ax.plot(x, m, color=color, ls=ls, marker=marker, ms=2.2, lw=1.1, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)


def dots(ax, x, vals, color, marker="o", label=None, off=0.0, ms=3.5):
    """Mean ± SD with the individual seeds as small dots."""
    ax.errorbar([x + off], [vals.mean()], yerr=[vals.std()], color=color, marker=marker, ms=ms, capsize=2,
                lw=1.0, ls="none", label=label)
    ax.scatter(np.full(len(vals), x + off), vals, s=5, color=color, alpha=0.5, lw=0, zorder=3)


def finish(fig, name):
    if os.environ.get("PREVIEW_DIR"):
        fig.savefig(os.path.join(os.environ["PREVIEW_DIR"], name + ".png"), dpi=220)
    fs.save(fig, os.path.join(HERE, name))


# ------------------------------------------------------------------- task


def fig_task():
    """(a) frames of an unseen human demonstration, (b) a scripted episode with random bursts."""
    from envs.pusht_env import PushTEnv

    strip = np.load(os.path.join(RESULTS, "baseline", "figure-data", "episode_strip.npz"))
    picks = [0, 2, 4, 7]

    env = PushTEnv(seed=3, image_size=96)
    env.reset(seed=3)
    rng = np.random.default_rng(3)
    acts, expert, burst = [], [], 0
    for _ in range(80):
        if burst == 0 and rng.random() < 0.3:
            burst = int(rng.integers(1, 4))
        if burst > 0:
            a = rng.uniform(-1, 1, 2)
            burst -= 1
            expert.append(False)
        else:
            a = env.expert_action(noise=0.05)
            expert.append(True)
        acts.append(np.clip(a, -1, 1))
        if env.step(a)[3]:
            break
    acts = np.asarray(acts)

    fig = plt.figure(figsize=(W, 0.31 * W), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.5])
    sub = gs[0].subgridspec(1, len(picks), wspace=0.04)
    first = None
    for c, k in enumerate(picks):
        a = fig.add_subplot(sub[0, c])
        a.imshow(strip["real"][k], interpolation="nearest")
        a.set_axis_off()
        a.text(0.5, 1.03, f"$t={strip['timesteps'][k]}$", transform=a.transAxes, ha="center", va="bottom", fontsize=6.5)
        first = first or a

    ax = fig.add_subplot(gs[1])
    t = np.arange(len(acts))
    for i, e in enumerate(expert):
        if not e:
            ax.axvspan(i - 0.5, i + 0.5, color=WAM, alpha=0.18, lw=0)
    ax.plot(t, acts[:, 0], color=TRUE, lw=1.0)
    ax.plot(t, acts[:, 1], color=VLA, lw=1.0, ls="--")
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("timestep")
    ax.set_ylabel("action, 100 px units")
    ax.legend([Line2D([], [], color=TRUE, lw=1.0), Line2D([], [], color=VLA, lw=1.0, ls="--"),
               plt.Rectangle((0, 0), 1, 1, color=WAM, alpha=0.18)], ["$a_x$", "$a_y$", "random burst"],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.5)
    fs.panel_labels([first, ax], pad=7.0)
    finish(fig, "fig-pusht-task")


# ------------------------------------------------------------- prediction


def fig_prediction(base, audit):
    """World model vs. copying, the no-op world and a linear model, factual and counterfactual."""
    lin_f = np.stack([np.asarray(audit["seeds"][s]["factual_nmse"]["state_action"]) for s in sorted(audit["seeds"])])
    lin_c = np.stack([np.asarray(audit["seeds"][s]["counterfactual_nmse"]["state_action"]) for s in sorted(audit["seeds"])])
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.42)

    ax = axes[0]
    k = np.arange(1, 16)
    band(ax, k, curves(base, lambda s: s["wam_eval"]["state_nmse_copy"])[:, 1:16], REF, '"nothing changes"', ls="--", marker="s")
    band(ax, k, lin_f[:, :15], LIN, "linear model, state + action", ls=":", marker="^")
    band(ax, k, curves(base, lambda s: s["wam_eval"]["state_nmse"])[:, 1:16], WAM, "pico-WAM")
    ax.set_yscale("log")
    ax.set_xlabel("steps $k$, recorded actions")
    ax.set_ylabel("state error, nMSE")

    ax = axes[1]
    k = np.arange(1, 11)
    band(ax, k, curves(base, lambda s: s["wam_eval"]["counterfactual"]["noop_nmse"]), REF, "true world, zero action", ls="--", marker="s")
    band(ax, k, lin_c, LIN, "linear model, state + action", ls=":", marker="^")
    band(ax, k, curves(base, lambda s: s["wam_eval"]["counterfactual"]["wam_nmse"]), WAM, "pico-WAM")
    ax.set_yscale("log")
    ax.set_xlabel("steps $k$, random actions")
    ax.set_xticks([2, 4, 6, 8, 10])
    ax.set_ylabel("counterfactual error, nMSE")

    ax = axes[2]
    k = np.arange(1, 16)
    band(ax, k, curves(base, lambda s: s["wam_eval"]["image_mse_copy"])[:, 1:16], REF, "copy current frame", ls="--", marker="s")
    band(ax, k, curves(base, lambda s: s["wam_eval"]["image_mse"])[:, 1:16], WAM, "pico-WAM, decoded")
    ax.set_xlabel("steps $k$")
    ax.set_ylabel("image error, pixel MSE")

    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-prediction")


# --------------------------------------------------- data and counterfactuals


DATA = [("baseline", "human\nLeRobot"), ("expert-p0", "scripted\nno bursts"), ("expert-p0.1", "scripted\nbursts 0.1"),
        ("expert-p0.3", "scripted\nbursts 0.3"), ("mixed-p0.3", "human +\nbursts 0.3")]


def block_cf(seed, k=9, key="wam_nmse_per_dim"):
    return float(np.mean([seed["wam_eval"]["counterfactual"][key][k][j] for j in BLOCK]))


def fig_data(exps):
    """Which data makes the world model right about the block, and the policy and planner good?"""
    present = [(n, l) for n, l in DATA if n in exps]
    x = np.arange(len(present))
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.42)

    ax = axes[0]
    for i, (name, _) in enumerate(present):
        dots(ax, i, per_seed(exps[name], lambda s: block_cf(s, key="noop_nmse_per_dim")), REF, marker="s", off=-0.15,
             label="true world, zero action" if i == 0 else None)
        dots(ax, i, per_seed(exps[name], block_cf), WAM, off=0.0, label="pico-WAM, random actions" if i == 0 else None)
        dots(ax, i, per_seed(exps[name], lambda s: np.mean([s["wam_eval"]["state_nmse_per_dim"][10][j] for j in BLOCK])),
             fs.PALETTE["orange"], marker="^", off=0.15, label="pico-WAM, recorded actions" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in present], fontsize=6)
    ax.set_yscale("log")
    ax.set_yticks([0.01, 0.02, 0.05], ["0.01", "0.02", "0.05"])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylabel("block error at 10 steps, nMSE")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=1, fontsize=6)

    ax = axes[1]
    for i, (name, _) in enumerate(present):
        s = exps[name]
        dots(ax, i, coverage(s, "vla_test"), VLA, off=-0.18, label="mini-VLA" if i == 0 else None)
        dots(ax, i, coverage(s, "wam_test"), WAM, off=0.0, label="pico-WAM, reactive" if i == 0 else None)
        if all("wam_mpc_test" in r for r in s):
            dots(ax, i, coverage(s, "wam_mpc_test"), PLAN, marker="D", off=0.18, label="pico-WAM, planning" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in present], fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("best coverage of the target, %")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.36), ncol=3, fontsize=6)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-data")


# --------------------------------------------------------------- planning


PLANNERS = [("wam_test", "policy\nonly"), ("wam_mpc_test", "plan\ndefault"), ("wam_mpc_wide_test", "plan\nwider noise"),
            ("wam_mpc_cem_test", "plan\n3 rounds"), ("wam_mpc_noprior_test", "plan\nno prior")]


def fig_planning(budget, base):
    """More search finds the model's errors: coverage and expected vs. real improvement per planner."""
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.42)
    x = np.arange(len(PLANNERS))

    ax = axes[0]
    vla = coverage(budget, "vla_test")
    ax.axhspan(vla.mean() - vla.std(), vla.mean() + vla.std(), color=VLA, alpha=0.12, lw=0)
    ax.axhline(vla.mean(), color=VLA, lw=0.8, ls="--", label="mini-VLA, mean ± SD")
    for i, (key, _) in enumerate(PLANNERS):
        dots(ax, i, coverage(budget, key), WAM if key == "wam_test" else PLAN, marker="o" if key == "wam_test" else "D")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in PLANNERS], fontsize=6)
    ax.set_ylim(0, 100)
    ax.set_ylabel("best coverage of the target, %")
    ax.legend(loc="upper left", fontsize=6)

    ax = axes[1]
    for i, (key, _) in enumerate(PLANNERS[1:], start=1):
        exp = per_seed(budget, lambda s: s[key]["plan_checks"]["expected_gain_px"])
        real = per_seed(budget, lambda s: s[key]["plan_checks"]["real_gain_px"])
        dots(ax, i, exp, PLAN, marker="D", off=-0.12, label="expected by the model" if i == 1 else None)
        dots(ax, i, real, TRUE, marker="o", off=0.12, label="real, in the simulator" if i == 1 else None)
    ax.axhline(0, color=REF, lw=0.6)
    ax.set_xticks(x[1:])
    ax.set_xticklabels([l for _, l in PLANNERS[1:]], fontsize=6)
    ax.set_ylabel("improvement over 12 steps, px")
    ax.legend(loc="upper left", fontsize=6)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-planning")


# --------------------------------------------------------------- ablations


# two groups of one-change variants: on the study base, and on the action-chunk model
VARIANTS = [("baseline", "full objective, $H=5$"), ("no-latent", "no latent consistency"), ("no-image", "no image loss"),
            ("horizon-1", "horizon $H=1$"), ("horizon-10", "horizon $H=10$"), ("pixel-input", "pixel input"),
            ("demos-100", "100 demonstrations"), ("demos-50", "50 demonstrations"), ("wider-model", "latent width 256"),
            ("epochs-100", "100 epochs"), ("diffusion-head", "diffusion head")]
CHUNK_VARIANTS = [("action-chunks", "action chunks, $K=8$"), ("chunks-k16", "chunks $K=16$, execute 8"),
                  ("chunks-temporal-ensembling", "average overlapping chunks"), ("chunks-dmodel-64", "latent width 64"),
                  ("chunks-plain-cnn", "plain CNN encoder"), ("chunks-no-decoder", "no image decoder")]


def fig_ablations(exps):
    groups = [[(n, l) for n, l in g if n in exps] for g in (VARIANTS, CHUNK_VARIANTS)]
    present = groups[0] + groups[1]
    refs = {n: g[0][0] for g in groups for n, _ in g}   # each row's reference: the first row of its group
    metrics = [
        ("state nMSE, $k=15$", lambda s: s["wam_eval"]["state_nmse"][15], WAM),
        ("block counterfactual\nnMSE, $k=10$", block_cf, WAM),
        ("best coverage, %", None, None),
    ]
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.64, sharey=True)
    n0, n1 = len(groups[0]), len(groups[1])
    y = np.concatenate([n1 + 0.6 + np.arange(n0)[::-1], np.arange(n1)[::-1]])   # a gap between the groups
    for ax, (label, getter, color) in zip(axes, metrics):
        ax.axhline(n1 - 0.2, color=REF, lw=0.5)
        for yi, (name, _) in zip(y, present):
            s = exps[name]
            if getter is not None:
                v = per_seed(s, getter)
                c = REF if refs[name] == name else color
                ax.errorbar(v.mean(), yi, xerr=v.std(), color=c, marker="o", ms=3, capsize=2, lw=1.0)
                ax.scatter(v, np.full(len(v), yi), s=5, color=c, alpha=0.5, lw=0, zorder=3)
            else:
                for key, c, m, off in [("vla_test", VLA, "s", 0.18), ("wam_test", WAM, "o", 0.0), ("wam_mpc_test", PLAN, "D", -0.18)]:
                    if all(key in r for r in s):
                        v = coverage(s, key)
                        ax.errorbar(v.mean(), yi + off, xerr=v.std(), color=c, marker=m, ms=2.8, capsize=1.5, lw=0.9)
        if getter is not None:   # each group's reference value, over that group's rows only
            for g, (lo, hi) in zip(groups, [(n1 - 0.2, y[0] + 0.5), (-0.5, n1 - 0.2)]):
                ax.plot([per_seed(exps[g[0][0]], getter).mean()] * 2, [lo, hi], color=REF, lw=0.6, ls=":")
        if getter is not None:
            ax.set_xscale("log")   # the pixel-input row is several times worse than the rest
            ticks = [0.01, 0.03, 0.1]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticklabels([l for _, l in present], fontsize=6.5)
    axes[0].set_ylabel("training variant")
    axes[2].set_xlim(0, 100)
    axes[2].legend([Line2D([], [], color=VLA, marker="s", ls="none", ms=3), Line2D([], [], color=WAM, marker="o", ls="none", ms=3),
                    Line2D([], [], color=PLAN, marker="D", ls="none", ms=3)], ["mini-VLA", "WAM, reactive", "WAM, planning"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=5.5)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-ablations")


# ------------------------------------------------- what makes Push-T succeed


RECIPE = [("baseline", "study base,\none action per step"), ("action-chunks", "action chunks, $K=8$"),
          ("chunks-epochs-100", "chunks, 100 epochs"), ("chunks-branched", "chunks, branched data"),
          ("chunks-restart-planner", "chunks, restarting planner"),
          ("branched-restart-planner", "chunks, branched data,\nrestarting planner"),
          ("branched-restart-100-epochs", "the same, 100 epochs"),
          ("head-3x512", "+ action head 3×512"), ("final", "+ image loss on moving pixels")]
POLICIES = [("vla_test", VLA, "s", 0.22, "mini-VLA"), ("wam_test", WAM, "o", 0.0, "WAM, reactive"),
            ("wam_mpc_test", PLAN, "D", -0.22, "WAM, planning")]


def fig_recipe(exps):
    """The changes that make Push-T succeed: coverage, successes and the block's counterfactual error."""
    present = [(n, l) for n, l in RECIPE if n in exps]
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.5, sharey=True)
    y = np.arange(len(present))[::-1]
    for yi, (name, _) in zip(y, present):
        s = exps[name]
        for key, c, m, off, _ in POLICIES:
            for ax, v in [(axes[0], coverage(s, key)), (axes[1], per_seed(s, lambda r: 100 * r[key]["success_rate"]))]:
                ax.errorbar(v.mean(), yi + off, xerr=v.std(), color=c, marker=m, ms=2.8, capsize=1.5, lw=0.9)
                ax.scatter(v, np.full(len(v), yi + off), s=4, color=c, alpha=0.45, lw=0, zorder=3)
        v = per_seed(s, block_cf)
        axes[2].errorbar(v.mean(), yi, xerr=v.std(), color=WAM, marker="o", ms=3, capsize=2, lw=1.0)
        axes[2].scatter(v, np.full(len(v), yi), s=5, color=WAM, alpha=0.5, lw=0, zorder=3)
    noop = per_seed(exps["baseline"], lambda r: block_cf(r, key="noop_nmse_per_dim")).mean()
    axes[2].axvline(noop, color=REF, lw=0.8, ls="--")
    axes[2].text(noop, y[0] + 0.55, "block stays put", color=REF, fontsize=5.5, ha="left", va="bottom")
    axes[0].set_xlim(0, 100)
    top = max(100 * r[k]["success_rate"] for n, _ in present for r in exps[n] for k, *_ in POLICIES)
    axes[1].set_xlim(-2, max(30, 1.15 * top))
    axes[2].set_xlim(0, None)
    for ax, label in zip(axes, ["best coverage, %", "success rate, %", "block counterfactual\nnMSE, $k=10$"]):
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticklabels([l for _, l in present], fontsize=6.5)
    axes[0].set_ylabel("configuration")
    axes[0].set_ylim(y[-1] - 0.6, y[0] + 1.0)
    axes[1].legend([Line2D([], [], color=c, marker=m, ls="none", ms=3) for _, c, m, _, _ in POLICIES],
                   [l for *_, l in POLICIES], loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=5.5)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-recipe")


# --------------------------------------------------------------- pilot


def fig_pilot(pilot):
    conds = [("bc", "actions\nonly"), ("current", "+ current\nstate"), ("future", "+ future\nstate")]
    fig, ax = fs.figure(width=fs.COLUMN, ratio=0.8)
    x = np.arange(len(conds))
    for r in pilot["coverage_rows"]:
        ax.plot(x, [100 * r[c] for c, _ in conds], color=REF, lw=0.6, alpha=0.7, marker="o", ms=2)
    means = [100 * pilot["coverage_summary"][c]["mean"] for c, _ in conds]
    sds = [100 * pilot["coverage_summary"][c]["sample_sd"] for c, _ in conds]
    ax.errorbar(x, means, yerr=sds, color=WAM, marker="s", ms=3.5, capsize=2, lw=1.4, label="mean ± SD")
    ax.set_xticks(x)
    ax.set_xticklabels([l for _, l in conds], fontsize=6.5)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel("best coverage from pixels, %")
    ax.legend(loc="lower right", fontsize=6)
    finish(fig, "fig-pusht-pilot")


# ----------------------------------------------------- qualitative figures


def figdata(name, run="baseline"):
    return np.load(os.path.join(RESULTS, run, "figure-data", name + ".npz"))


def fig_episode_strip():
    d = figdata("episode_strip")
    rows = [("real", d["real"]), ("decoded $z_t$", d["recon"]),
            ("imagined,\nreal actions", d["imagined"]), ("dreamed,\nown policy", d["dreamed"])]
    t = d["timesteps"]
    fig, axes = plt.subplots(len(rows), len(t), figsize=(W, 0.60 * W * len(rows) / len(t) + 0.3),
                             layout="constrained", gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    for r, (name, frames) in enumerate(rows):
        for c in range(len(t)):
            ax = axes[r, c]
            ax.imshow(frames[c], interpolation="nearest")
            ax.set_axis_off()
            if r == 0:
                ax.text(0.5, 1.06, f"$t={t[c]}$", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
            if c == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, ha="right", va="center", fontsize=7)
    finish(fig, "fig-pusht-episode-strip")


def fig_beliefs():
    d = figdata("episode_beliefs")
    labels = [str(l) for l in d["labels"]]
    show = ["agent_x", "agent_y", "block_x", "block_y"]
    fig, axes = fs.figure_grid(1, 4, width=W, ratio=0.34, sharex=True)
    t = np.arange(len(d["true"]))
    for ax, name in zip(axes, show):
        j = labels.index(name)
        ax.plot(t, d["true"][:, j], color=TRUE, lw=1.4, label="true state")
        ax.plot(t, d["now"][:, j], color=VLA, lw=1.0, label="read from encoded frame")
        ax.plot(t, d["imagined"][:, j], color=WAM, lw=1.0, ls="--", label="imagined from $t=0$, real actions")
        ax.plot(t, d["dreamed"][:, j], color=DREAM, lw=1.0, ls=":", label="dreamed from $t=0$, own policy")
        ax.set_ylabel(NAMES[name])
        ax.set_xlabel("timestep")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=2)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-beliefs")


def fig_what_if():
    d = figdata("what_if")
    names = [str(n) for n in d["names"]]
    fig = plt.figure(figsize=(W, 0.40 * W), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.25])

    ax = fig.add_subplot(gs[0])
    ax.imshow(d["start_image"], extent=[0, 1, 0, 1], alpha=0.35, interpolation="nearest")
    cmap = plt.get_cmap("twilight", 10)
    for i in range(len(names)):
        color = TRUE if i == 0 else cmap(i)
        ax.plot(d["true"][i, :, 0], d["true"][i, :, 1], color=color, lw=0.8, alpha=0.75)
        ax.plot(d["imagined"][i, :, 0], d["imagined"][i, :, 1], color=color, lw=1.3, ls="--")
        ax.scatter(d["imagined"][i, -1, 0], d["imagined"][i, -1, 1], color=color, s=7, zorder=4, lw=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("agent $x$")
    ax.set_ylabel("agent $y$")
    ax.legend([Line2D([], [], color="0.3", ls="--", lw=1.3), Line2D([], [], color="0.3", lw=0.8)],
              ["imagined", "true simulator"], loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6)

    sub = gs[1].subgridspec(3, 3, wspace=0.04, hspace=0.04)
    order = {(0, 0): 4, (0, 1): 3, (0, 2): 2, (1, 0): 5, (1, 1): 0, (1, 2): 1, (2, 0): 6, (2, 1): 7, (2, 2): 8}
    frame_axes = []
    for (r, c), i in order.items():
        a = fig.add_subplot(sub[r, c])
        a.imshow(d["final_frames"][i], interpolation="nearest")
        a.set_axis_off()
        a.text(0.05, 0.95, names[i], transform=a.transAxes, ha="left", va="top", fontsize=5,
               bbox={"boxstyle": "square,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})
        frame_axes.append(a)

    # (c) endpoint error averaged over the 8 push directions, per seed
    ax_c = fig.add_subplot(gs[2])
    for i, (key, ref, label) in enumerate([("endpoint_error", "ignore_push_error", "agent"),
                                           ("block_endpoint_error", "block_ignore_push_error", "block")]):
        err, ign = d[key][:, 1:].mean(1), d[ref][:, 1:].mean(1)
        for off, v, color in [(-0.18, ign, fs.MUTED), (0.18, err, WAM)]:
            ax_c.bar(i + off, v.mean(), 0.34, color=color, edgecolor=REF if color == fs.MUTED else color, lw=0.5)
            ax_c.scatter(np.full(len(v), i + off), v, s=6, color="black", zorder=3, lw=0)
    ax_c.set_xticks([0, 1])
    ax_c.set_xticklabels(["agent", "block"])
    ax_c.set_ylabel("endpoint error, unit square")
    ax_c.legend([plt.Rectangle((0, 0), 1, 1, fc=fs.MUTED, ec=REF, lw=0.5), plt.Rectangle((0, 0), 1, 1, fc=WAM)],
                ["ignore the push", "pico-WAM"], loc="upper right", fontsize=6)

    fs.panel_labels([ax, frame_axes[0], ax_c], pad=4.0)
    finish(fig, "fig-pusht-what-if")


def fig_latent_map():
    d = figdata("latent_map")
    short = {"agent_x": "$x$", "agent_y": "$y$", "agent_vx": "$v_x$", "agent_vy": "$v_y$",
             "block_x": "$b_x$", "block_y": "$b_y$", "block_cos": r"$\cos\theta$", "block_sin": r"$\sin\theta$"}
    labels = [short[str(l)] for l in d["labels"]]
    ev = d["explained"]
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.48, gridspec_kw={"width_ratios": [1.0, 1.05]})

    ax = axes[0]
    sc = ax.scatter(d["xy"][:, 0], d["xy"][:, 1], c=d["progress"], cmap=fs.SEQUENTIAL, s=2, alpha=0.6, lw=0, rasterized=True)
    ax.plot(d["real_traj"][:, 0], d["real_traj"][:, 1], color=TRUE, lw=1.2, label="encoded real frames")
    ax.plot(d["imagined_traj"][:, 0], d["imagined_traj"][:, 1], color=WAM, lw=1.2, ls="--", label="imagined from $t=0$")
    ax.set_xlabel(f"PC1, {ev[0]:.0%} of variance")
    ax.set_ylabel(f"PC2, {ev[1]:.0%} of variance")
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
    fig.colorbar(sc, ax=ax, label="episode progress", fraction=0.05, pad=0.02)

    ax = axes[1]
    im = ax.imshow(d["corr"], cmap=fs.DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("state variable")
    ax.set_yticks(range(len(ev)))
    ax.set_yticklabels([f"PC{i + 1}, {e:.0%}" for i, e in enumerate(ev)])
    ax.grid(False)
    for i in range(d["corr"].shape[0]):
        for j in range(d["corr"].shape[1]):
            v = d["corr"][i, j]
            if abs(v) >= 0.5:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.5, color="white" if abs(v) > 0.75 else "black")
    fig.colorbar(im, ax=ax, label="Pearson correlation", fraction=0.05, pad=0.02)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-latent-map")


def fig_keypoints():
    runs = [("baseline", "state +\nimage input"), ("pixel-input", "image\ninput only")]
    runs = [(n, t) for n, t in runs if os.path.exists(os.path.join(RESULTS, n, "figure-data", "keypoints_run.npz"))]
    fig = plt.figure(figsize=(W, max(0.12 * W * len(runs) + 0.35, 1.25)), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[4.0, 1.0])
    strips = gs[0].subgridspec(len(runs), 8, wspace=0.03, hspace=0.03)
    cmap = plt.get_cmap("tab20")
    for r, (name, title) in enumerate(runs):
        d = figdata("keypoints_run", name)
        size = d["frames"].shape[1]
        for c in range(d["frames"].shape[0]):
            a = fig.add_subplot(strips[r, c])
            a.imshow(d["frames"][c], interpolation="nearest")
            kp = d["keypoints"][c]
            a.scatter((kp[:, 0] + 1) / 2 * (size - 1), (kp[:, 1] + 1) / 2 * (size - 1),
                      c=[cmap(i % 20) for i in range(kp.shape[0])], s=7, marker="x", lw=0.7)
            a.set_axis_off()
            if r == 0:
                a.text(0.5, 1.04, f"$t={d['timesteps'][c]}$", transform=a.transAxes, ha="center", va="bottom", fontsize=6)
            if c == 0:
                a.text(-0.1, 0.5, title, transform=a.transAxes, ha="right", va="center", fontsize=6.5)
                a.text(-0.1, 1.04, "ab"[r], transform=a.transAxes, ha="right", va="bottom", fontsize=7, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    for i, (name, _) in enumerate(runs):
        stats = json.load(open(os.path.join(RESULTS, name, "figure-data", "keypoint_stats.json")))["run"]
        for off, key, color in [(-0.18, "tracking_r2", WAM), (0.18, "block_tracking_r2", fs.PALETTE["orange"])]:
            v = np.asarray(stats[key])
            ax.bar(i + off, v.mean(), 0.34, color=color, alpha=0.85, label=("agent" if key == "tracking_r2" else "block") if i == 0 else None)
            ax.scatter(np.full(len(v), i + off), v, s=6, color="black", zorder=3, lw=0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels(["state +\nimage", "image\nonly"][: len(runs)], fontsize=6.5)
    ax.set_ylabel("keypoint tracking, $R^2$")
    ax.legend(loc="lower right", fontsize=5.5)
    fs.panel_labels([ax], labels=["c"])
    finish(fig, "fig-pusht-keypoints")


def fig_efficiency():
    d = load_json("final", "efficiency-laptop", "efficiency.json") or load_json("baseline", "efficiency", "efficiency.json")
    P = d["parameters"]
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.42, gridspec_kw={"width_ratios": [1.25, 0.85, 1.0]})
    ax = axes[0]
    groups = [("encoders + fusion", ["img_encoder", "txt_encoder", "state_encoder", "fusion"], "#BBBBBB"),
              ("action head", ["policy_head"], VLA), ("dynamics + readout", ["dynamics", "state_readout"], WAM),
              ("image decoder", ["image_decoder"], fs.PALETTE["orange"])]
    models = [("mini-VLA", P["vla"]), ("WAM,\nstates only", {k: v for k, v in P["wam"].items() if k != "image_decoder"}), ("WAM", P["wam"])]
    y = np.arange(len(models))[::-1]
    for yi, (name, parts) in zip(y, models):
        left = 0.0
        for gname, keys, color in groups:
            w = sum(parts.get(k, 0) for k in keys) / 1e3
            ax.barh(yi, w, left=left, color=color, height=0.6, label=gname if yi == y[-1] else None)
            left += w
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in models], fontsize=6.5)
    ax.set_xlabel("parameters, thousands")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)

    ax = axes[1]
    T = d["train_step_ms"]
    names = [("mini-VLA", "vla", VLA), ("WAM,\nstates", "wam_state_only", WAM), ("WAM", "wam", fs.PALETTE["orange"])]
    ax.bar(range(3), [T[k] for _, k, _ in names], color=[c for _, _, c in names], width=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(3))
    ax.set_xticklabels([n for n, _, _ in names], fontsize=6.5)
    ax.set_ylabel("training step, ms")

    ax = axes[2]
    L = d["latency_ms"]
    ks = [1, 5, 10, 20]
    ax.plot(ks, [L[f"imagine_{k}_states"] for k in ks], color=WAM, marker="o", ms=3, label="latent + state readout")
    ax.plot(ks, [L[f"imagine_{k}_states_and_images"] for k in ks], color=fs.PALETTE["orange"], marker="s", ms=3, label="+ decode every image")
    ax.axhline(L["act_wam"], color=REF, ls="--", lw=0.8, label="one policy action")
    ax.set_yscale("log")
    ax.set_xticks(ks)
    ax.set_xlabel("imagined steps $k$")
    ax.set_ylabel("latency, ms")
    handles, names = [], []
    for a in (axes[0], axes[2]):   # one legend for the whole figure, below it
        h, n = a.get_legend_handles_labels()
        handles += h
        names += n
    fig.legend(handles, names, loc="outside lower center", ncol=4, fontsize=5.5)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-efficiency")


# ------------------------------------------------------------ model size


CAPACITY = [("branched-restart-100-epochs", "head 1×128"), ("head-3x512", "head 3×512"), ("head-3x1024", "head 3×1024"),
            ("head-6x512", "head 6×512"), ("flow-head-3x512", "flow head 3×512"), ("latent-256", "latent 256, head 1×128"),
            ("head-3x512-latent-256", "latent 256, head 3×512"), ("head-3x512-no-decoder", "head 3×512, no decoder")]


def fig_capacity(exps):
    """How big should pico-WAM be? Policy-head and latent sizes against coverage, successes and parameters."""
    present = [(n, l) for n, l in CAPACITY if n in exps]
    fig, axes = fs.figure_grid(1, 3, width=W, ratio=0.46, sharey=True, gridspec_kw={"width_ratios": [1.0, 1.0, 0.9]})
    y = np.arange(len(present))[::-1]
    for yi, (name, _) in zip(y, present):
        s = exps[name]
        for key, c, m, off, _ in POLICIES:
            for ax, v in [(axes[0], coverage(s, key)), (axes[1], per_seed(s, lambda r: 100 * r[key]["success_rate"]))]:
                ax.errorbar(v.mean(), yi + off, xerr=v.std(), color=c, marker=m, ms=2.8, capsize=1.5, lw=0.9)
                ax.scatter(v, np.full(len(v), yi + off), s=4, color=c, alpha=0.45, lw=0, zorder=3)
        eff = load_json(name, "efficiency", "efficiency.json")
        P = eff["parameters"]
        dec = P["wam"].get("image_decoder", 0) / 1e6
        head = P["wam"]["policy_head"] / 1e6
        rest = P["wam_total"] / 1e6 - dec - head
        for left, w, color in [(0, rest, "#BBBBBB"), (rest, head, VLA), (rest + head, dec, fs.PALETTE["orange"])]:
            axes[2].barh(yi, w, left=left, color=color, height=0.55)
    axes[0].axhspan(y[1] - 0.5, y[1] + 0.5, color=fs.MUTED, alpha=0.35, lw=0, zorder=0)   # the default
    for ax in axes[:2]:
        ax.axhspan(y[1] - 0.5, y[1] + 0.5, color=fs.MUTED, alpha=0.35, lw=0, zorder=0)
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(-2, 60)
    for ax, label in zip(axes, ["best coverage, %", "success rate, %", "WAM parameters, M"]):
        ax.set_xlabel(label)
        ax.set_yticks(y)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    axes[0].set_yticklabels([l for _, l in present], fontsize=6.5)
    axes[0].set_ylabel("model size")
    axes[1].legend([Line2D([], [], color=c, marker=m, ls="none", ms=3) for _, c, m, _, _ in POLICIES],
                   [l for *_, l in POLICIES], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=5.5)
    axes[2].legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#BBBBBB", VLA, fs.PALETTE["orange"]]],
                   ["encoders, dynamics", "action head", "image decoder"], loc="upper center", bbox_to_anchor=(0.5, -0.22),
                   ncol=1, fontsize=5.5)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-capacity")


def fig_pixels(exps):
    """With pixels only, the world model's training signal is what makes the policy work."""
    groups = [("head-3x512", "state + image\ninput"), ("pixel-input-head-3x512", "image input\nonly")]
    groups = [(n, l) for n, l in groups if n in exps]
    fig, axes = fs.figure_grid(1, 2, width=W, ratio=0.36)
    for ax, getter, label in [(axes[0], lambda s, k: coverage(s, k), "best coverage, %"),
                              (axes[1], lambda s, k: per_seed(s, lambda r: 100 * r[k]["success_rate"]), "success rate, %")]:
        for gi, (name, _) in enumerate(groups):
            for (key, c, m, _, lab), off in zip(POLICIES, [-0.26, 0.0, 0.26]):
                v = getter(exps[name], key)
                ax.bar(gi + off, v.mean(), 0.24, color=c, alpha=0.9, label=lab if gi == 0 else None)
                ax.scatter(np.full(len(v), gi + off), v, s=6, color="black", zorder=3, lw=0)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([l for _, l in groups], fontsize=6.5)
        ax.set_ylabel(label)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 100)
    axes[1].legend(loc="upper right", fontsize=5.5)
    fs.panel_labels(axes)
    finish(fig, "fig-pusht-pixels")


# ------------------------------------------------------------ dreams and the decoder


DREAMS = [("head-3x512-no-decoder", "no decoder, $H=5$", fs.PALETTE["blue"], ":"),
          ("head-3x512-no-decoder-horizon-8", "no decoder, $H=8$", fs.PALETTE["cyan"], ":"),
          ("head-3x512-no-decoder-horizon-15", "no decoder, $H=15$", fs.PALETTE["purple"], ":"),
          ("head-3x512", "decoder, $H=5$", fs.PALETTE["blue"], "-"),
          ("horizon-15", "decoder, $H=15$", fs.PALETTE["purple"], "-"),
          ("final", "decoder, $H=5$, moving pixels", fs.PALETTE["green"], "-")]


def dream_rows(name):
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULTS, name, "seed[0-9]*"))):
        e = json.load(open(os.path.join(d, "wam_eval.json")))
        extra = os.path.join(d, "wam_eval_dreams.json")
        if "dream" not in e and os.path.exists(extra):
            e = {**e, **json.load(open(extra))}
        rows.append(e)
    return rows


def fig_dreams():
    """How long a dream stays right, and what the moving-pixel loss does to the decoder."""
    fig = plt.figure(figsize=(W, 0.5 * W), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.25])
    ax = fig.add_subplot(gs[0])
    for name, label, color, ls in DREAMS:
        rows = dream_rows(name)
        if not rows or any("dream" not in r for r in rows):
            continue
        err = np.stack([r["dream"]["block_px"] for r in rows])
        k = np.arange(1, err.shape[1] + 1)
        ax.plot(k, err.mean(0), color=color, ls=ls, lw=1.2, label=label)
    ax.axvline(12, color=REF, lw=0.6, ls="--")
    ax.text(12.5, 1.0, "planner horizon", color=REF, fontsize=5.5, rotation=90, va="bottom", transform=ax.get_xaxis_transform())
    ax.set_xlabel("steps dreamed ahead")
    ax.set_ylabel("T error against the simulator, px")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=5.5)

    picks = [2, 4, 6, 7]   # indices into the stored strip
    strips = [("real", figdata("episode_strip", "head-3x512")["real"]),
              ("imagined,\npixel loss", figdata("episode_strip", "head-3x512")["imagined"]),
              ("imagined,\nmoving pixels", figdata("episode_strip", "final")["imagined"])]
    t = figdata("episode_strip", "final")["timesteps"]
    sub = gs[1].subgridspec(len(strips), len(picks), wspace=0.03, hspace=0.05)
    first = None
    for r, (name, frames) in enumerate(strips):
        for c, i in enumerate(picks):
            a = fig.add_subplot(sub[r, c])
            a.imshow(frames[i], interpolation="nearest")
            a.set_axis_off()
            if r == 0:
                a.text(0.5, 1.04, f"$t={t[i]}$", transform=a.transAxes, ha="center", va="bottom", fontsize=6)
            if c == 0:
                a.text(-0.08, 0.5, name, transform=a.transAxes, ha="right", va="center", fontsize=6)
            first = first or a
    fs.panel_labels([ax, first], pad=4.0)
    finish(fig, "fig-pusht-dreams")


# ------------------------------------------------------------ illustrations


def draw_scene(ax, state, block=True, agent=True, goal=True, alpha=1.0):
    """Push-T drawn as vectors from a state: walls, the green target, the grey T and the blue agent."""
    from envs.pusht_env import (AGENT_COLOR, AGENT_RADIUS, BLOCK_COLOR, GOAL_COLOR, GOAL_POSE, T_PARTS, WALL_COLOR,
                                from_state, to_world)
    rgb = lambda c: tuple(v / 255 for v in c)  # noqa: E731
    phys = from_state(state)
    if goal:
        ax.add_patch(plt.Rectangle((5, 5), 501, 501, fill=False, ec=rgb(WALL_COLOR), lw=1.2))
        for part in T_PARTS:
            ax.add_patch(plt.Polygon(to_world(part, GOAL_POSE), color=rgb(GOAL_COLOR), lw=0, alpha=0.8))
    if block:
        for part in T_PARTS:
            ax.add_patch(plt.Polygon(to_world(part, phys[4:7]), color=rgb(BLOCK_COLOR), lw=0, alpha=alpha))
    if agent:
        ax.add_patch(plt.Circle(phys[:2], AGENT_RADIUS, color=rgb(AGENT_COLOR), lw=0, alpha=alpha))
    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect("equal")
    ax.set_axis_off()


def t_outline(ax, state, **kw):
    from envs.pusht_env import T_OUTLINE, from_state, to_world
    ax.add_patch(plt.Polygon(to_world(T_OUTLINE, from_state(state)[4:7]), fill=False, **kw))


def agent_px(states):
    states = np.asarray(states)
    return states[..., 0] * 512, (1 - states[..., 1]) * 512


def fig_branches():
    d = figdata("branches", "final")
    a, b = d["window"]
    demo = d["demo"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 0.5 * W), layout="constrained")
    for ax in axes:
        draw_scene(ax, demo[a], alpha=0.35)
        t_outline(ax, demo[b], ec=TRUE, lw=0.9)
        x, y = agent_px(demo[a:b + 1])
        ax.plot(x, y, color=VLA, lw=1.4)
    for br in d["branches"]:
        x, y = agent_px(br)
        axes[1].plot(x, y, color=WAM, lw=0.8, alpha=0.9)
        t_outline(axes[1], br[-1], ec=WAM, lw=0.6, ls="--")
    handles = [Line2D([], [], color=VLA, lw=1.4), Line2D([], [], color=TRUE, lw=0.9),
               Line2D([], [], color=WAM, lw=0.8), Line2D([], [], color=WAM, lw=0.6, ls="--")]
    fig.legend(handles, ["the person's agent path", "where the person left the T",
                         "branch: the same pushes, perturbed", "where the T ends after a branch"],
               loc="outside lower center", ncol=2, fontsize=6)
    fs.panel_labels(axes, pad=2.0)
    finish(fig, "fig-pusht-branches")


def fig_planner():
    d = figdata("planner", "final")
    costs, chosen, policy = d["costs"], int(d["chosen"]), int(d["policy"])
    fig = plt.figure(figsize=(W, 0.44 * W), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
    ax = fig.add_subplot(gs[0])
    draw_scene(ax, d["state"], alpha=0.6)
    order = np.argsort(costs)[::-1]
    for i in order[::4]:   # every fourth candidate, worst first, so the best are drawn on top
        x, y = agent_px(np.concatenate([d["state"][None], d["imagined"][i]]))
        ax.plot(x, y, color=REF, lw=0.4, alpha=0.35)
    for idx, color, lw in [(policy, VLA, 1.3), (chosen, PLAN, 1.6)]:
        x, y = agent_px(np.concatenate([d["state"][None], d["imagined"][idx]]))
        ax.plot(x, y, color=color, lw=lw)
    t_outline(ax, d["imagined"][chosen][-1], ec=PLAN, lw=1.0, ls="--")
    t_outline(ax, d["real_chosen"][-1], ec=TRUE, lw=0.9)
    from envs.pusht_env import from_state
    phys = from_state(d["state"])
    cx, cy = (phys[0] + phys[4]) / 2, (phys[1] + phys[5]) / 2   # zoom on the agent and the T
    ax.set_xlim(cx - 150, cx + 150)
    ax.set_ylim(cy + 150, cy - 150)
    ax.legend([Line2D([], [], color=REF, lw=0.6), Line2D([], [], color=VLA, lw=1.3), Line2D([], [], color=PLAN, lw=1.6),
               Line2D([], [], color=PLAN, lw=1.0, ls="--"), Line2D([], [], color=TRUE, lw=0.9)],
              ["imagined candidates", "the policy's plan", "the chosen plan", "imagined T after 12 steps",
               "real T after 12 steps"], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=5.5)

    ax2 = fig.add_subplot(gs[1])
    ax2.hist(costs, bins=30, color=fs.MUTED, edgecolor=REF, lw=0.3)
    ax2.axvline(costs[policy], color=VLA, lw=1.2, label="the policy's plan")
    ax2.axvline(costs[chosen], color=PLAN, lw=1.4, label="the chosen plan, lowest cost")
    ax2.set_xlabel("imagined cost: T corners to target, px")
    ax2.set_ylabel("candidate plans")
    ax2.legend(loc="upper right", fontsize=5.5)
    fs.panel_labels([ax, ax2], pad=4.0)
    finish(fig, "fig-pusht-planner")


def fig_episode():
    d = figdata("episode", "final")
    states, cov = d["states"], d["coverage"]
    T = len(states) - 1
    picks = np.linspace(0, T, 6).round().astype(int)
    fig = plt.figure(figsize=(W, 0.5 * W), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.62])
    sub = gs[0].subgridspec(1, len(picks), wspace=0.03)
    first = None
    for c, t in enumerate(picks):
        a = fig.add_subplot(sub[0, c])
        draw_scene(a, states[t])
        a.text(0.5, 1.02, f"$t={t}$, {100 * cov[t]:.0f}%", transform=a.transAxes, ha="center", va="bottom", fontsize=6.5)
        first = first or a
    ax = fig.add_subplot(gs[1])
    ax.plot(np.arange(len(cov)), 100 * cov, color=PLAN, lw=1.1)
    ax.axhline(95, color=REF, lw=0.8, ls="--")
    ax.text(2, 96.5, "success", color=REF, fontsize=6, va="bottom")
    ax.scatter(picks, 100 * cov[picks], s=8, color=TRUE, zorder=3, lw=0)
    ax.set_ylim(0, 110)
    ax.set_xlim(0, T)
    ax.set_xlabel("timestep")
    ax.set_ylabel("coverage, %")
    fs.panel_labels([first, ax], pad=7.0)
    finish(fig, "fig-pusht-episode")


def main():
    names = [os.path.basename(p) for p in sorted(glob.glob(os.path.join(RESULTS, "*"))) if os.path.isdir(p)]
    exps = {n: load_runs(n) for n in names if glob.glob(os.path.join(RESULTS, n, "seed[0-9]*"))}
    fig_task()
    fig_prediction(exps["baseline"], load_json("baseline", "dynamics_audit.json"))
    fig_data(exps)
    if "planner-budget" in exps:
        fig_planning(exps["planner-budget"], exps["baseline"])
    fig_ablations(exps)
    fig_recipe(exps)
    fig_capacity(exps)
    fig_dreams()
    fig_pixels(exps)
    if os.path.exists(os.path.join(RESULTS, "final", "figure-data", "planner.npz")):
        fig_branches()
        fig_planner()
        fig_episode()
    fig_episode_strip()
    fig_beliefs()
    fig_what_if()
    fig_latent_map()
    fig_keypoints()
    fig_efficiency()
    pilot = load_json("supervision-pilot", "summary.json")
    if pilot:
        fig_pilot(pilot)


if __name__ == "__main__":
    main()
