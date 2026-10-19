"""Every Push-T number quoted in the paper, read from docs/results/pusht/.

Writes docs/paper/pusht-numbers.tex, one LaTeX macro per number (e.g.
\\PTbasePlan = mean best coverage of the planning WAM, in percent), and prints
the same values as a table for the tutorial and README. Means are over
training seeds; "sd" is the population standard deviation across seeds, as in
docs/figures/pusht_figures.py (the supervision pilot uses the sample SD).

    python docs/figures/pusht_numbers.py
"""

import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("PUSHT_RESULTS", os.path.join(HERE, "..", "results", "pusht"))
OUT = os.path.join(HERE, "..", "paper", "pusht-numbers.tex")
BLOCK = [4, 5, 6, 7]

RUNS = {  # macro prefix -> results folder
    "base": "baseline", "hone": "horizon-1", "hten": "horizon-10", "nolat": "no-latent", "noimg": "no-image",
    "pix": "pixel-input", "dfifty": "demos-50", "dhundred": "demos-100", "expz": "expert-p0", "expone": "expert-p0.1",
    "expthree": "expert-p0.3", "mixed": "mixed-p0.3", "diff": "diffusion-head", "chunk": "action-chunks",
    "epochs": "epochs-100", "budget": "planner-budget", "wide": "wider-model",
    # on top of action chunks
    "chunkEpochs": "chunks-epochs-100", "chunkSixteen": "chunks-k16", "chunkEnsemble": "chunks-temporal-ensembling",
    "chunkNarrow": "chunks-dmodel-64", "chunkCnn": "chunks-plain-cnn", "chunkNodec": "chunks-no-decoder",
    "chunkBranch": "chunks-branched", "chunkRestart": "chunks-restart-planner", "branchRestart": "branched-restart-planner",
    "branchRestartEpochs": "branched-restart-100-epochs",
    # the capacity round, on top of branched-restart-100-epochs; "final" is its 3x512 action head
    "headBase": "head-3x512", "headWide": "head-3x1024", "headDeep": "head-6x512", "headFlow": "flow-head-3x512",
    "latentWide": "latent-256", "headLatent": "head-3x512-latent-256", "headNodec": "head-3x512-no-decoder",
    "headPix": "pixel-input-head-3x512",
    # the dream round, on top of head-3x512; "final" adds the image loss on moving pixels
    "final": "final", "hzFifteen": "horizon-15", "hzEightNodec": "head-3x512-no-decoder-horizon-8",
    "hzFifteenNodec": "head-3x512-no-decoder-horizon-15",
}


def seeds(name):
    out = []
    for d in sorted(glob.glob(os.path.join(RESULTS, name, "seed[0-9]*"))):
        out.append({os.path.basename(p)[:-5]: json.load(open(p)) for p in glob.glob(os.path.join(d, "*.json"))})
    return out


def stat(values, scale=1.0):
    v = np.asarray(values, dtype=float) * scale
    return v.mean(), v.std()


def main():
    macros = {}

    def put(name, value, fmt):
        macros[name] = fmt.format(value)

    for prefix, folder in RUNS.items():
        S = seeds(folder)
        if not S:
            continue
        P = "PT" + prefix
        for key, short in [("wam_test", "Wam"), ("vla_test", "Vla"), ("wam_mpc_test", "Plan"),
                           ("wam_mpc_cem_test", "PlanCem"), ("wam_mpc_wide_test", "PlanWide"), ("wam_mpc_noprior_test", "PlanNoprior"),
                           ("wam_mpc_warm_test", "PlanWarm")]:
            if all(key in s for s in S):
                m, sd = stat([s[key]["mean_max_coverage"] for s in S], 100)
                put(P + short, m, "{:.1f}")
                put(P + short + "Sd", sd, "{:.1f}")
                put(P + short + "Succ", sum(round(s[key]["success_rate"] * s[key]["episodes"]) for s in S), "{:d}")
                if key == "wam_test" and all("vla_test" in s for s in S):   # how far the WAM's own policy trails mini-VLA
                    put(P + "Gap", stat([s["vla_test"]["mean_max_coverage"] - s["wam_test"]["mean_max_coverage"] for s in S], 100)[0], "{:.1f}")
                if "plan_checks" in S[0][key]:
                    for c, cname in [("expected_gain_px", "Expected"), ("real_gain_px", "Real"), ("plan_error_px", "PlanErr")]:
                        put(P + short + cname, stat([s[key]["plan_checks"][c] for s in S])[0], "{:.1f}")
        if all("wam_eval" in s for s in S):
            e = [s["wam_eval"] for s in S]
            for k, kname in [(1, "One"), (15, "Fifteen")]:
                put(P + "State" + kname, stat([x["state_nmse"][k] for x in e])[0], "{:.4f}")
                put(P + "Copy" + kname, stat([x["state_nmse_copy"][k] for x in e])[0], "{:.4f}")
            put(P + "StateFifteenSd", stat([x["state_nmse"][15] for x in e])[1], "{:.4f}")
            cf = [x["counterfactual"] for x in e]
            put(P + "Cf", stat([c["wam_nmse"][9] for c in cf])[0], "{:.4f}")
            put(P + "CfSd", stat([c["wam_nmse"][9] for c in cf])[1], "{:.4f}")
            put(P + "Noop", stat([c["noop_nmse"][9] for c in cf])[0], "{:.3f}")
            put(P + "CfBlock", stat([np.mean([c["wam_nmse_per_dim"][9][j] for j in BLOCK]) for c in cf])[0], "{:.4f}")
            put(P + "NoopBlock", stat([np.mean([c["noop_nmse_per_dim"][9][j] for j in BLOCK]) for c in cf])[0], "{:.4f}")
            # block error relative to "the block stays put" on the same starts: below 1 beats it
            put(P + "CfBlockRatio", stat([np.mean([c["wam_nmse_per_dim"][9][j] for j in BLOCK]) /
                                          np.mean([c["noop_nmse_per_dim"][9][j] for j in BLOCK]) for c in cf])[0], "{:.2f}")
            put(P + "FactBlock", stat([np.mean([x["state_nmse_per_dim"][10][j] for j in BLOCK]) for x in e])[0], "{:.4f}")
            if all(x["image_mse"] for x in e):   # empty when the WAM has no image decoder
                put(P + "ImgOne", stat([x["image_mse"][1] for x in e])[0], "{:.5f}")
                put(P + "ImgCopyOne", stat([x["image_mse_copy"][1] for x in e])[0], "{:.5f}")
                put(P + "ImgFifteen", stat([x["image_mse"][15] for x in e])[0], "{:.5f}")
                put(P + "ImgCopyFifteen", stat([x["image_mse_copy"][15] for x in e])[0], "{:.5f}")
            put(P + "RtwoFifteen", stat([np.mean([r for r in x["state_r2"][15] if r is not None]) for x in e])[0], "{:.3f}")

    for prefix, folder in RUNS.items():   # dreams checked in the simulator, and the decoder on moving pixels
        rows = []
        for d in sorted(glob.glob(os.path.join(RESULTS, folder, "seed[0-9]*"))):
            e = json.load(open(os.path.join(d, "wam_eval.json"))) if os.path.exists(os.path.join(d, "wam_eval.json")) else {}
            extra = os.path.join(d, "wam_eval_dreams.json")   # computed afterwards for runs older than the check
            if "dream" not in e and os.path.exists(extra):
                e = {**e, **json.load(open(extra))}
            if "dream" in e:
                rows.append(e)
        if rows:
            P = "PT" + prefix
            for k, kname in [(5, "Five"), (10, "Ten"), (20, "Twenty"), (40, "Forty")]:
                put(P + "DreamAgent" + kname, np.mean([r["dream"]["agent_px"][k - 1] for r in rows]), "{:.1f}")
                put(P + "DreamT" + kname, np.mean([r["dream"]["block_px"][k - 1] for r in rows]), "{:.1f}")
            if all("image_mse_moving" in r for r in rows):
                put(P + "MovingImg", np.mean([r["image_mse_moving"] for r in rows]), "{:.4f}")

    checks_path = os.path.join(RESULTS, "data_checks.json")
    if os.path.exists(checks_path):
        c = json.load(open(checks_path))
        put("PTdataClipped", c["clipped_actions"], "{:d}")
        put("PTdataBlockErrPninetynine", c["replay"]["block_error_px_p99"], "{:.2f}")
        put("PTdataBlockMove", c["replay"]["moving_block_step_px_median"], "{:.1f}")
        put("PTdataRender", c["render"]["mean_abs_pixel_difference"], "{:.2f}")
        put("PTdataDemoCoverage", 100 * c["demos"]["best_coverage_mean"], "{:.1f}")
        put("PTdataDemoSuccesses", c["demos"]["successes"], "{:d}")
        for e in c["expert"]:
            tag = "Zero" if e["random_action_prob"] == 0 else "Tenth"
            put("PTexpert" + tag + "Success", 100 * e["success_rate"], "{:.0f}")
            put("PTexpert" + tag + "Steps", e["mean_steps"], "{:.0f}")

    audit_path = os.path.join(RESULTS, "baseline", "dynamics_audit.json")
    if os.path.exists(audit_path):
        a = json.load(open(audit_path))["seeds"]
        rows = list(a.values())
        put("PTlinFactOne", stat([r["factual_nmse"]["state_action"][0] for r in rows])[0], "{:.4f}")
        put("PTlinFactFifteen", stat([r["factual_nmse"]["state_action"][14] for r in rows])[0], "{:.4f}")
        put("PTlinCf", stat([r["counterfactual_nmse"]["state_action"][9] for r in rows])[0], "{:.4f}")
        put("PTlinCfBlock", stat([np.mean([r["counterfactual_nmse_per_dim"]["state_action"][9][j] for j in BLOCK]) for r in rows])[0], "{:.4f}")
        put("PTlinCfAgentPx", stat([r["counterfactual_px_at_10"]["state_action"]["agent"] for r in rows])[0], "{:.1f}")
        put("PTlinCfBlockPx", stat([r["counterfactual_px_at_10"]["state_action"]["block"] for r in rows])[0], "{:.1f}")
        put("PTstateonlyFactFifteen", stat([r["factual_nmse"]["state_only"][14] for r in rows])[0], "{:.3f}")

    eff = os.path.join(RESULTS, "baseline", "efficiency", "efficiency.json")
    if os.path.exists(eff):
        d = json.load(open(eff))
        put("PTparamsWam", d["parameters"]["wam_total"], "{:,d}")
        put("PTparamsVla", d["parameters"]["vla_total"], "{:,d}")
        put("PTparamsDecoder", d["parameters"]["wam"]["image_decoder"], "{:,d}")
        put("PTstepWam", d["train_step_ms"]["wam"] / 1e3, "{:.2f}")
        put("PTstepStates", d["train_step_ms"]["wam_state_only"] / 1e3, "{:.2f}")
        put("PTstepVla", d["train_step_ms"]["vla"], "{:.0f}")
        put("PTactMs", d["latency_ms"]["act_wam"], "{:.1f}")
        put("PTimagineTenMs", d["latency_ms"]["imagine_10_states"], "{:.2f}")
        put("PTimagineTenImgMs", d["latency_ms"]["imagine_10_states_and_images"], "{:.1f}")

    for prefix, folder in RUNS.items():   # parameter counts, from each run's own benchmark
        eff = os.path.join(RESULTS, folder, "efficiency", "efficiency.json")
        if os.path.exists(eff):
            d = json.load(open(eff))
            put("PT" + prefix + "ParamsWam", d["parameters"]["wam_total"], "{:,d}")
            put("PT" + prefix + "ParamsVla", d["parameters"]["vla_total"], "{:,d}")
            put("PT" + prefix + "ParamsWamM", d["parameters"]["wam_total"] / 1e6, "{:.2f}")
            put("PT" + prefix + "ParamsVlaM", d["parameters"]["vla_total"] / 1e6, "{:.2f}")
            put("PT" + prefix + "ParamsHead", d["parameters"]["wam"]["policy_head"], "{:,d}")
            if "image_decoder" in d["parameters"]["wam"]:
                put("PT" + prefix + "ParamsDecoder", d["parameters"]["wam"]["image_decoder"], "{:,d}")
                put("PT" + prefix + "ParamsNodecM", (d["parameters"]["wam_total"] - d["parameters"]["wam"]["image_decoder"]) / 1e6, "{:.2f}")

    laptop = os.path.join(RESULTS, "final", "efficiency-laptop", "efficiency.json")
    if os.path.exists(laptop):   # the final model on one laptop CPU thread
        d = json.load(open(laptop))
        put("PTfinalLaptopStepWam", d["train_step_ms"]["wam"] / 1e3, "{:.2f}")
        put("PTfinalLaptopStepStates", d["train_step_ms"]["wam_state_only"] / 1e3, "{:.2f}")
        put("PTfinalLaptopStepVla", d["train_step_ms"]["vla"], "{:.0f}")
        put("PTfinalLaptopActMs", d["latency_ms"]["act_wam"], "{:.2f}")
        put("PTfinalLaptopPlanMs", d["latency_ms"]["plan_step"], "{:.0f}")

    laptop = os.path.join(RESULTS, "baseline", "efficiency-laptop", "efficiency.json")
    if os.path.exists(laptop):   # one thread of an Apple-silicon laptop CPU, like the point-mass benchmark
        d = json.load(open(laptop))
        put("PTlaptopStepWam", d["train_step_ms"]["wam"] / 1e3, "{:.2f}")
        put("PTlaptopStepStates", d["train_step_ms"]["wam_state_only"] / 1e3, "{:.2f}")
        put("PTlaptopStepVla", d["train_step_ms"]["vla"], "{:.0f}")
        put("PTlaptopActMs", d["latency_ms"]["act_wam"], "{:.2f}")
        put("PTlaptopPlanMs", d["latency_ms"]["plan_step"], "{:.0f}")
        put("PTlaptopImagineTenMs", d["latency_ms"]["imagine_10_states"], "{:.2f}")
        put("PTlaptopImagineTenImgMs", d["latency_ms"]["imagine_10_states_and_images"], "{:.1f}")

    whatif = os.path.join(RESULTS, "baseline", "figure-data", "what_if.npz")
    if os.path.exists(whatif):   # 15-step constant pushes in 8 directions, unit-square endpoint distance
        w = np.load(whatif)
        for key, name in [("endpoint_error", "Agent"), ("ignore_push_error", "AgentIgnore"),
                          ("block_endpoint_error", "Block"), ("block_ignore_push_error", "BlockIgnore")]:
            v = w[key][:, 1:].mean(1)   # mean over the 8 directions, per seed
            put("PTwhatif" + name, v.mean(), "{:.3f}")
            put("PTwhatif" + name + "Sd", v.std(), "{:.3f}")

    for run, tag in [("baseline", "State"), ("pixel-input", "Pixel")]:   # best-keypoint tracking R^2, mean over seeds
        kp = os.path.join(RESULTS, run, "figure-data", "keypoint_stats.json")
        if os.path.exists(kp):
            r = json.load(open(kp))["run"]
            put("PTkpAgent" + tag, np.mean(r["tracking_r2"]), "{:.2f}")
            put("PTkpBlock" + tag, np.mean(r["block_tracking_r2"]), "{:.2f}")

    if "PTbaseStateFifteen" in macros:
        put("PTbaseCopyRatio", float(macros["PTbaseCopyFifteen"]) / float(macros["PTbaseStateFifteen"]), "{:.0f}")

    pilot = os.path.join(RESULTS, "supervision-pilot", "summary.json")
    if os.path.exists(pilot):
        p = json.load(open(pilot))
        for c, cname in [("bc", "Bc"), ("current", "Cur"), ("future", "Fut")]:
            put("PTpilot" + cname, 100 * p["coverage_summary"][c]["mean"], "{:.1f}")
            put("PTpilot" + cname + "Sd", 100 * p["coverage_summary"][c]["sample_sd"], "{:.1f}")
        put("PTpilotFutMinusCur", 100 * p["coverage_summary"]["future-minus-current"]["mean"], "{:.1f}")
        put("PTpilotCurMinusBc", 100 * p["coverage_summary"]["current-minus-bc"]["mean"], "{:.1f}")

    with open(OUT, "w") as f:
        f.write("% Generated by docs/figures/pusht_numbers.py from docs/results/pusht/. Do not edit by hand.\n")
        for k, v in sorted(macros.items()):   # a typeset minus sign, not a hyphen
            f.write(f"\\newcommand{{\\{k}}}{{{v.replace('-', '\\ensuremath{-}', 1) if v.startswith('-') else v}}}\n")
    width = max(len(k) for k in macros)
    for k, v in sorted(macros.items()):
        print(f"{k:<{width}}  {v}")
    print(f"\n{len(macros)} numbers -> {OUT}")


if __name__ == "__main__":
    main()
