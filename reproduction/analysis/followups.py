"""Tables and macros for the follow-up studies, from their bundled reports.

Run: python analysis/followups.py
Requires numpy and scipy. Reads analysis/ral-evidence-v2.json, ral-evidence-f0.json and the
follow-up reports (f1-candidate-audit, s3a-history-bc, s3b-diffusion-policy, s4-planners,
s4-imitation-followup, architecture-v1, s5a-original-pusht, s5b-fresh-prediction). A report that is not bundled yet is simply omitted.
Every interval is the study's own paired Student-t interval over training runs.
"""
from pathlib import Path
import json
import os

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / 'analysis'
OUT = Path(os.environ.get('EVIDENCE_OUT', ROOT / 'generated'))


def load(name):
    p = A / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def paired(v):
    v = np.asarray(v, float); h = t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return {'mean': float(v.mean()), 'ci95': [float(v.mean() - h), float(v.mean() + h)], 'n': len(v), 'pos': int((v > 0).sum())}


def num(v, d=1):
    s = f'{v:.{d}f}'
    if float(s) == 0: s = s.lstrip('-')  # no negative zero
    return s.replace('-', '\\ensuremath{-}')


def ci(r, d=1, scale=1.0):
    return f'{num(r["mean"] * scale, d)} [{num(r["ci95"][0] * scale, d)}, {num(r["ci95"][1] * scale, d)}]'


macros = []
def macro(name, value):
    macros.append(f'\\newcommand{{\\{name}}}{{{value}}}')


def macro_ci(name, r):
    # the combined 'mean [low, high]' form plus its parts, for 'X points [a, b]' in the text
    macro(name, ci(r)); macro(name + 'Mean', num(r['mean']))
    macro(name + 'Low', num(r['ci95'][0])); macro(name + 'High', num(r['ci95'][1]))


def evidence_effects(x):
    """Primary interaction and restart-vs-BC contrasts from an evidence-study extract's report."""
    e = x['report']['effects']
    return {k: {'mean': v['mean'], 'ci95': v['ci95'], 'n': len(v['per_run_percentage_points']),
                'pos': sum(p > 0 for p in v['per_run_percentage_points'])} for k, v in e.items()}


def prediction_rows(x):
    """Per-run prediction metrics from an evidence-study extract: demo, branch and their paired difference."""
    seeds = x['config']['training_seeds']; H, K = x['config']['prediction_horizon'], x['config']['counterfactual_horizon'] - 1
    def get(cond, f):
        return np.array([f(x['runs'][f'{s}/{cond}']['test/prediction']) for s in seeds])
    fs = {'state15': lambda p: p['factual']['state_nmse'][H],
          'block10': lambda p: float(np.asarray(p['counterfactual']['wam_nmse_per_dim'])[K, 4:8].mean()),
          'zero10': lambda p: float(np.asarray(p['counterfactual']['noop_nmse_per_dim'])[K, 4:8].mean()),
          'rmse10': lambda p: p['counterfactual']['block_position_rmse_px'][K]}
    return {k: (get('wam_demo', f), get('wam_branch', f)) for k, f in fs.items()}


def main():
    v2, f0 = load('ral-evidence-v2'), load('ral-evidence-f0')
    f1, s3a, s3b, s4 = load('f1-candidate-audit'), load('s3a-history-bc'), load('s3b-diffusion-policy'), load('s4-planners')
    s5a, s5b = load('s5a-original-pusht'), load('s5b-fresh-prediction')
    si, ar = load('s4-imitation-followup'), load('architecture-v1')

    # ---------------- Table: why planning fails without branches and works with them.
    rows = ['\\begin{tabular}{@{}lrrr@{}}', '\\toprule', 'Measure & Demos & + branches & Difference \\\\', '\\midrule']
    def group(title): rows.append(f'\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}} \\\\')
    def line(label, a, b, d, digits):
        rows.append(f'{label} & {num(np.mean(a) if not isinstance(a, float) else a, digits)} & {num(np.mean(b) if not isinstance(b, float) else b, digits)} & {ci(d, digits)} \\\\')
    pr = prediction_rows(v2)
    group('Prediction, 21 unseen test demonstrations')
    line('State nMSE, recorded, 15 steps', pr['state15'][0], pr['state15'][1], paired(pr['state15'][1] - pr['state15'][0]), 3)
    line(f'Block nMSE, random, 10 steps$^a$', pr['block10'][0], pr['block10'][1], paired(pr['block10'][1] - pr['block10'][0]), 3)
    line('Block RMSE, px, random, 10 steps', pr['rmse10'][0], pr['rmse10'][1], paired(pr['rmse10'][1] - pr['rmse10'][0]), 1)
    macro('MxZeroBlockTest', num(float(np.mean(pr['zero10'][0])), 3))
    macro('MxBlockTestDemo', num(float(np.mean(pr['block10'][0])), 3)); macro('MxBlockTestBranch', num(float(np.mean(pr['block10'][1])), 3))
    macro('MxRmseTestDemo', num(float(np.mean(pr['rmse10'][0])))); macro('MxRmseTestBranch', num(float(np.mean(pr['rmse10'][1]))))
    if s5b:
        c, d = s5b['report']['conditions'], s5b['report']['contrasts']
        group(f'Prediction, {s5b["report"]["episodes"]} fresh scripted-expert episodes')
        for key, label, digits in [('factual_nmse_k15', 'State nMSE, recorded, 15 steps', 3),
                                   ('counterfactual_block_nmse_k10', 'Block nMSE, random, 10 steps$^a$', 3),
                                   ('block_position_rmse_px_k10', 'Block RMSE, px, random, 10 steps', 1)]:
            line(label, c[f'wam_demo/{key}']['mean'], c[f'wam_branch/{key}']['mean'], d[f'wam_branch_minus_wam_demo/{key}'], digits)
        macro('MxZeroBlockFresh', num(c['wam_demo/zero_action_block_nmse_k10']['mean'], 3))
        macro('MxBlockFreshDemo', num(c['wam_demo/counterfactual_block_nmse_k10']['mean'], 3)); macro('MxBlockFreshBranch', num(c['wam_branch/counterfactual_block_nmse_k10']['mean'], 3))
        macro('MxRmseFreshDemo', num(c['wam_demo/block_position_rmse_px_k10']['mean'])); macro('MxRmseFreshBranch', num(c['wam_branch/block_position_rmse_px_k10']['mean']))
        macro('MxRmseFreshDiff', ci(d['wam_branch_minus_wam_demo/block_position_rmse_px_k10']))
        macro('MxFreshEpisodes', s5b['report']['episodes']); macro('MxFreshTransitions', f'{s5b["report"]["transitions"]:,}'.replace(',', '{,}'))
    if f1:
        c, d = f1['report']['conditions'], f1['report']['contrasts']; n = f1['registration']['n_states']
        group('Candidate ranking, ' + f'{n:,}'.replace(',', '{,}') + ' common states')
        for key, label, digits in [('common/spearman/all', 'Rank correlation with true cost', 2),
                                   ('common/elite_overlap/all', 'Overlap of top 32', 2),
                                   ('common/regret/all', 'Selection regret$^b$', 2),
                                   ('common/optimism/all', 'Selection optimism$^b$', 2),
                                   ('own/gain_over_prior/all', 'Gain over policy prior$^{b,c}$', 2)]:
            line(label, c[f'wam_demo/{key}']['mean'], c[f'wam_branch/{key}']['mean'], d[f'wam_branch_minus_wam_demo/{key}'], digits)
        r = d['wam_branch_minus_wam_demo/common/regret/all']
        macro('MxRegretDemo', num(c['wam_demo/common/regret/all']['mean'], 2)); macro('MxRegretBranch', num(c['wam_branch/common/regret/all']['mean'], 2))
        macro('MxRegretDiff', ci(r, 2)); macro('MxRegretPos', sum(v > 0 for v in r['per_run'])); macro('MxRuns', len(r['per_run']))
        macro('MxRhoDemo', num(c['wam_demo/common/spearman/all']['mean'], 2)); macro('MxRhoBranch', num(c['wam_branch/common/spearman/all']['mean'], 2))
        macro('MxStates', f'{n:,}'.replace(',', '{,}')); macro('MxDemoStates', f1['registration']['n_demo_states'])
        macro('MxExpertStates', f'{f1["registration"]["n_expert_states"]:,}'.replace(',', '{,}'))
        for key, name in [('common/elite_overlap/all', 'MxOverlap'), ('common/optimism/all', 'MxOptimism'), ('own/gain_over_prior/all', 'MxGain')]:
            macro(name + 'Demo', num(c[f'wam_demo/{key}']['mean'], 2)); macro(name + 'Branch', num(c[f'wam_branch/{key}']['mean'], 2))
            macro(name + 'Pos', sum(v > 0 for v in d[f'wam_branch_minus_wam_demo/{key}']['per_run']))
        for key, name in [('common/regret/demo', 'MxRegretDiffDemoStates'), ('common/regret/expert', 'MxRegretDiffExpertStates')]:
            macro(name, ci(d[f'wam_branch_minus_wam_demo/{key}'], 2))
    rows += ['\\bottomrule', '\\end{tabular}']
    (OUT / 'table-mechanism.tex').write_text('\n'.join(rows) + '\n')

    # ---------------- Table: replications and baselines, restart planning with branches against a reference.
    rows = ['\\begin{tabular}{@{}llrr@{}}', '\\toprule', 'Study, seeds & Reference & Coverage & Success \\\\', '\\midrule']
    ev = evidence_effects(v2); imit = [ev['mean_max_coverage/restart_branch_vs_bc']]  # restart search minus each imitation reference
    rows.append(f'Matched, 15 & BC & {ci(ev["mean_max_coverage/restart_branch_vs_bc"])} & {ci(ev["success_rate/restart_branch_vs_bc"])} \\\\')
    if f0:
        e0 = evidence_effects(f0); imit.append(e0['mean_max_coverage/restart_branch_vs_bc'])
        rows.append(f'Horizon-matched, 15 & BC & {ci(e0["mean_max_coverage/restart_branch_vs_bc"])} & {ci(e0["success_rate/restart_branch_vs_bc"])} \\\\')
        macro_ci('FzInteraction', e0['mean_max_coverage/interaction']); macro('FzInteractionPos', e0['mean_max_coverage/interaction']['pos'])
        macro('FzBranchEffect', ci(e0['mean_max_coverage/restart_branch_effect']))
        macro_ci('FzFutureVsCurrent', e0['mean_max_coverage/future_vs_current_branch_policy'])
        cond = f0['report']['conditions']
        for key, name in [('wam_demo/restart', 'FzDemoRestart'), ('wam_demo/policy', 'FzDemoPolicy'), ('wam_branch/restart', 'FzBranchRestart'),
                          ('wam_demo/warm', 'FzDemoWarm'), ('wam_branch/warm', 'FzBranchWarm'), ('bc/policy', 'FzBcPolicy')]:
            macro(name, num(cond[key]['mean_max_coverage']['mean']))
    if s5a:
        c = s5a['report']['contrasts']; imit.append(c['wam_branch_restart minus bc_policy, mean_max_coverage'])
        rows.append(f'Original gym-pusht, 15 & BC & {ci(c["wam_branch_restart minus bc_policy, mean_max_coverage"])} & {ci(c["wam_branch_restart minus bc_policy, success_rate"])} \\\\')
        for key, name in [('wam_branch_restart minus wam_branch_warm, mean_max_coverage', 'SaRestartWarm'),
                          ('wam_branch_restart minus wam_demo_restart, mean_max_coverage', 'SaBranchEffect'),
                          ('wam_branch_restart minus wam_branch_policy, mean_max_coverage', 'SaRestartPolicy'),
                          ('wam_branch_restart minus bc_policy, mean_max_reward', 'SaRewardVsBc')]:
            macro_ci(name, c[key])
        fid = s5a['registration']['replay_fidelity']
        macro('SaFidelityPninetynine', num(fid['one_step_block_px_p99'], 2)); macro('SaFidelityTransitions', f'{fid["transitions"]:,}'.replace(',', '{,}'))
        ev5 = s5a['report']['evaluations']
        macro('SaBcCov', num(ev5['bc_policy']['mean_max_coverage']['mean'])); macro('SaRestartCov', num(ev5['wam_branch_restart']['mean_max_coverage']['mean']))
    if s3a:
        c = s3a['report']['contrasts']; ev3 = s3a['report']['evaluations']; imit.append(c['v2_wam_branch_restart minus history_bc, mean_max_coverage'])
        rows.append(f'Tuned BC, 15 & tuned BC & {ci(c["v2_wam_branch_restart minus history_bc, mean_max_coverage"])} & {ci(c["v2_wam_branch_restart minus history_bc, success_rate"])} \\\\')
        macro_ci('TbVsBc', c['history_bc minus v2_bc_policy, mean_max_coverage'])
        macro('TbCov', num(ev3['history_bc']['mean_max_coverage']['mean'])); macro('TbSucc', num(ev3['history_bc']['success_rate']['mean']))
        macro('TbBcSucc', num(ev3['v2_bc_policy']['success_rate']['mean']))
        sel = s3a['tuning']['selected']
        macro('TbSelected', f'history {sel["history"]}, chunk {sel["chunk"]}, width {sel["width"]}, learning rate {sel["lr"]:g}')
        best2 = max((r for r in s3a['tuning']['all'] if r['hp']['history'] == 2), key=lambda r: r['mean_max_coverage'])
        macro('TbBestHistoryTwo', num(100 * best2['mean_max_coverage'])); macro('TbSelectedTuning', num(100 * max(r['mean_max_coverage'] for r in s3a['tuning']['all'])))
    if s3b:
        c = s3b['report']['contrasts']; ev3b = s3b['report']['evaluations']; imit.append(c['v2_wam_branch_restart minus diffusion_policy, mean_max_coverage'])
        rows.append(f'Diffusion Policy, 5 & DP & {ci(c["v2_wam_branch_restart minus diffusion_policy, mean_max_coverage"])} & {ci(c["v2_wam_branch_restart minus diffusion_policy, success_rate"])} \\\\')
        macro('DpCov', num(ev3b['diffusion_policy']['mean_max_coverage']['mean'])); macro('DpSucc', num(ev3b['diffusion_policy']['success_rate']['mean']))
        macro('DpRestartSucc', num(ev3b['v2_wam_branch_restart']['success_rate']['mean'])); macro('DpBcSucc', num(ev3b['v2_bc_policy']['success_rate']['mean']))
        macro_ci('DpVsBc', c['diffusion_policy minus v2_bc_policy, mean_max_coverage'])
        macro_ci('DpRestartVsDp', c['v2_wam_branch_restart minus diffusion_policy, mean_max_coverage'])
        r = c['v2_wam_branch_restart minus diffusion_policy, success_rate']; macro_ci('DpRestartVsDpSucc', r)
        macro('DpRestartVsDpSuccPos', r['runs_above_zero']); macro('DpRuns', len(r['per_run']))
    words = {3: 'three', 4: 'four', 5: 'five', 6: 'six'}
    macro('CmpCount', words[len(imit)]); macro('CmpFollowups', words[len(imit) - 1])
    macro('CmpSpanLow', num(min(r['ci95'][0] for r in imit))); macro('CmpSpanHigh', num(max(r['ci95'][1] for r in imit)))
    macro('CmpAllIncludeZero', 'yes' if all(r['ci95'][0] <= 0 <= r['ci95'][1] for r in imit) else 'no')
    if s4:
        c = s4['report']['contrasts']; e4 = s4['report']['evaluations']
        rows.append('\\midrule')
        rows.append(f'State dynamics, 15 & latent & {ci(c["state_branch_restart minus latent_branch_restart, mean_max_coverage"])} & {ci(c["state_branch_restart minus latent_branch_restart, success_rate"])} \\\\')
        rows.append(f'Oracle dynamics, 15 & latent & {ci(c["oracle_restart minus latent_branch_restart, mean_max_coverage"])} & {ci(c["oracle_restart minus latent_branch_restart, success_rate"])} \\\\')
        for key, name in [('state_branch_restart minus latent_branch_restart, mean_max_coverage', 'SfStateVsLatent'),
                          ('state_branch_restart minus latent_branch_restart, success_rate', 'SfStateVsLatentSucc'),
                          ('oracle_restart minus latent_branch_restart, mean_max_coverage', 'SfOracleVsLatent'),
                          ('oracle_restart minus latent_branch_restart, success_rate', 'SfOracleVsLatentSucc'),
                          ('state_branch_restart minus state_demo_restart, mean_max_coverage', 'SfBranchEffect'),
                          ('oracle_restart minus oracle_warm, mean_max_coverage', 'SfOracleRestartWarm'),
                          ('latent_branch_restart minus latent_branch_warm, mean_max_coverage', 'SfLatentRestartWarm'),
                          ('state_branch_restart minus state_branch_warm, mean_max_coverage', 'SfStateRestartWarm')]:
            macro_ci(name, c[key]); macro(name + 'Pos', c[key]['runs_above_zero'])
        for key, name in [('latent_branch_restart', 'SfLatentRestart'), ('state_branch_restart', 'SfStateRestart'),
                          ('state_demo_restart', 'SfStateDemoRestart'), ('oracle_restart', 'SfOracleRestart'), ('oracle_warm', 'SfOracleWarm'),
                          ('latent_branch_warm', 'SfLatentWarm'), ('state_branch_warm', 'SfStateWarm')]:
            macro(name, num(e4[key]['mean_max_coverage']['mean'])); macro(name + 'Succ', num(e4[key]['success_rate']['mean']))
    rows += ['\\bottomrule', '\\end{tabular}']
    (OUT / 'table-replication.tex').write_text('\n'.join(rows) + '\n')

    # ---------------- Exploratory: frozen imitation evaluated on the dynamics-model study's final starts.
    if si:
        c = si['report']['contrasts']
        for ref, tag in [('bc', 'Bc'), ('history_bc', 'Tb'), ('diffusion_policy', 'Dp')]:
            for side, name in [('oracle_restart', 'SiOracle'), ('latent_branch_restart', 'SiLatent')]:
                macro_ci(f'{name}Vs{tag}', c[f'{side} minus {ref}, mean_max_coverage'])
                macro_ci(f'{name}Vs{tag}Succ', c[f'{side} minus {ref}, success_rate'])
        macro('SiDpRuns', len(c['oracle_restart minus diffusion_policy, mean_max_coverage']['per_run']))

    # ---------------- Table: architecture variants minus the full model, reactive and restart coverage.
    if ar:
        c, e = ar['report']['contrasts'], ar['report']['evaluations']
        rows = ['\\begin{tabular}{@{}lrr@{}}', '\\toprule', 'Variant minus full & Reactive & Restart \\\\', '\\midrule']
        for key, label in [('detach_aux', 'Auxiliary gradients blocked$^a$'), ('image_only', 'Image only'), ('state_only', 'State only'),
                           ('no_text', 'No instruction'), ('no_future', 'No future-state loss'), ('no_latent', 'No latent loss')]:
            pol, res = c[f'{key}_policy minus full_policy, mean_max_coverage'], c[f'{key}_restart minus full_restart, mean_max_coverage']
            rows.append(f'{label} & {ci(pol)} & {ci(res)} \\\\')
            tag = ''.join(w.capitalize() for w in key.split('_'))
            macro_ci(f'Ar{tag}Policy', pol); macro_ci(f'Ar{tag}Restart', res)
            macro(f'Ar{tag}PolicyCov', num(e[f'{key}_policy']['mean_max_coverage']['mean'])); macro(f'Ar{tag}RestartCov', num(e[f'{key}_restart']['mean_max_coverage']['mean']))
            macro(f'Ar{tag}RestartPos', res['runs_above_zero'])
        rows += ['\\midrule', f'Full model, coverage \\% & {num(e["full_policy"]["mean_max_coverage"]["mean"])} & {num(e["full_restart"]["mean_max_coverage"]["mean"])} \\\\', '\\bottomrule', '\\end{tabular}']
        (OUT / 'table-architecture.tex').write_text('\n'.join(rows) + '\n')
        macro('ArFullPolicyCov', num(e['full_policy']['mean_max_coverage']['mean'])); macro('ArFullRestartCov', num(e['full_restart']['mean_max_coverage']['mean']))
        macro('ArRuns', len(c[ar['report']['primary']]['per_run']))
    (OUT / 'followups-statistics.tex').write_text('% Generated by analysis/followups.py. Do not edit.\n' + '\n'.join(macros) + '\n')
    print('\n'.join(macros))


if __name__ == '__main__':
    main()
