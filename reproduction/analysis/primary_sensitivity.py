"""Retrospective influence checks on the two archived primary run differences.

No new data, model selection or replacement of the original Student-t analyses.
Bootstrap percentiles assume exchangeable paired runs conditional on the data/protocol.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

PAPER = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_DRAWS = 50000


def interval(values):
    radius = t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean() - radius), float(values.mean() + radius)]


def summarize(record):
    values = np.asarray(record.get('per_run', record.get('per_run_percentage_points')), dtype=float)
    assert values.shape == (15,) and np.isfinite(values).all()
    np.testing.assert_allclose(values.mean(), record['mean'], rtol=0, atol=1e-10)
    np.testing.assert_allclose(interval(values), record['ci95'], rtol=0, atol=1e-10)
    omissions = [np.delete(values, i) for i in range(len(values))]
    means = [float(v.mean()) for v in omissions]
    intervals = [interval(v) for v in omissions]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = values[rng.integers(len(values), size=(BOOTSTRAP_DRAWS, len(values)))].mean(axis=1)
    return {'original_mean': float(values.mean()), 'original_ci95': interval(values),
            'median_run_difference': float(np.median(values)),
            'positive_runs': int((values > 0).sum()), 'n_runs': len(values),
            'leave_one_out': [{'omitted_run_index': i, 'mean': m, 'ci95': ci}
                              for i, (m, ci) in enumerate(zip(means, intervals))],
            'leave_one_out_mean_range': [min(means), max(means)],
            'minimum_leave_one_out_ci_lower': min(ci[0] for ci in intervals),
            'bootstrap_percentile_ci95': np.quantile(draws, [.025, .975]).tolist()}


def main():
    main_study = json.loads((PAPER / 'analysis/ral-evidence-v2.json').read_text())
    factorial = json.loads((PAPER / 'analysis/tmlr-factorial-v1.json').read_text())
    effects = {
        'data_search_interaction': summarize(main_study['report']['effects']['mean_max_coverage/interaction']),
        'exact_scoring_with_learned_proposals': summarize(factorial['report']['contrasts']['LE minus LL, mean_max_coverage']),
    }
    report = {'status': 'Retrospective sensitivity analysis, conditional on the original studies; not independent confirmation.',
              'bootstrap_seed': BOOTSTRAP_SEED, 'bootstrap_draws': BOOTSTRAP_DRAWS,
              'scope': 'Resampling whole paired run differences preserves pairing. Does not quantify new datasets/tasks or replace original primary Student-t intervals.',
              'effects': effects}
    out = PAPER / 'generated'
    out.mkdir(exist_ok=True)
    (out / 'primary-sensitivity.json').write_text(json.dumps(report, indent=2) + '\n')
    macros = []
    for key, prefix in [('data_search_interaction', 'SnInteraction'),
                        ('exact_scoring_with_learned_proposals', 'SnScoring')]:
        v = effects[key]
        for suffix, value in [('LooLow', v['leave_one_out_mean_range'][0]),
                              ('LooHigh', v['leave_one_out_mean_range'][1]),
                              ('MinCiLower', v['minimum_leave_one_out_ci_lower'])]:
            macros.append('\\newcommand{\\' + prefix + suffix + '}{' + f'{value:.1f}' + '}')
    (out / 'sensitivity-statistics.tex').write_text('\n'.join(macros) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
