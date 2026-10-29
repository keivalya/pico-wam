"""Inspect the saved Fig. 1 predictions and replay its actions, without checkpoints.

This descriptive check keeps the original median-regret case selection unchanged.
It does not regenerate neural predictions or evaluate closed-loop task completion.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

PAPER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER.parent))
from envs.pusht_env import PushTEnv, WORKSPACE
from followup.common import GeneralPlanner


def main():
    data = json.loads((PAPER / 'analysis/tmlr-example.json').read_text())
    start = np.asarray(data['start_state'], dtype=np.float32)
    cost = GeneralPlanner(None, None, 'cpu').cost
    rows = {}
    for name, row in data['conditions'].items():
        actions = np.asarray(row['actions'], dtype=np.float32)
        predicted = np.asarray(row['predicted_states'], dtype=np.float32)
        saved = np.asarray(row['true_states'], dtype=np.float32)
        replay = PushTEnv(image_size=64).simulate(start, actions)
        np.testing.assert_allclose(replay, saved, rtol=0, atol=1e-6)
        # All four pose coordinates must agree, including cos/sin orientation.
        unchanged = bool(np.max(np.abs(replay[:, 4:8] - start[4:8])) < 1e-7)
        rows[name] = {
            'candidate_index': row['candidate_index'], 'selection_regret': row['regret'],
            'predicted_cost': float(cost(torch.tensor(predicted)[None])[0]),
            'simulator_cost': float(cost(torch.tensor(replay)[None])[0]),
            'endpoint_position_error_px': float(np.linalg.norm(predicted[-1, 4:6] - replay[-1, 4:6]) * WORKSPACE),
            'block_pose_unchanged_at_all_steps': unchanged,
            'max_replay_difference': float(np.abs(replay - saved).max()),
        }
    assert all(r['block_pose_unchanged_at_all_steps'] for r in rows.values())
    result = {'seed': data['seed'], 'bank_index': data['bank_index'],
              'selection_rule': data['selection_rule'], 'conditions': rows,
              'scope': 'Saved predictions plus simulator replay. Lower true cost here reflects agent approach; neither sequence moves the block. This is not a closed-loop failure test.'}
    output = PAPER / 'generated/decision-trace.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
