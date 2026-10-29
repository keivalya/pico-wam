"""Study 4: a learned state-space dynamics model for the same planner, plus the oracle diagnostic.

The state model is a residual MLP on normalized state,

    s_{k+1} = s_k + sigma * f([(s_k - mu) / sigma, a_k]),

with the latent dynamics' hidden size, 2 x 256. It is trained on exactly the auxiliary windows
that the evidence study's wam_demo or wam_branch saw at each update: the same sampler streams,
batch size, horizon, update count, optimizer and normalization statistics. Only the dynamics'
state space differs: physical state here, the shared latent there.
"""
import time

import numpy as np
import torch
import torch.nn as nn


class StateDynamicsMLP(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, state_dim))
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.config = dict(state_dim=state_dim, action_dim=action_dim, hidden=hidden)

    def set_state_stats(self, mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))  # as PicoWAMPolicy

    def step(self, state, action):
        """Raw state (B, 8) and action (B, 2) -> next raw state (B, 8)."""
        n = (state - self.state_mean) / self.state_std
        return state + self.state_std * self.net(torch.cat([n, action], -1))

    def rollout(self, state, actions):
        """(B, 8), (B, H, 2) -> (B, H, 8) predicted raw states."""
        out = []
        for k in range(actions.shape[1]):
            state = self.step(state, actions[:, k]); out.append(state)
        return torch.stack(out, 1)


def train_state_model(source_manifest, condition, seed, steps=None, device='cpu', log_every=1000):
    """Train on the source study's auxiliary sampler for `condition`; returns (model, history)."""
    import torch.nn.functional as F
    from scripts.evidence_protocol import sample_indices
    from scripts.train import TrainingDataset
    c = source_manifest['protocol']['config']
    if condition not in ('wam_demo', 'wam_branch'):
        raise ValueError('The state model mirrors the dynamics data of wam_demo or wam_branch.')
    steps = steps or c['steps']
    data = TrainingDataset(source_manifest['datasets']['branches']['path'], horizon=c['horizon'], action_chunk=c['action_chunk'])
    train_ids = source_manifest['protocol']['splits']['train']; starts = data.starts_in(train_ids)
    demo_mask = np.isin(data.episode_ids, train_ids) & (data.is_expert > .5)
    demo_starts, branch_starts = starts[data.is_expert[starts] > .5], starts[data.is_expert[starts] < .5]
    data.to(device)
    torch.manual_seed(seed); np.random.seed(seed)
    model = StateDynamicsMLP().to(device)
    model.set_state_stats(data.states[demo_mask].mean(0), data.states[demo_mask].std(0))
    opt = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
    history, begin = [], time.time()
    for step in range(steps):
        torch.manual_seed(seed * 1000000 + step)
        _, aux_idx = sample_indices(demo_starts, branch_starts, c, seed, step, condition)
        b = next(data.batches(aux_idx, c['batch_size'], False))
        pred = model.rollout(b['state'], b['actions'])
        norm = lambda s: (s - model.state_mean) / model.state_std
        loss = F.mse_loss(norm(pred), norm(b['future_states']))
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at update {step + 1}')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            history.append({'step': step + 1, 'loss': float(loss.detach())})
    return model, {'history': history, 'seconds': time.time() - begin, 'steps': steps, 'condition_data': condition,
                   'demo_windows': int(len(demo_starts)), 'branch_windows': int(len(branch_starts))}
