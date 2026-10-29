"""Study 3: competent imitation baselines on the evidence study's data, splits and start banks.

history_bc        BC's exact observation encoder (image, instruction and state, from PicoWAMPolicy)
                  applied to the last h frames, concatenated, and an MLP head predicting K actions;
                  K // 2 are executed. Hyperparameters come from a predeclared grid scored on a
                  separate tuning bank before any final training (see `tune` below).
diffusion_policy  Chi et al.'s state-based Push-T recipe: their ConditionalUnet1D (vendored, unmodified),
                  diffusers' DDPMScheduler with their settings, their EMA, AdamW and cosine schedule,
                  limits normalization, padded 16-step windows, two observation steps, and 8 executed
                  actions starting at the current step. Adapter: the observation is this simulator's
                  8-dimensional state, and actions are its relative displacements in [-1, 1]. The
                  final EMA weights are evaluated; the original reports the best checkpoint
                  seen during training, which a locked protocol cannot do.

Both export TRAINERS and AGENT_BUILDERS, which followup/study.py merges into its registry.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from followup.common import instruction_tokens
from scripts.evidence_protocol import digest, write_json


def demo_index(src, device):
    """Training-split demonstrations of the source study as tensors, plus per-episode bounds."""
    from scripts.train import TrainingDataset
    sm = src[1]
    data = TrainingDataset(sm['datasets']['demos']['path'], horizon=1, action_chunk=1)
    train_ids = np.asarray(sm['protocol']['splits']['train'])
    keep = np.nonzero(np.isin(data.episode_ids, train_ids))[0]
    ep = data.episode_ids
    first = np.zeros(len(ep), dtype=np.int64)
    for i in range(1, len(ep)):
        first[i] = first[i - 1] if ep[i] == ep[i - 1] else i
    data.to(device)
    return data, keep, torch.as_tensor(first, device=device), data.t_episode_last


def window(idx, offsets, first, last):
    """Indices idx + offsets, held at the episode's first and last transitions, as Chi et al. pad."""
    w = idx[:, None] + offsets[None]
    return torch.maximum(torch.minimum(w, last[idx][:, None]), first[idx][:, None])


# ------------------------------------------------------------------ history-based BC


class HistoryBC(nn.Module):
    def __init__(self, vocab_size, history, chunk, width, depth, dropout=0.0, d_model=128):
        super().__init__()
        from models.wam_policy import PicoWAMPolicy
        # Identical encoders to BC; only encode_obs and the state statistics are used.
        self.encoder = PicoWAMPolicy(vocab_size=vocab_size, state_dim=8, action_dim=2, d_model=d_model, model_type='vla',
                                     predict_state=False, predict_image=False, use_state_input=True, action_chunk=1)
        layers, w = [], history * d_model
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.ReLU()] + ([nn.Dropout(dropout)] if dropout else []); w = width
        self.head = nn.Sequential(*layers, nn.Linear(w, chunk * 2))
        self.history, self.chunk = history, chunk
        self.config = dict(vocab_size=vocab_size, history=history, chunk=chunk, width=width, depth=depth, dropout=dropout, d_model=d_model)

    def forward(self, images, tokens, states):
        """(B, h, 3, H, W), (B, T), (B, h, 8) -> (B, K, 2); index -1 is the current frame."""
        z = self.encoder.encode_sequence(images, tokens, states)
        return self.head(z.flatten(1)).view(-1, self.chunk, 2)


def train_history_bc(src, hp, seed, steps, batch_size, lr, device, log_every=1000):
    torch.manual_seed(seed); np.random.seed(seed)
    data, keep, first, last = demo_index(src, device)
    vocab = data.vocab
    model = HistoryBC(max(vocab.values()) + 1, hp['history'], hp['chunk'], hp['width'], hp['depth'], hp.get('dropout', 0.0)).to(device)
    train_states = data.states[np.isin(np.arange(len(data.states)), keep)]
    model.encoder.set_state_stats(train_states.mean(0), train_states.std(0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs_off = torch.arange(-hp['history'] + 1, 1, device=device); act_off = torch.arange(hp['chunk'], device=device)
    keep_t = torch.as_tensor(keep, device=device); history, begin = [], time.time()
    model.train()
    for step in range(steps):
        g = torch.Generator(device='cpu').manual_seed(seed * 1000000 + step)
        idx = keep_t[torch.randint(len(keep_t), (batch_size,), generator=g).to(device)]
        oi, ai = window(idx, obs_off, first, last), window(idx, act_off, first, last)
        images = data.t_images[oi].float() / 255.0
        pred = model(images, data.t_text_ids[idx], data.t_states[oi])
        loss = F.mse_loss(pred, data.t_actions[ai])
        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at update {step + 1}')
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % log_every == 0 or step + 1 == steps:
            history.append({'step': step + 1, 'loss': float(loss.detach())})
    model.eval()
    return model, vocab, {'history': history, 'seconds': time.time() - begin, 'steps': steps, 'hp': hp}


class HistoryAgent:
    """Predicts a chunk from the last h observations and executes `execute` of its actions."""
    plans = False

    def __init__(self, model, execute):
        self.model, self.execute = model, execute

    def reset(self, episode_seed):
        torch.manual_seed(episode_seed); self.queue, self.frames = [], []

    def act(self, image, tokens, state):
        self.frames = (self.frames + [(image, state)])[-self.model.history:]
        if not self.queue:
            frames = [self.frames[0]] * (self.model.history - len(self.frames)) + self.frames  # repeat the first frame
            images = torch.stack([f[0][0] for f in frames])[None]; states = torch.stack([f[1][0] for f in frames])[None]
            self.queue = list(self.model(images, tokens, states)[0, :self.execute].cpu().numpy())
        return self.queue.pop(0)


def _save(out, payload):
    torch.save(payload, out / 'model.tmp'); (out / 'model.tmp').replace(out / 'model.pt')


def trainer_history_bc(path, m, src, spec, seed, out, device):
    tuning = path / 'tuning' / 'selection.json'
    if not tuning.exists():
        raise ValueError('Run the predeclared tuning first: python -m followup.baselines tune.')
    sel = json.loads(tuning.read_text())
    model, vocab, info = train_history_bc(src, sel['selected'], seed, spec['steps'], spec['batch_size'], sel['selected']['lr'], device)
    _save(out, {'state_dict': model.state_dict(), 'config': model.config, 'vocab': vocab, 'info': info, 'selection_sha256': digest(tuning)})
    return info


def agent_history_bc(m, src, spec, seed, control, out, device):
    saved = torch.load(out / 'model.pt', map_location=device, weights_only=False)
    model = HistoryBC(**saved['config']).to(device); model.load_state_dict(saved['state_dict']); model.eval()
    return HistoryAgent(model, max(1, saved['config']['chunk'] // 2)), instruction_tokens(saved['vocab'], device), {'checkpoint_sha256': digest(out / 'model.pt')}


def tune(args):
    """Predeclared history-BC grid: one training seed, its own tuning bank, select by mean coverage."""
    from followup.common import run_episodes
    from followup.study import load
    path, m, src = load(args.study); c = m['config']; t = c['tuning']
    grid = [dict(zip(t['grid'], v)) for v in itertools.product(*t['grid'].values())]
    if not 0 <= args.index < len(grid):
        raise ValueError(f'Grid has {len(grid)} configurations.')
    out = path / 'tuning'; out.mkdir(exist_ok=True)
    target = out / f'config-{args.index}.json'
    if target.exists(): print('Already tuned; preserved.'); return
    device = torch.device(args.device); hp = {**grid[args.index], 'depth': grid[args.index].get('depth', 3)}
    model, vocab, info = train_history_bc(src, hp, t['seed'], c['methods']['history_bc']['steps'], c['methods']['history_bc']['batch_size'], hp['lr'], device)
    seeds = list(range(t['bank_start'], t['bank_start'] + t['bank_episodes']))
    r = run_episodes(HistoryAgent(model, max(1, hp['chunk'] // 2)), instruction_tokens(vocab, device), seeds, c['max_steps'], c['check_plans_every'], device)
    write_json(target, {'index': args.index, 'hp': hp, 'mean_max_coverage': r['mean_max_coverage'], 'success_rate': r['success_rate'],
                        'episode_seeds': seeds, 'train': info, 'manifest_sha256': digest(path / 'manifest.json')})
    print('Tuned', args.index, hp, f'{r["mean_max_coverage"]:.3f}')


def select(args):
    """Fix the history-BC configuration: highest tuning coverage, then success, then fewer parameters."""
    from followup.study import load
    path, m, src = load(args.study); t = m['config']['tuning']
    n = math.prod(len(v) for v in t['grid'].values())
    rows = [json.loads((path / 'tuning' / f'config-{i}.json').read_text()) for i in range(n)]
    size = lambda hp: hp['width'] * hp.get('depth', 3) * hp['history']
    best = max(rows, key=lambda r: (r['mean_max_coverage'], r['success_rate'], -size(r['hp'])))
    target = path / 'tuning' / 'selection.json'
    if target.exists(): raise FileExistsError('Selection already fixed.')
    write_json(target, {'selected': best['hp'], 'rule': 'max tuning coverage, then success, then smaller head',
                        'all': [{k: r[k] for k in ('index', 'hp', 'mean_max_coverage', 'success_rate')} for r in rows],
                        'files_sha256': {f'config-{i}.json': digest(path / 'tuning' / f'config-{i}.json') for i in range(n)}})
    print('Selected', best['hp'], f'{best["mean_max_coverage"]:.3f}')


# ------------------------------------------------------------------ Diffusion Policy


class LimitsNormalizer:
    """Chi et al.'s LinearNormalizer in 'limits' mode: each dimension's [min, max] maps to [-1, 1]."""

    def __init__(self, x, range_eps=1e-4):
        lo, hi = x.min(0).values, x.max(0).values; rng = hi - lo; flat = rng < range_eps
        rng = torch.where(flat, torch.full_like(rng, 2.0), rng)
        self.scale = 2.0 / rng; self.offset = -1.0 - self.scale * lo
        self.offset = torch.where(flat, -lo, self.offset)

    def state_dict(self):
        return {'scale': self.scale, 'offset': self.offset}

    @classmethod
    def from_state(cls, d):
        n = cls.__new__(cls); n.scale, n.offset = d['scale'], d['offset']; return n

    def to(self, device):
        self.scale, self.offset = self.scale.to(device), self.offset.to(device); return self

    def normalize(self, x): return x * self.scale + self.offset
    def unnormalize(self, x): return (x - self.offset) / self.scale


def dp_components(spec, device):
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from followup.vendor.diffusion_policy.conditional_unet1d import ConditionalUnet1D
    r = spec['recipe']
    net = ConditionalUnet1D(input_dim=2, global_cond_dim=8 * r['n_obs_steps'], diffusion_step_embed_dim=r['diffusion_step_embed_dim'],
                            down_dims=r['down_dims'], kernel_size=r['kernel_size'], n_groups=r['n_groups'],
                            cond_predict_scale=r['cond_predict_scale']).to(device)
    sched = DDPMScheduler(num_train_timesteps=r['num_train_timesteps'], beta_start=r['beta_start'], beta_end=r['beta_end'],
                          beta_schedule=r['beta_schedule'], variance_type=r['variance_type'], clip_sample=r['clip_sample'],
                          prediction_type=r['prediction_type'])
    return net, sched


def dp_windows(src, horizon, n_obs, n_action, device):
    """Chi et al.'s SequenceSampler: windows padded by n_obs - 1 before and n_action - 1 after each episode."""
    data, keep, first, last = demo_index(src, device)
    ep = data.episode_ids; starts = []
    for e in np.unique(ep[keep]):
        idx = np.nonzero(ep == e)[0]; a, b = idx[0], idx[-1]
        starts += list(range(a - (n_obs - 1), b - horizon + n_action + 1))
    return data, torch.as_tensor(np.asarray(starts), device=device), first, last


def train_diffusion_policy(src, spec, seed, device, max_steps=None, log_every=1000, checkpoint=None, checkpoint_every=50):
    from diffusers.optimization import get_scheduler
    from followup.vendor.diffusion_policy.ema_model import EMAModel
    import copy
    r = spec['recipe']; torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cudnn.benchmark = True  # autotuned kernels: speed only, same computation
    data, starts, first, last = dp_windows(src, r['horizon'], r['n_obs_steps'], r['n_action_steps'], device)
    offs = torch.arange(r['horizon'], device=device)
    # Every window index, held inside its own episode: the first valid index anchors the bounds.
    anchor = torch.clamp(starts + r['n_obs_steps'] - 1, min=0)
    def idx_of(s):
        a = anchor[s]; w = starts[s][:, None] + offs[None]
        return torch.maximum(torch.minimum(w, last[a][:, None]), first[a][:, None])
    all_idx = idx_of(torch.arange(len(starts), device=device))
    obs_all = data.t_states[all_idx[:, :r['n_obs_steps']]].reshape(-1, 8); act_all = data.t_actions[all_idx].reshape(-1, 2)
    norm_obs, norm_act = LimitsNormalizer(obs_all), LimitsNormalizer(act_all)
    net, sched = dp_components(spec, device)
    ema_net = copy.deepcopy(net); ema = EMAModel(ema_net, update_after_step=0, inv_gamma=1.0, power=r['ema_power'], min_value=0.0, max_value=0.9999)
    opt = torch.optim.AdamW(net.parameters(), lr=r['lr'], betas=tuple(r['betas']), eps=r['eps'], weight_decay=r['weight_decay'])
    per_epoch = math.ceil(len(starts) / r['batch_size']); total = r['num_epochs'] * per_epoch
    steps = min(total, max_steps) if max_steps else total
    lr_sched = get_scheduler(r['lr_scheduler'], optimizer=opt, num_warmup_steps=r['lr_warmup_steps'], num_training_steps=total)
    history, begin, step, first_epoch, resumed = [], time.time(), 0, 0, False
    if checkpoint is not None and Path(checkpoint).exists():
        # Resume exactly: weights, EMA, optimizer, schedule, counters and both random streams.
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(ck['net']); ema_net.load_state_dict(ck['ema_net']); opt.load_state_dict(ck['opt'])
        lr_sched.load_state_dict(ck['lr_sched']); ema.optimization_step, ema.decay = ck['ema_step'], ck['ema_decay']
        step, first_epoch, history = ck['step'], ck['epoch'] + 1, ck['history']
        torch.set_rng_state(ck['cpu_rng'])
        if ck['cuda_rng'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ck['cuda_rng'])
        resumed = True
    net.train()
    for epoch in range(first_epoch, r['num_epochs']):
        g = torch.Generator(device='cpu').manual_seed(seed * 100000 + epoch)
        perm = torch.randperm(len(starts), generator=g).to(device)
        for k in range(per_epoch):
            if step >= steps: break
            ids = all_idx[perm[k * r['batch_size']:(k + 1) * r['batch_size']]]
            obs = norm_obs.normalize(data.t_states[ids[:, :r['n_obs_steps']]]).flatten(1)
            traj = norm_act.normalize(data.t_actions[ids])
            noise = torch.randn_like(traj)
            t = torch.randint(0, sched.config.num_train_timesteps, (len(traj),), device=device).long()
            loss = F.mse_loss(net(sched.add_noise(traj, noise, t), t, global_cond=obs), noise)
            if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at update {step + 1}')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); lr_sched.step(); ema.step(net)
            step += 1
            if step % log_every == 0 or step == steps:
                history.append({'step': step, 'epoch': epoch, 'loss': float(loss.detach()), 'lr': lr_sched.get_last_lr()[0]})
        if checkpoint is not None and (epoch + 1) % checkpoint_every == 0 and step < steps:
            tmp = Path(checkpoint).with_suffix('.tmp')
            torch.save({'net': net.state_dict(), 'ema_net': ema_net.state_dict(), 'opt': opt.state_dict(), 'lr_sched': lr_sched.state_dict(),
                        'ema_step': ema.optimization_step, 'ema_decay': ema.decay, 'step': step, 'epoch': epoch, 'history': history,
                        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
            tmp.replace(checkpoint)
        if step >= steps: break
    return ema_net.eval(), norm_obs, norm_act, {'history': history, 'seconds_this_invocation': time.time() - begin, 'steps': step, 'resumed': resumed,
                                                 'planned_steps': total, 'windows': int(len(starts)), 'params': sum(p.numel() for p in net.parameters())}


class DiffusionAgent:
    """Two observation steps in, 16 actions predicted, 8 executed from the current step, as Chi et al."""
    plans = False

    def __init__(self, net, sched, norm_obs, norm_act, recipe, device):
        self.net, self.sched, self.no, self.na, self.r, self.device = net, sched, norm_obs, norm_act, recipe, device

    def reset(self, episode_seed):
        torch.manual_seed(episode_seed); self.queue, self.states = [], []
        self.generator = torch.Generator(device=self.device).manual_seed(episode_seed + 654321)

    @torch.no_grad()
    def act(self, image, tokens, state):
        To = self.r['n_obs_steps']
        self.states = (self.states + [state[0]])[-To:]
        if not self.queue:
            obs = torch.stack([self.states[0]] * (To - len(self.states)) + self.states)[None]
            cond = self.no.normalize(obs).flatten(1)
            traj = torch.randn((1, self.r['horizon'], 2), generator=self.generator, device=self.device)
            self.sched.set_timesteps(self.r['num_inference_steps'])
            for t in self.sched.timesteps:
                traj = self.sched.step(self.net(traj, t, global_cond=cond), t, traj, generator=self.generator).prev_sample
            act = self.na.unnormalize(traj)[0]
            start = To - 1  # oa_step_convention
            self.queue = list(act[start:start + self.r['n_action_steps']].cpu().numpy())
        return self.queue.pop(0)


def trainer_diffusion_policy(path, m, src, spec, seed, out, device):
    partial = out / 'partial.pt'
    net, no, na, info = train_diffusion_policy(src, spec, seed, device, checkpoint=partial)
    _save(out, {'ema_state_dict': net.state_dict(), 'norm_obs': no.state_dict(), 'norm_act': na.state_dict(), 'info': info})
    partial.unlink(missing_ok=True)  # resume state only; the final EMA weights are model.pt
    return info


def agent_diffusion_policy(m, src, spec, seed, control, out, device):
    from models.wam_policy import PicoWAMPolicy  # noqa: F401  (tokens come from the source study's vocabulary)
    saved = torch.load(out / 'model.pt', map_location=device, weights_only=False)
    net, sched = dp_components(spec, device); net.load_state_dict(saved['ema_state_dict']); net.eval()
    ck = torch.load(src[0] / 'runs' / f'seed-{seed}' / 'bc' / 'model.pt', map_location='cpu', weights_only=False)
    agent = DiffusionAgent(net, sched, LimitsNormalizer.from_state(saved['norm_obs']).to(device),
                           LimitsNormalizer.from_state(saved['norm_act']).to(device), spec['recipe'], device)
    return agent, instruction_tokens(ck['vocab'], device), {'checkpoint_sha256': digest(out / 'model.pt')}


TRAINERS = {'history_bc': trainer_history_bc, 'diffusion_policy': trainer_diffusion_policy}
AGENT_BUILDERS = {'history_bc': agent_history_bc, 'diffusion_policy': agent_diffusion_policy}


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('tune'); a.add_argument('--study', required=True); a.add_argument('--index', type=int, required=True)
    a.add_argument('--device', default='cpu')
    a = sub.add_parser('select'); a.add_argument('--study', required=True)
    args = p.parse_args()
    {'tune': tune, 'select': select}[args.command](args)


if __name__ == '__main__':
    main()
