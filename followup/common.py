"""Shared pieces of the follow-up studies: the episode runner, agents and a pluggable planner.

run_episodes() is the evidence study's evaluate_control() with the policy or planner behind an
agent interface, so new methods are scored on exactly the same episodes, step limit, coverage
definition and plan-check schedule. tests/test_followup.py checks that it reproduces
evaluate_control() exactly for the evidence study's own checkpoints.
"""
import math
import time

import numpy as np
import torch

from envs.pusht_env import PushTEnv
from scripts.plan import LatentPlanner
from scripts.test import to_tensor
from utils.tokenizer import SimpleTokenizer


def render_tensor(env, state, device):
    """The model's image input for any state: the env's own renderer, as in closed loop."""
    return to_tensor(env._render(np.asarray(state, dtype=np.float32)), 64, device)


def instruction_tokens(vocab, device):
    tok = SimpleTokenizer(vocab=vocab)
    return torch.tensor([tok.encode(PushTEnv.instruction)], device=device)


# ------------------------------------------------------------------ agents
#
# An agent sees what evaluate_control's policy or planner saw: the image and state tensors and
# the instruction tokens. reset() runs before each episode with that episode's seed.


class ChunkPolicyAgent:
    """A PicoWAMPolicy acting reactively: predict K actions, execute K // 2, as in the evidence study."""
    plans = False

    def __init__(self, model, execute=None):
        self.model = model
        self.execute = execute or max(1, model.config['action_chunk'] // 2)

    def reset(self, episode_seed):
        torch.manual_seed(episode_seed); self.queue = []

    def act(self, image, tokens, state):
        if not self.queue:
            self.queue = list(self.model.act_chunk(image, tokens, state)[0, :self.execute].cpu().numpy())
        return self.queue.pop(0)


class PlannerAgent:
    """Any planner with act(), check_last_plan(), reset() and a torch generator."""
    plans = True

    def __init__(self, planner):
        self.planner = planner

    def reset(self, episode_seed):
        torch.manual_seed(episode_seed)
        self.planner.reset(); self.planner.generator.manual_seed(episode_seed + 123456)

    def act(self, image, tokens, state):
        return self.planner.act(image, tokens, state)[0].cpu().numpy()

    def check(self, sim, state):
        return self.planner.check_last_plan(sim, state)


def run_episodes(agent, tokens, episode_seeds, max_steps, check_every, device):
    """evaluate_control() with an agent; returns the same fields."""
    env, sim = PushTEnv(image_size=64), PushTEnv(image_size=64)
    success_flags, coverages, lengths, checks = [], [], [], []
    begin = time.time()
    with torch.no_grad():
        for episode_seed in episode_seeds:
            agent.reset(episode_seed)
            image, state, info = env.reset(seed=episode_seed)
            best = float(info.get('coverage', 0)); success = False; episode_checks = []
            for step in range(max_steps):
                im = to_tensor(image, 64, device); st = torch.as_tensor(state, dtype=torch.float32, device=device)[None]
                action = agent.act(im, tokens, st)
                if agent.plans and step % check_every == 0:
                    episode_checks.append({'step': step, **agent.check(sim, np.asarray(state))})
                image, state, _, done, info = env.step(action)
                best = max(best, float(info['coverage'])); success = success or bool(info.get('success', False))
                if done or success: break
            success_flags.append(success); coverages.append(best); lengths.append(step + 1)
            checks.append({'episode_seed': episode_seed, 'checks': episode_checks})
    return {'episode_seeds': list(episode_seeds), 'success': success_flags, 'max_coverage': coverages, 'lengths': lengths,
            'success_rate': float(np.mean(success_flags)), 'mean_max_coverage': float(np.mean(coverages)),
            'plan_checks_by_episode': checks, 'wall_seconds': time.time() - begin,
            'coverage_definition': 'maximum over the reset state and every post-action state of the episode'}


# ------------------------------------------------------------------ planner


class _CostOnly:
    """Gives LatentPlanner's cost function a device without a model."""
    is_wam = predict_state = True

    def __init__(self, device):
        self.state_mean = torch.zeros(8, device=device)


class GeneralPlanner:
    """scripts/plan.py's search with the dynamics behind two functions.

        prior_fn(image, tokens, state)         -> (H, 2) the policy's plan inside the dynamics
        predict_fn(image, tokens, state, seqs) -> (N, H, 8) predicted states for N sequences

    With the latent dynamics of a PicoWAMPolicy this is LatentPlanner, step for step and draw for
    draw; tests/test_followup.py checks that. Swapping the two functions swaps the world model
    while the policy, cost, candidates and search stay fixed.
    """

    def __init__(self, prior_fn, predict_fn, device, horizon=12, samples=256, elites=32, iterations=1, noise=0.1,
                 approach_weight=0.05, warm_start=False, prior='policy', seed=0):
        if prior != 'policy':
            raise ValueError('Only the policy prior is used in the follow-ups.')
        self.prior_fn, self.predict_fn = prior_fn, predict_fn
        self.horizon, self.samples, self.elites, self.iterations = horizon, samples, elites, iterations
        self.noise, self.warm_start = noise, warm_start
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.cost = LatentPlanner(_CostOnly(device), approach_weight=approach_weight).cost
        self.previous = None

    def reset(self):
        self.previous = None

    @torch.no_grad()
    def act(self, image, tokens, state):
        H, N = self.horizon, self.samples
        prior = self.prior_fn(image, tokens, state)
        mean = prior.clone()
        if self.warm_start and self.previous is not None:
            mean = torch.cat([self.previous[1:], prior[-1:]], 0)
        std = torch.full_like(mean, self.noise)
        best_seq, best_cost = None, math.inf
        for _ in range(self.iterations):
            eps = torch.randn((N, H, mean.shape[-1]), generator=self.generator, device=mean.device)
            seqs = (mean + std * eps).clamp(-1.0, 1.0)
            seqs[0], seqs[1] = mean, prior
            costs = self.cost(self.predict_fn(image, tokens, state, seqs))
            order = torch.argsort(costs)
            if costs[order[0]] < best_cost:
                best_cost, best_seq = float(costs[order[0]]), seqs[order[0]].clone()
            elite = seqs[order[: self.elites]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.05)
        self.previous = best_seq
        self.last = {'inputs': (image, tokens, state), 'plan': best_seq, 'cost': best_cost}
        return best_seq[:1]

    @torch.no_grad()
    def check_last_plan(self, sim, state):
        from envs.pusht_env import from_state
        from envs.pusht_expert import pose_error
        plan = self.last['plan']
        predicted = self.predict_fn(*self.last['inputs'], plan[None])[0, -1].cpu().numpy()
        true = sim.simulate(state, plan.cpu().numpy())[-1]
        now = pose_error(from_state(state)[4:7])
        return {'plan_error_px': pose_error(from_state(predicted)[4:7], from_state(true)[4:7]),
                'expected_gain_px': now - pose_error(from_state(predicted)[4:7]),
                'real_gain_px': now - pose_error(from_state(true)[4:7])}


# ------------------------------------------------------------------ dynamics behind the planner


def latent_functions(model, horizon):
    """The PicoWAMPolicy's own latent imagination and rollout, as LatentPlanner uses them."""
    def prior_fn(image, tokens, state):
        return model.imagine(image, tokens, state, horizon)['actions'][0]

    def predict_fn(image, tokens, state, seqs):
        z0 = model.encode_obs(image, tokens, state)
        return model.denormalize_state(model.state_readout(model.dynamics.rollout(z0.expand(len(seqs), -1), seqs)))
    return prior_fn, predict_fn


def first_policy_action(model, env, tokens, state, device):
    """The reactive policy's first action for a state, seen through a rendered image."""
    st = torch.as_tensor(state, dtype=torch.float32, device=device)[None]
    return model.policy_head.sample(model.encode_obs(render_tensor(env, state, device), tokens, st))[:, :model.action_dim]


def state_model_functions(policy_model, state_model, horizon, device):
    """A learned state-space model replaces the latent dynamics; the policy prior acts inside it."""
    env = PushTEnv(image_size=64)

    def prior_fn(image, tokens, state):
        s, acts = state, []
        for _ in range(horizon):
            a = first_policy_action(policy_model, env, tokens, s[0].cpu().numpy(), device)
            acts.append(a[0]); s = state_model.step(s, a)
        return torch.stack(acts)

    def predict_fn(image, tokens, state, seqs):
        return state_model.rollout(state.expand(len(seqs), -1), seqs)
    return prior_fn, predict_fn


def oracle_functions(policy_model, horizon, device):
    """Privileged diagnostic: the true simulator scores candidates and imagines the prior."""
    env = PushTEnv(image_size=64)

    def prior_fn(image, tokens, state):
        s, acts = state[0].cpu().numpy(), []
        for _ in range(horizon):
            a = first_policy_action(policy_model, env, tokens, s, device)
            acts.append(a[0]); s = env.simulate(s, a.cpu().numpy())[-1]
        return torch.stack(acts)

    def predict_fn(image, tokens, state, seqs):
        s0 = state[0].cpu().numpy()
        return torch.as_tensor(np.stack([env.simulate(s0, q) for q in seqs.cpu().numpy()]), device=device)
    return prior_fn, predict_fn
