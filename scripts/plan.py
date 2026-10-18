"""Plan with the world model: model-predictive control in pico-WAM's latent space.

At every step the planner

    1. encodes the current observation into z_t
    2. imagines N candidate action sequences H steps ahead with the latent
       dynamics, starting from the policy's own imagined actions plus noise
    3. reads the predicted block pose out of every imagined latent and scores
       each sequence by how far the T would end up from the target
    4. executes the first action of the best sequence

and at the next step starts again from 1: every search begins from what the
policy would do, never from the last step's plan.

Why not reuse the last plan (warm_start=True)? The last plan is re-scored
and kept unless a candidate beats it, so over an episode the search piles
up hundreds of rounds and drifts away from the policy, toward actions whose
outcome the model gets wrong. On development episodes this is why a
longer-trained, more accurate world model planned worse: its chosen plans
expected to bring the T 74 px closer to the target in 12 steps, while the
simulator moved it 7 px further away.

Nothing here touches the simulator: every cost comes from the WAM's own
predictions, so the planner only works as well as the world model does.
A VLA has no dynamics to plan with.

The defaults keep the search close to the policy: one round of candidates
(iterations=1) with small noise. On development episodes (env seeds 90000+,
never used for testing) a stronger search, three cross-entropy rounds with
noise 0.3, picked actions whose imagined outcome was far better than the
real one: the planner exploits the model's errors.

    from scripts.plan import LatentPlanner
    planner = LatentPlanner(model)
    action = planner.act(image, text_ids, state)   # same signature as model.act
"""

import math

import numpy as np
import torch

from envs.pusht_env import AGENT_RADIUS, GOAL_POSE, T_COG, T_OUTLINE, WORKSPACE


class LatentPlanner:
    def __init__(self, model, horizon=12, samples=256, elites=32, iterations=1, noise=0.1,
                 approach_weight=0.05, warm_start=False, prior="policy", seed=0):
        if not (model.is_wam and model.predict_state):
            raise ValueError("Planning needs a WAM that predicts states (its dynamics and state readout).")
        self.model = model
        self.horizon, self.samples, self.elites, self.iterations = horizon, samples, elites, iterations
        self.noise, self.approach_weight, self.warm_start, self.prior = noise, approach_weight, warm_start, prior
        self.generator = torch.Generator(device=model.state_mean.device).manual_seed(seed)

        dev = model.state_mean.device
        self.corners = torch.tensor(T_OUTLINE, dtype=torch.float32, device=dev)   # (8, 2) block frame, px
        self.cog = torch.tensor(T_COG, dtype=torch.float32, device=dev)
        self.goal = self._corners(torch.tensor(GOAL_POSE, dtype=torch.float32, device=dev)[None])[0]
        self.previous = None

    def reset(self):
        self.previous = None

    # ------------------------------------------------------------------ cost

    def _pose(self, states):
        """Read-out states (..., 8), unit square with y up -> block pose (..., 3) in pixels, y down."""
        x = states[..., 4] * WORKSPACE
        y = (1 - states[..., 5]) * WORKSPACE
        angle = torch.atan2(-states[..., 7], states[..., 6])
        return torch.stack([x, y, angle], dim=-1)

    def _corners(self, pose):
        c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
        rot = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)  # (..., 2, 2)
        return torch.einsum("...ij,kj->...ki", rot, self.corners) + pose[..., None, :2]

    def cost(self, states):
        """
        states: (N, H, 8) imagined states. Lower is better:
            mean distance (px) between the T's corners and the target's, averaged
            over the imagined steps with more weight on the last ones, plus a
            small pull of the agent toward the T, so pushes are found even when
            the agent starts far from it.
        """
        pose = self._pose(states)
        error = (self._corners(pose) - self.goal).norm(dim=-1).mean(-1)              # (N, H)
        weights = torch.linspace(0.5, 1.5, error.shape[1], device=error.device)
        c, s = torch.cos(pose[..., 2]), torch.sin(pose[..., 2])
        center = pose[..., :2] + torch.stack([c * self.cog[0] - s * self.cog[1], s * self.cog[0] + c * self.cog[1]], -1)
        agent = torch.stack([states[..., 0] * WORKSPACE, (1 - states[..., 1]) * WORKSPACE], -1)
        gap = ((agent - center).norm(dim=-1) - AGENT_RADIUS).clamp_min(0)              # (N, H)
        return (error * weights).mean(-1) + self.approach_weight * gap.mean(-1)

    # ------------------------------------------------------------------ plan

    @torch.no_grad()
    def act(self, image, text_ids, state):
        model, H, N = self.model, self.horizon, self.samples
        z0 = model.encode_obs(image, text_ids, state)                                   # (1, D)

        # the policy's own plan, imagined: a_k = pi(z_k); or, without a policy, standing still
        if self.prior == "policy":
            prior = model.imagine(image, text_ids, state, H)["actions"][0]              # (H, 2)
        else:
            prior = torch.zeros((H, model.config["action_dim"]), device=z0.device)
        mean = prior.clone()
        if self.warm_start and self.previous is not None:  # start from the last plan, shifted by one step
            mean = torch.cat([self.previous[1:], prior[-1:]], 0)
        std = torch.full_like(mean, self.noise)

        best_seq, best_cost = None, math.inf
        for _ in range(self.iterations):
            eps = torch.randn((N, H, mean.shape[-1]), generator=self.generator, device=mean.device)
            seqs = (mean + std * eps).clamp(-1.0, 1.0)
            seqs[0], seqs[1] = mean, prior                     # always score the current mean and the policy
            z = model.dynamics.rollout(z0.expand(N, -1), seqs)                          # (N, H, D)
            costs = self.cost(model.denormalize_state(model.state_readout(z)))
            order = torch.argsort(costs)
            if costs[order[0]] < best_cost:
                best_cost, best_seq = float(costs[order[0]]), seqs[order[0]].clone()
            elite = seqs[order[: self.elites]]
            mean, std = elite.mean(0), elite.std(0).clamp_min(0.05)

        self.previous = best_seq
        self.last = {"z0": z0, "plan": best_seq, "cost": best_cost}
        return best_seq[:1]                                                              # (1, 2), like model.act

    @torch.no_grad()
    def check_last_plan(self, sim, state):
        """
        Compare the last chosen plan's imagined outcome with the simulator's, from
        the same state: block-pose error between them, and the improvement in
        block-pose error (both in pixels) that the model expected vs the real one.
        """
        from envs.pusht_env import from_state
        from envs.pusht_expert import pose_error

        plan = self.last["plan"]
        z = self.model.dynamics.rollout(self.last["z0"], plan[None])
        predicted = self.model.denormalize_state(self.model.state_readout(z))[0, -1].cpu().numpy()
        true = sim.simulate(state, plan.cpu().numpy())[-1]
        now = pose_error(from_state(state)[4:7])
        return {
            "plan_error_px": pose_error(from_state(predicted)[4:7], from_state(true)[4:7]),
            "expected_gain_px": now - pose_error(from_state(predicted)[4:7]),
            "real_gain_px": now - pose_error(from_state(true)[4:7]),
        }
