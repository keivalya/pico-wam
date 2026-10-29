"""A tiny 2D point-mass world for learning WAMs on a laptop CPU.

Why a toy world?
    Meta-World is great, but it needs MuJoCo and minutes of rendering.
    This world renders in microseconds with numpy, and its true state is
    small and fully known, so we can check whether the WAM latent space
    really encodes "where things are" and "where they are going".

World:
    - a red agent (point mass with momentum) in the unit square
    - two goals: a blue square and a green square
    - the instruction picks the goal: "go to the blue goal" / "go to the green goal"

State (6-dim):
    [agent_x, agent_y, vel_x, vel_y, target_x, target_y]

    target_x/y is the goal named in the instruction. The other goal is a
    distractor that only appears in the image, so the policy must read the
    language to know which one to go to.

Action (2-dim):
    force in [-1, 1]^2

Dynamics:
    v_{t+1} = 0.8 * v_t + 0.01 * a_t
    p_{t+1} = clip(p_t + v_{t+1}, 0, 1)

Success = within 0.05 of the instructed goal AND (almost) stopped, so the
expert has to brake before arriving.

The momentum term makes the world model non-trivial: the next position
depends on velocity, which is NOT visible in a single image.
"""

import numpy as np


STATE_LABELS = ["agent_x", "agent_y", "vel_x", "vel_y", "target_x", "target_y"]

GOAL_COLORS = {
    "blue": np.array([40, 90, 230], dtype=np.uint8),
    "green": np.array([30, 170, 60], dtype=np.uint8),
}
AGENT_COLOR = np.array([220, 40, 40], dtype=np.uint8)
BACKGROUND = np.array([245, 245, 240], dtype=np.uint8)

# expert PD gains (on position error and normalized velocity)
KP, KD = 5.0, 2.0


class ToyPointEnv:
    """
    Same interface as MetaWorldMT1Wrapper:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    """

    def __init__(
        self,
        seed=42,
        image_size=64,
        damping=0.8,
        force_scale=0.01,
        success_radius=0.05,
        success_speed=0.2,
    ):
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.damping = damping
        self.force_scale = force_scale
        self.success_radius = success_radius
        self.success_speed = success_speed
        # terminal speed under a constant unit force; used to scale velocity to ~[-1, 1]
        self.max_speed = force_scale / (1.0 - damping)

        self.state_dim = len(STATE_LABELS)
        self.action_dim = 2
        self.state_labels = STATE_LABELS

        self.reset()
        self.obs_shape = self._render().shape

    # world -----------------------------------------------------------------

    def reset(self, seed=None, target=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.pos = self.rng.uniform(0.1, 0.9, size=2)
        self.vel = np.zeros(2)

        # two goals, far enough from each other and from the agent
        while True:
            goals = self.rng.uniform(0.1, 0.9, size=(2, 2))
            far_apart = np.linalg.norm(goals[0] - goals[1]) > 0.3
            far_from_agent = np.all(np.linalg.norm(goals - self.pos, axis=1) > 0.2)
            if far_apart and far_from_agent:
                break

        self.goals = {"blue": goals[0], "green": goals[1]}
        self.target = target if target is not None else self.rng.choice(["blue", "green"])
        self.instruction = f"go to the {self.target} goal"

        return self._render(), self._state(), self._info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        self.vel = self.damping * self.vel + self.force_scale * action
        self.pos = np.clip(self.pos + self.vel, 0.0, 1.0)

        dist = np.linalg.norm(self.pos - self.goals[self.target])
        reward = -dist
        info = self._info()
        done = bool(info["success"])

        return self._render(), self._state(), reward, done, info

    def close(self):
        pass

    def simulate(self, state, actions):
        """
        Ground-truth counterfactual: where would the agent go under `actions`
        starting from `state`? (Used to check the WAM's "what if" imagination.)

        state:   (6,) as returned by _state()
        actions: (H, 2)
        returns: (H, 6) states
        """
        pos = np.asarray(state[:2], dtype=np.float64).copy()
        vel = np.asarray(state[2:4], dtype=np.float64) * self.max_speed
        out = []
        for a in np.clip(actions, -1.0, 1.0):
            vel = self.damping * vel + self.force_scale * a
            pos = np.clip(pos + vel, 0.0, 1.0)
            out.append(np.concatenate([pos, vel / self.max_speed, state[4:6]]))
        return np.asarray(out, dtype=np.float32)

    def _state(self):
        return np.concatenate(
            [self.pos, self.vel / self.max_speed, self.goals[self.target]]
        ).astype(np.float32)

    def _info(self):
        dist = np.linalg.norm(self.pos - self.goals[self.target])
        speed = np.linalg.norm(self.vel) / self.max_speed
        success = dist < self.success_radius and speed < self.success_speed
        return {"success": int(success), "instruction": self.instruction}

    # expert ----------------------------------------------------------------

    def expert_action(self, noise=0.1):
        """PD controller toward the instructed goal (+ a bit of noise for diversity)."""
        err = self.goals[self.target] - self.pos
        action = KP * err - KD * self.vel / self.max_speed
        action = action + noise * self.rng.standard_normal(2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # rendering -------------------------------------------------------------

    def _render(self):
        n = self.image_size
        img = np.empty((n, n, 3), dtype=np.uint8)
        img[:] = BACKGROUND

        ys, xs = np.mgrid[0:n, 0:n]
        xs = (xs + 0.5) / n
        ys = 1.0 - (ys + 0.5) / n  # y axis points up

        half = 0.05
        for name, g in self.goals.items():
            mask = (np.abs(xs - g[0]) < half) & (np.abs(ys - g[1]) < half)
            img[mask] = GOAL_COLORS[name]

        mask = (xs - self.pos[0]) ** 2 + (ys - self.pos[1]) ** 2 < 0.045 ** 2
        img[mask] = AGENT_COLOR

        return img

    def render(self):
        return self._render()
