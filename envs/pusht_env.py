"""Push-T: push a T-shaped block onto a fixed target pose.

The task from Diffusion Policy (Chi et al., RSS 2023), rebuilt with the same
physics constants so it runs headless with numpy, pymunk and OpenCV, and so
it replays the human demonstrations of LeRobot's Push-T dataset (see
scripts/download_pusht.py):

    - a blue disk (the agent) moves in a 512 x 512 workspace, tracking a
      target point with a PD controller (k_p = 100, k_v = 20) at 100 Hz
    - a grey T-shaped block slides when the agent pushes it (frictionless
      contacts, no damping memory: the block moves only while it is pushed)
    - a green T marks the goal pose; success = the block covers > 95% of it

Two changes make it a better fit for a small world model:

    1. An action moves the PD target relative to the agent: a in [-1, 1]^2
       puts the target up to STEP pixels away, instead of at an absolute
       position. A random action is then a local nudge, and "what if I
       pushed left?" is a meaningful question. Recorded absolute targets
       convert exactly unless they are more than STEP pixels away on an
       axis (29 of the 25,650 human demonstration frames).
    2. The block's velocity is reset at every control step (10 Hz). With the
       original zero damping it only carries over one 10 ms physics substep
       anyway, and resetting it makes the observed state a complete
       description of the world: simulate(state, actions) reproduces step()
       exactly, so every prediction can be checked against the simulator.
       From a recorded state it predicts the human demonstrations' next
       block pose to within 0.6 px in 99% of steps.

State (8-dim), positions in the unit square with y pointing up (as in the image):
    [agent_x, agent_y, agent_vx, agent_vy, block_x, block_y, block_cos, block_sin]

    block_x/y is the middle of the T's top bar edge and the angle is stored
    as cos/sin so it never wraps. Agent velocity (unit squares per second) is
    not visible in a single image.

Action (2-dim):
    displacement of the PD target in [-1, 1]^2 (x right, y up), times STEP pixels
"""

import numpy as np
import pymunk


STATE_LABELS = ["agent_x", "agent_y", "agent_vx", "agent_vy", "block_x", "block_y", "block_cos", "block_sin"]

# physics, in pixels with y pointing down (the original's coordinates)
WORKSPACE = 512.0
AGENT_RADIUS = 15.0
SIM_HZ, CONTROL_HZ = 100, 10
KP, KV = 100.0, 20.0
STEP = 100.0
GOAL_POSE = (256.0, 256.0, np.pi / 4)  # x, y, angle
SUCCESS_COVERAGE = 0.95

# the T as two rectangles in block coordinates; the origin is the middle of
# the bar's top edge and the stem points down at angle 0
T_PARTS = (
    ((-60.0, 30.0), (60.0, 30.0), (60.0, 0.0), (-60.0, 0.0)),
    ((-15.0, 30.0), (-15.0, 120.0), (15.0, 120.0), (15.0, 30.0)),
)
T_OUTLINE = np.array(
    [(-60, 0), (60, 0), (60, 30), (15, 30), (15, 120), (-15, 120), (-15, 30), (-60, 30)], dtype=np.float64
)
T_AREA = 120.0 * 30.0 + 30.0 * 90.0
T_COG = (0.0, 45.0)                                   # mean of the two centroids, as in the original
T_MOMENT = 2 * pymunk.moment_for_poly(1.0, T_PARTS[0])  # as in the original
WALLS = (((5, 506), (5, 5)), ((5, 5), (506, 5)), ((506, 5), (506, 506)), ((5, 506), (506, 506)))
TARGET_LIMITS = (5 + 2 + AGENT_RADIUS, 506 - 2 - AGENT_RADIUS)  # where the agent fits inside the walls

# colors as they appear in the LeRobot dataset's frames
BACKGROUND = (255, 255, 255)
WALL_COLOR = (211, 211, 211)
GOAL_COLOR = (144, 238, 144)
BLOCK_COLOR = (144, 163, 182)
AGENT_COLOR = (82, 124, 252)
DATASET_SIZE = 96  # the dataset's frames: a 512 px canvas resized to 96 x 96


# ------------------------------------------------------------------ geometry


def rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]])


def to_world(points, pose):
    """Block-frame points (N, 2) -> world pixels for a pose (x, y, angle)."""
    return np.asarray(points) @ rotation(pose[2]).T + np.asarray(pose[:2])


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon(subject, clip):
    """Sutherland-Hodgman: the part of convex polygon `subject` inside convex polygon `clip`."""
    sign = np.sign(cross(clip[1] - clip[0], clip[2] - clip[1]))  # orientation of `clip`
    out = list(subject)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        inside = lambda p: sign * cross(b - a, p - a) >= 0
        pts, out = out, []
        for p, q in zip(pts, pts[1:] + pts[:1]):
            if inside(q):
                if not inside(p):
                    out.append(intersect(p, q, a, b))
                out.append(q)
            elif inside(p):
                out.append(intersect(p, q, a, b))
        if not out:
            break
    return np.array(out)


def intersect(p, q, a, b):
    d1, d2 = q - p, b - a
    t = cross(a - p, d2) / cross(d1, d2)
    return p + t * d1


def cross(u, v):
    return u[0] * v[1] - u[1] * v[0]


def coverage(pose, goal=GOAL_POSE):
    """Fraction of the goal T covered by the block (exact polygon intersection)."""
    area = 0.0
    for part in T_PARTS:
        block = to_world(part, pose)
        for goal_part in T_PARTS:
            inter = clip_polygon(block, to_world(goal_part, goal))
            if len(inter) >= 3:
                area += polygon_area(inter)
    return area / T_AREA


# -------------------------------------------------------------------- state


def to_state(phys):
    """Physics (pixels, y down) -> observed state (unit square, y up)."""
    ax, ay, vx, vy, bx, by, th = phys
    w = WORKSPACE
    return np.array(
        [ax / w, 1 - ay / w, vx / w, -vy / w, bx / w, 1 - by / w, np.cos(th), -np.sin(th)], dtype=np.float32
    )


def from_state(state):
    """Observed state -> physics: [agent x, y, vx, vy, block x, y, angle] in pixels, y down."""
    s = np.asarray(state, dtype=np.float64)
    w = WORKSPACE
    return np.array([s[0] * w, (1 - s[1]) * w, s[2] * w, -s[3] * w, s[4] * w, (1 - s[5]) * w, np.arctan2(-s[7], s[6])])


def physics_step(phys, action):
    """One 10 Hz control step: move the PD target by `action`, run ten 100 Hz substeps."""
    ax, ay, vx, vy, bx, by, th = phys
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    tx, ty = np.clip([ax + STEP * a[0], ay - STEP * a[1]], 0.0, WORKSPACE)  # the original's action range

    space = pymunk.Space()
    space.damping = 0.0  # no velocity memory: the block stops as soon as nothing pushes it
    space.add(*[pymunk.Segment(space.static_body, p, q, 2) for p, q in WALLS])

    agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    agent.position = (ax, ay)
    space.add(agent, pymunk.Circle(agent, AGENT_RADIUS))

    block = pymunk.Body(1.0, T_MOMENT)
    parts = [pymunk.Poly(block, part) for part in T_PARTS]
    block.center_of_gravity = T_COG
    block.angle = th          # the angle turns the block about its center of gravity,
    block.position = (bx, by)  # so set it before the position (as in the original)
    space.add(block, *parts)

    dt = 1.0 / SIM_HZ
    for _ in range(SIM_HZ // CONTROL_HZ):
        px, py = agent.position
        vx += (KP * (tx - px) - KV * vx) * dt
        vy += (KP * (ty - py) - KV * vy) * dt
        agent.velocity = (vx, vy)
        space.step(dt)

    (ax, ay), (bx, by) = agent.position, block.position
    return np.array([ax, ay, vx, vy, bx, by, block.angle])


# ---------------------------------------------------------------------- env


class PushTEnv:
    """
    Same interface as ToyPointEnv:
    - reset() -> (image, state, info)
    - step(action) -> (image, state, reward, done, info)
    """

    instruction = "Push the T-shaped block onto the T-shaped target."  # LeRobot's task string

    def __init__(self, seed=42, image_size=64):
        self.rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.state_dim = len(STATE_LABELS)
        self.action_dim = 2
        self.state_labels = STATE_LABELS
        self.expert = None

        self.reset()
        self.obs_shape = self._render().shape

    # world -----------------------------------------------------------------

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # the original's start distribution; skip starts where the agent
        # overlaps the block or the block already covers half the goal
        while True:
            agent = self.rng.uniform(50, 450, size=2)
            block = self.rng.uniform(100, 400, size=2)
            angle = self.rng.uniform(-np.pi, np.pi)
            pose = (*block, angle)
            clear = distance_to_t(agent, pose) > AGENT_RADIUS + 10
            if clear and coverage(pose) < 0.5:
                break

        self.state = to_state(np.array([*agent, 0.0, 0.0, *block, angle]))
        self.t = 0
        if self.expert is not None:
            self.expert.reset()
        return self._render(), self.state.copy(), self._info()

    def step(self, action):
        # the env keeps nothing but the observed state, so simulate() matches step() exactly
        self.state = to_state(physics_step(from_state(self.state), action))
        self.t += 1
        info = self._info()
        reward = min(info["coverage"] / SUCCESS_COVERAGE, 1.0)
        done = bool(info["success"])
        return self._render(), self.state.copy(), reward, done, info

    def close(self):
        pass

    def simulate(self, state, actions):
        """
        Ground-truth counterfactual: what happens under `actions` from `state`?

        state:   (8,) as returned by step()
        actions: (H, 2)
        returns: (H, 8) states
        """
        out, s = [], np.asarray(state, dtype=np.float32)
        for a in actions:
            s = to_state(physics_step(from_state(s), a))
            out.append(s)
        return np.asarray(out, dtype=np.float32)

    def block_pose(self, state=None):
        p = from_state(self.state if state is None else state)
        return p[4:7]

    def _info(self):
        cov = coverage(self.block_pose())
        return {"success": int(cov > SUCCESS_COVERAGE), "coverage": float(cov), "instruction": self.instruction}

    # expert ----------------------------------------------------------------

    def expert_action(self, noise=0.0):
        """Scripted pusher that plans with the true simulator (see envs/pusht_expert.py)."""
        if self.expert is None:
            from envs.pusht_expert import PushTExpert
            self.expert = PushTExpert()
        action = self.expert.act(from_state(self.state), self.t)
        action = action + noise * self.rng.standard_normal(2)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    # rendering -------------------------------------------------------------

    def _render(self, state=None):
        """Draw like the original: a 512 px canvas resized to 96 x 96, then to image_size."""
        import cv2

        n = int(WORKSPACE)
        img = np.full((n, n, 3), BACKGROUND, dtype=np.uint8)
        phys = from_state(self.state if state is None else state)

        for a, b in WALLS:
            cv2.line(img, a, b, WALL_COLOR, 4)
        for pose, color in ((GOAL_POSE, GOAL_COLOR), (phys[4:7], BLOCK_COLOR)):
            for part in T_PARTS:
                cv2.fillPoly(img, [np.round(to_world(part, pose)).astype(np.int32)], color)
        cv2.circle(img, tuple(int(round(c)) for c in phys[:2]), int(AGENT_RADIUS), AGENT_COLOR, -1)

        img = cv2.resize(img, (DATASET_SIZE, DATASET_SIZE))
        if self.image_size != DATASET_SIZE:
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        return img

    def render(self):
        return self._render()


def draw(state, size=192):
    """A sharp picture of a state, for videos: drawn straight at `size` pixels with smooth
    edges, instead of through the dataset's 96 px frames. The model never sees these."""
    import cv2

    k = size / WORKSPACE
    img = np.full((size, size, 3), BACKGROUND, dtype=np.uint8)
    phys = from_state(state)
    for a, b in WALLS:
        cv2.line(img, tuple(int(round(c * k)) for c in a), tuple(int(round(c * k)) for c in b), WALL_COLOR, 2, cv2.LINE_AA)
    for pose, color in ((GOAL_POSE, GOAL_COLOR), (phys[4:7], BLOCK_COLOR)):
        for part in T_PARTS:
            cv2.fillPoly(img, [np.round(to_world(part, pose) * k * 16).astype(np.int32)], color, cv2.LINE_AA, shift=4)
    center = tuple(int(round(c * k * 16)) for c in phys[:2])
    cv2.circle(img, center, int(round(AGENT_RADIUS * k * 16)), AGENT_COLOR, -1, cv2.LINE_AA, shift=4)
    return img


def distance_to_t(point, pose):
    """Distance from a world point to the T's outline (0 inside is not distinguished)."""
    outline = to_world(T_OUTLINE, pose)
    return min(point_segment_distance(point, a, b) for a, b in zip(outline, np.roll(outline, -1, axis=0)))


def point_segment_distance(p, a, b):
    ab, ap = b - a, np.asarray(p) - a
    t = np.clip(np.dot(ap, ab) / np.dot(ab, ab), 0.0, 1.0)
    return float(np.linalg.norm(ap - t * ab))
