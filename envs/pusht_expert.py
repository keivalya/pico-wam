"""A scripted Push-T expert that plans with the true simulator.

The original Push-T demonstrations are human teleoperation, so there is no
built-in expert. This one does what a careful human does, one push at a time:

    1. choose a push: try pushing every face of the T at many contact points,
       simulate each push with the real physics, and keep the one that brings
       the block closest to the goal (mean distance between corresponding
       corners of the block and the goal T)
    2. walk to the start of that push without touching the block (shortest
       path around the T's corners, kept one agent radius away)
    3. push for the planned number of steps, then choose the next push

Pushes get gentler as the block gets closer, so the last ones can place it
to within a pixel or two, and a random action in between (see
scripts/collect_data.py) makes it re-plan from wherever things ended up.
The expert reads the full state and simulates
about a hundred candidate pushes per decision: fine for collecting data,
far too slow and privileged to be a policy itself.
"""

import heapq

import numpy as np

from envs.pusht_env import (
    AGENT_RADIUS,
    GOAL_POSE,
    STEP,
    T_OUTLINE,
    TARGET_LIMITS,
    distance_to_t,
    physics_step,
    point_segment_distance,
    rotation,
    to_world,
)

GAP = 3.0                       # free space between the agent and the face before a push
CLEARANCE = AGENT_RADIUS + GAP  # the agent's center stays this far from the T while walking
SPACING = 12.0                  # pixels between candidate contact points
ANGLES = (-0.4, 0.0, 0.4)       # push directions relative to the face normal (radians)
MAX_PUSH = 20                   # steps per push
LEAD = 40.0                     # pixels the target leads the agent when walking or pushing hard
PATH_COST = 0.01                # pose error (pixels) worth one pixel of walking
CONVEX = (0, 1, 2, 4, 5, 7)     # T_OUTLINE corners a path can go around


def pose_error(pose, goal=GOAL_POSE):
    """Mean distance (pixels) between the block's corners and the goal's."""
    return float(np.linalg.norm(to_world(T_OUTLINE, pose) - to_world(T_OUTLINE, goal), axis=1).mean())


def contact_candidates():
    """Points along the T's outline with their outward normals, in block coordinates."""
    points, normals = [], []
    for a, b in zip(T_OUTLINE, np.roll(T_OUTLINE, -1, axis=0)):
        edge = b - a
        length = np.linalg.norm(edge)
        n = max(1, int(length // SPACING))
        for t in (np.arange(n) + 0.5) / n:
            points.append(a + t * edge)
            normals.append(np.array([edge[1], -edge[0]]) / length)
    return np.array(points), np.array(normals)


CONTACTS, NORMALS = contact_candidates()


def to_action(direction):
    """World displacement (pixels, y down) -> action (y up, in units of STEP)."""
    return np.array([direction[0], -direction[1]]) / STEP


class PushTExpert:
    def __init__(self):
        self.reset()

    def reset(self):
        self.plan = None
        self.last_step = None

    # ------------------------------------------------------------------ act

    def act(self, phys, step):
        """
        phys: [agent x, y, vx, vy, block x, y, angle] in pixels (y down)
        step: the env's step counter; a gap means someone else acted in between
        """
        agent, pose = phys[:2], phys[4:7]

        if self.last_step is not None and step != self.last_step + 1:
            self.plan = None  # random actions interrupted the plan: start over from here
        self.last_step = step

        if self.plan is not None and self.plan["pushing"] == 0:
            # the block was disturbed (e.g. by a random action) before the push began
            if pose_error(pose, self.plan["pose"]) > 3.0:
                self.plan = None
        if self.plan is None:
            self.plan = self.choose_push(phys)
            if self.plan is None:
                return np.zeros(2)  # nowhere useful to push from: wait

        plan = self.plan
        if plan["pushing"] == 0:
            start = self.push_start(plan, pose)
            if np.linalg.norm(start - agent) > 1.5 or np.linalg.norm(phys[2:4]) > 30:
                return self.walk_to(agent, start, pose)
            plan["direction"] = rotation(pose[2]) @ plan["local_direction"]

        plan["pushing"] += 1
        if plan["pushing"] >= plan["steps"]:
            self.plan = None
        return to_action(plan["lead"] * plan["direction"])

    # ----------------------------------------------------------- choose push

    @staticmethod
    def push_start(plan, pose):
        return to_world(plan["contact"] + plan["normal"] * CLEARANCE, pose)

    def choose_push(self, phys):
        agent, pose = phys[:2], phys[4:7]
        error = pose_error(pose)
        lead = LEAD * float(np.clip(error / 30.0, 0.1, 0.8))   # gentler pushes near the goal
        rot = rotation(pose[2])

        options = []
        for contact, normal in zip(CONTACTS, NORMALS):
            start = to_world(contact + normal * CLEARANCE, pose)
            inside = np.all((start >= TARGET_LIMITS[0]) & (start <= TARGET_LIMITS[1]))
            if not inside or distance_to_t(start, pose) < CLEARANCE - 0.5:
                continue  # off the table, or wedged into a corner of the T
            for angle in ANGLES:
                local = rotation(angle) @ -normal
                direction = rot @ local
                sim = np.array([*start, 0.0, 0.0, *pose])
                best_k, best_err = 0, error
                for k in range(1, MAX_PUSH + 1):
                    sim = physics_step(sim, to_action(lead * direction))
                    err = pose_error(sim[4:7])
                    if err < best_err:
                        best_k, best_err = k, err
                    elif k - best_k >= 3:
                        break  # getting worse: longer pushes won't help
                if best_k:
                    options.append((best_err, best_k, contact, normal, local))

        # walking matters too: add the path length for the most promising pushes
        best = None
        for err, k, contact, normal, local in sorted(options, key=lambda o: o[0])[:12]:
            path = self.path(agent, to_world(contact + normal * CLEARANCE, pose), pose)
            if path is None:
                continue
            score = err + PATH_COST * path_length(path)
            if best is None or score < best[0]:
                best = (score, k, contact, normal, local)
        if best is None:
            return None
        _, k, contact, normal, local = best
        return {"contact": contact, "normal": normal, "local_direction": local, "steps": k,
                "lead": lead, "pose": pose.copy(), "pushing": 0}

    # ------------------------------------------------------------------ walk

    def walk_to(self, agent, goal, pose):
        start = backed_off(agent, pose)
        if start is not agent:
            return to_action(start - agent)  # touching the block: back away from it first
        path = self.path(agent, goal, pose) or [agent, goal]
        # aim LEAD pixels ahead along the path, so the agent doesn't stop at every corner
        return to_action(point_along(path, LEAD) - agent)

    def path(self, start, goal, pose):
        """Shortest walk from start to goal that keeps CLEARANCE from the T, or None."""
        outline = to_world(T_OUTLINE, pose)
        backed = backed_off(start, pose)
        nodes = [np.asarray(backed, dtype=np.float64), np.asarray(goal, dtype=np.float64)]
        for i in CONVEX:
            prev, nxt = outline[i - 1], outline[(i + 1) % len(outline)]
            out = (outline[i] - prev) / np.linalg.norm(outline[i] - prev) + (outline[i] - nxt) / np.linalg.norm(outline[i] - nxt)
            corner = outline[i] + out / np.linalg.norm(out) * (CLEARANCE + 3) * np.sqrt(2)
            if np.all((corner >= TARGET_LIMITS[0]) & (corner <= TARGET_LIMITS[1])):
                nodes.append(corner)

        def clear(p, q):
            return all(segment_distance(p, q, a, b) >= CLEARANCE - 0.5
                       for a, b in zip(outline, np.roll(outline, -1, axis=0)))

        # Dijkstra from node 0 (start) to node 1 (goal)
        dist, prev, todo = {0: 0.0}, {}, [(0.0, 0)]
        while todo:
            d, i = heapq.heappop(todo)
            if i == 1:
                route = [1]
                while route[-1] != 0:
                    route.append(prev[route[-1]])
                route = [nodes[j] for j in reversed(route)]
                return route if backed is start else [np.asarray(start, dtype=np.float64)] + route
            if d > dist[i]:
                continue
            for j in range(1, len(nodes)):
                if j == i:
                    continue
                step = np.linalg.norm(nodes[j] - nodes[i])
                if d + step < dist.get(j, np.inf) and clear(nodes[i], nodes[j]):
                    dist[j], prev[j] = d + step, i
                    heapq.heappush(todo, (d + step, j))
        return None


def backed_off(agent, pose):
    """`agent` itself if it is clear of the T, otherwise the nearest point that is."""
    outline = to_world(T_OUTLINE, pose)
    edges = list(zip(outline, np.roll(outline, -1, axis=0)))
    p = agent
    for _ in range(10):  # a few rounds settle into an inside corner of the T
        nearest = min((closest_point(p, a, b) for a, b in edges), key=lambda q: np.linalg.norm(p - q))
        gap = np.linalg.norm(p - nearest)
        if gap >= CLEARANCE - 0.5:
            break
        away = (p - nearest) / max(gap, 1e-6)
        if inside_polygon(p, outline):
            away = -away
        p = nearest + away * (CLEARANCE + 1)
    return agent if p is agent or np.allclose(p, agent) else p


def inside_polygon(p, poly):
    """Even-odd rule."""
    inside = False
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            inside ^= p[0] < x
    return inside


def point_along(path, distance):
    """The point `distance` pixels along a polyline (its end if it is shorter)."""
    for a, b in zip(path, path[1:]):
        length = np.linalg.norm(b - a)
        if distance <= length:
            return a + (b - a) * distance / max(length, 1e-9)
        distance -= length
    return path[-1]


def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path, path[1:])))


def closest_point(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0)
    return a + t * ab


def segment_distance(p, q, a, b):
    """Minimum distance between segments pq and ab."""
    d1, d2 = q - p, b - a
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) > 1e-12:
        s = ((a - p)[0] * d2[1] - (a - p)[1] * d2[0]) / denom
        t = ((a - p)[0] * d1[1] - (a - p)[1] * d1[0]) / denom
        if 0 <= s <= 1 and 0 <= t <= 1:
            return 0.0  # they cross
    return min(
        point_segment_distance(p, a, b),
        point_segment_distance(q, a, b),
        point_segment_distance(a, p, q),
        point_segment_distance(b, p, q),
    )
