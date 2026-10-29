"""Collect WAM-compatible demonstration data from an expert policy.

    --env pusht      Push-T with a scripted expert (pymunk, minutes on a laptop); for
                     LeRobot's human demonstrations use scripts/download_pusht.py
    --env toy        2D point-mass world from the paper (numpy only, seconds on a laptop)
    --env metaworld  Meta-World MT1 task with its scripted expert

VLA data:
    observation_t -> action_t

pico-WAM data:
    observation_t -> action_t + observation_t+1

Saved .npz keys:
    images       : (N, H, W, 3)
    states       : (N, state_dim)
    actions      : (N, action_dim)
    next_images  : (N, H, W, 3)
    next_states  : (N, state_dim)
    is_expert    : (N,)          1 if a_t came from the expert, 0 if it was a random exploration action
    episode_ids  : (N,)          which episode each transition belongs to
    timesteps    : (N,)          step index inside the episode
    text_ids     : (N, T_text)
    texts        : (N,)
    vocab        : dict
    state_labels : (state_dim,)  human-readable name of every state dimension

Transitions are stored in episode order, so the WAM can train on
multi-step windows (o_t, a_t, ..., a_t+H-1, o_t+H) inside one episode.

Why random actions (--random-action-prob)?
    In pure expert data the action is a function of the state, so a world
    model can predict o_t+1 from o_t alone and simply ignore a_t. It then
    "imagines" the expert's future no matter what action you ask about.
    Mixing in random actions breaks that shortcut: the world model trains on
    every transition, the policy imitates only the expert ones (is_expert).
"""

import os
import argparse
import time

import numpy as np

from utils.tokenizer import SimpleTokenizer


# per-env defaults for flags left unset. Push-T needs precise final pushes, so
# it gets fewer random bursts: at 0.3 they undo the expert's placement and
# most episodes never finish.
ENV_DEFAULTS = {
    "pusht": {"max_steps": 300, "random_action_prob": 0.1, "expert_noise": 0.05},
    "toy": {"max_steps": 150, "random_action_prob": 0.3, "expert_noise": 0.1},
    "metaworld": {"max_steps": 150, "random_action_prob": 0.3, "expert_noise": 0.0},
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="pusht", choices=["pusht", "toy", "metaworld"])
    parser.add_argument("--env-name", type=str, default="push-v3", help="Meta-World task")

    parser.add_argument(
        "--camera-name",
        type=str,
        default="topview",
        help=(
            "Meta-World camera: corner, corner2, corner3, corner4, "
            "topview, behindGripper, gripperPOV"
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Default: 300 for pusht, 150 otherwise.")

    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Images are resized to (image-size, image-size) before saving.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="data/dataset.npz",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional sleep between steps for visualization.",
    )

    parser.add_argument(
        "--instruction",
        type=str,
        default="push the object to the goal",
        help="Fixed instruction for all Meta-World episodes (pusht and toy have their own).",
    )

    parser.add_argument(
        "--stop-on-success",
        action="store_true",
        help="End each episode early once the env reports success.",
    )

    parser.add_argument(
        "--random-action-prob",
        type=float,
        default=None,
        help="Probability of starting a burst of random (non-expert) actions at each step. "
        "Default: 0.1 for pusht, 0.3 otherwise.",
    )

    parser.add_argument(
        "--random-burst",
        type=int,
        default=3,
        help="Max length of a random-action burst (actual length ~ U{1..random-burst}).",
    )

    parser.add_argument(
        "--expert-noise",
        type=float,
        default=None,
        help="pusht and toy: std of Gaussian noise added to expert actions. Default: 0.05 for pusht, 0.1 for toy.",
    )

    args = parser.parse_args()
    for name, value in ENV_DEFAULTS[args.env].items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    return args


def resize(img, size):
    if img.shape[0] == size and img.shape[1] == size:
        return img
    import cv2
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def tokenize_texts(texts):
    tokenizer = SimpleTokenizer(vocab=None)
    tokenizer.build_from_texts(texts)

    text_ids_list = [tokenizer.encode(t) for t in texts]
    max_len = max(len(seq) for seq in text_ids_list)

    text_ids = np.zeros((len(texts), max_len), dtype=np.int64)

    for i, seq in enumerate(text_ids_list):
        text_ids[i, : len(seq)] = np.asarray(seq, dtype=np.int64)

    return text_ids, tokenizer.vocab


class ExpertEnv:
    """Common interface: reset(ep) -> (img, state, text), expert(), step(a) -> (img, state, done, success)."""

    def __init__(self, args):
        self.args = args

        if args.env == "pusht":
            from envs.pusht_env import PushTEnv, STATE_LABELS

            self.env = PushTEnv(seed=args.seed, image_size=args.image_size)
            self.state_labels = STATE_LABELS

        elif args.env == "toy":
            from envs.toy_point_env import ToyPointEnv, STATE_LABELS

            self.env = ToyPointEnv(seed=args.seed, image_size=args.image_size)
            self.state_labels = STATE_LABELS

        else:
            from envs.metaworld_env import MetaWorldMT1Wrapper, METAWORLD_STATE_LABELS
            from metaworld.policies import ENV_POLICY_MAP

            self.env = MetaWorldMT1Wrapper(
                env_name=args.env_name,
                seed=args.seed,
                render_mode="rgb_array",
                camera_name=args.camera_name,
            )
            self.policy = ENV_POLICY_MAP[args.env_name]()
            self.state_labels = METAWORLD_STATE_LABELS[: self.env.state_dim]

    def reset(self, ep):
        img, state, info = self.env.reset(seed=self.args.seed + ep)
        self.state = state
        if self.args.env in ("pusht", "toy"):
            return img, state, info["instruction"]
        return img, state, self.args.instruction

    def random_action(self, rng):
        if self.args.env in ("pusht", "toy"):
            return rng.uniform(-1.0, 1.0, size=self.env.action_dim).astype(np.float32)
        return self.env.env.action_space.sample().astype(np.float32)

    def expert(self):
        if self.args.env in ("pusht", "toy"):
            return self.env.expert_action(noise=self.args.expert_noise)
        return np.asarray(self.policy.get_action(self.state), dtype=np.float32)

    def step(self, action):
        img, state, reward, done, info = self.env.step(action)
        self.state = state
        return img, state, done, int(info.get("success", 0))

    def close(self):
        self.env.close()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    env = ExpertEnv(args)

    images = []
    states = []
    actions = []
    next_images = []
    next_states = []
    is_expert = []
    episode_ids = []
    timesteps = []
    texts = []

    rng = np.random.default_rng(args.seed)

    total_successes = 0

    for ep in range(args.episodes):
        image_t, state_t, instruction = env.reset(ep)
        image_t = resize(image_t, args.image_size)

        done = False
        steps = 0
        episode_success = 0
        burst = 0

        while not done and steps < args.max_steps:
            # Expert action a_t, or (sometimes) a random exploration action
            if burst == 0 and rng.random() < args.random_action_prob:
                burst = rng.integers(1, args.random_burst + 1)

            expert_step = burst == 0
            if expert_step:
                action_t = env.expert()
            else:
                action_t = env.random_action(rng)
                burst -= 1

            # Step to next observation o_{t+1}
            image_tp1, state_tp1, done, success = env.step(action_t)
            image_tp1 = resize(image_tp1, args.image_size)

            episode_success = max(episode_success, success)

            # Store WAM transition:
            # observation_t -> action_t + observation_t+1
            images.append(image_t.copy())
            states.append(np.asarray(state_t, dtype=np.float32).copy())
            actions.append(action_t.copy())
            next_images.append(image_tp1.copy())
            next_states.append(np.asarray(state_tp1, dtype=np.float32).copy())
            is_expert.append(float(expert_step))
            episode_ids.append(ep)
            timesteps.append(steps)
            texts.append(instruction)

            image_t, state_t = image_tp1, state_tp1
            steps += 1

            # (pusht and toy already report done=True on success)
            if args.stop_on_success and success == 1:
                done = True

            if args.sleep > 0:
                time.sleep(args.sleep)

        total_successes += episode_success

        print(
            f"Episode {ep + 1}/{args.episodes} "
            f"finished after {steps} steps, "
            f"success={episode_success}"
        )

    env.close()

    images = np.stack(images, axis=0)
    states = np.stack(states, axis=0)
    actions = np.stack(actions, axis=0)
    next_images = np.stack(next_images, axis=0)
    next_states = np.stack(next_states, axis=0)
    is_expert = np.asarray(is_expert, dtype=np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    timesteps = np.asarray(timesteps, dtype=np.int64)
    texts = np.asarray(texts)

    text_ids, vocab = tokenize_texts(texts)

    np.savez_compressed(
        args.output_path,
        images=images,
        states=states,
        actions=actions,
        next_images=next_images,
        next_states=next_states,
        is_expert=is_expert,
        episode_ids=episode_ids,
        timesteps=timesteps,
        text_ids=text_ids,
        texts=texts,
        vocab=vocab,
        state_labels=np.asarray(env.state_labels),
        env=np.asarray(args.env),
    )

    print()
    print("Saved WAM-compatible dataset to:", args.output_path)
    print("  images:      ", images.shape)
    print("  states:      ", states.shape)
    print("  actions:     ", actions.shape)
    print("  next_images: ", next_images.shape)
    print("  next_states: ", next_states.shape)
    print("  text_ids:    ", text_ids.shape)
    print("  vocab size:  ", len(vocab))
    print("  expert frac: ", f"{is_expert.mean():.2f}")
    print("  successes:   ", f"{total_successes}/{args.episodes}")


if __name__ == "__main__":
    main()
