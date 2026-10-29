"""Download LeRobot's Push-T demonstrations and convert them for pico-WAM.

    python -m scripts.download_pusht --output-path data/pusht.npz

lerobot/pusht holds the 206 human demonstrations from Diffusion Policy
(Chi et al., 2023): 25,650 frames at 10 Hz, each with a 96 x 96 image (all
stored in one AV1 video), the agent's position, and the action, an absolute
target position for the agent. It does not record where the T is, but
lerobot/pusht_keypoints holds the same frames with the T's eight corners,
from which the block's pose follows exactly.

Each frame becomes the env's 8-dim state (envs/pusht_env.py):

    agent position   recorded
    agent velocity   not recorded: recovered by replaying the PD controller
                     from rest, which reproduces every recorded position to
                     within a ten-thousandth of a pixel
    block pose       the bar's and the stem's centers, from the corners

and each absolute target becomes the env's relative action,
(target - position) / STEP, clipped to [-1, 1].

Transitions (o_t, a_t, o_t+1) stop one frame before the end of each
episode, since the last action has no recorded outcome. Every action is a
human one (is_expert = 1): unlike scripts/collect_data.py, this data has no
random-action bursts.

The saved .npz has the same keys as scripts/collect_data.py writes.
"""

import argparse
import json
import os

import numpy as np

from envs.pusht_env import KP, KV, SIM_HZ, CONTROL_HZ, STATE_LABELS, STEP, PushTEnv, to_state
from scripts.collect_data import tokenize_texts


# pinned revisions, so every run converts exactly the same data
DATASET = ("lerobot/pusht", "7628202a2180972f291ba1bc6723834921e72c19")
KEYPOINTS = ("lerobot/pusht_keypoints", "ace8c161a68bc025c21a5f29f85b86a9a2c5e64b")
IMAGE_KEY = "observation.image"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, default="data/pusht.npz")
    parser.add_argument("--image-size", type=int, default=64, help="Frames are resized from 96 x 96 to this.")
    parser.add_argument("--episodes", type=int, default=None, help="Keep only the first N episodes (quick tests).")
    return parser.parse_args()


# ---------------------------------------------------------------- download


def fetch(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo[0], filename, repo_type="dataset", revision=repo[1])


def read_parquet(paths, columns=None):
    """Concatenate parquet files into a dict of numpy arrays (list columns are stacked)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.concat_tables([pq.read_table(p, columns=columns) for p in paths])
    out = {}
    for name in table.column_names:
        col = table.column(name).to_pylist()
        numeric_list = isinstance(col[0], list) and isinstance(col[0][0], (int, float))
        out[name] = np.asarray(col, dtype=np.float64 if numeric_list else None)
    return out


def load_lerobot(repo):
    """Every frame of a LeRobot v3 dataset, its episode table and its info.json."""
    with open(fetch(repo, "meta/info.json")) as f:
        info = json.load(f)
    episodes = read_parquet([fetch(repo, "meta/episodes/chunk-000/file-000.parquet")])  # one file for 206 episodes
    files = sorted({(int(c), int(i)) for c, i in zip(episodes["data/chunk_index"], episodes["data/file_index"])})
    frames = read_parquet([fetch(repo, info["data_path"].format(chunk_index=c, file_index=i)) for c, i in files])
    order = np.argsort(frames["index"])
    return {k: v[order] for k, v in frames.items()}, episodes, info


def decode_frames(info, episodes, size):
    """All video frames in dataset order, resized to (size, size) as the env does."""
    import av
    import cv2

    fps = info["fps"]
    out = []
    keys = (f"videos/{IMAGE_KEY}/chunk_index", f"videos/{IMAGE_KEY}/file_index")
    for chunk, file in sorted({(int(c), int(i)) for c, i in zip(episodes[keys[0]], episodes[keys[1]])}):
        path = fetch(DATASET, info["video_path"].format(video_key=IMAGE_KEY, chunk_index=chunk, file_index=file))
        video = []
        with av.open(path) as container:
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                if img.shape[0] != size:
                    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
                video.append(img)
        mine = (episodes[keys[0]] == chunk) & (episodes[keys[1]] == file)
        for start, length in zip(episodes[f"videos/{IMAGE_KEY}/from_timestamp"][mine], episodes["length"][mine]):
            first = int(round(start * fps))
            out.extend(video[first : first + int(length)])
    return np.stack(out)


# ----------------------------------------------------------------- convert


def block_poses(corners):
    """(N, 16) corners, bar first then stem -> (N, 3) poses (x, y, angle) in pixels, y down."""
    corners = corners.reshape(-1, 8, 2)
    bar, stem = corners[:, :4].mean(axis=1), corners[:, 4:].mean(axis=1)
    d = stem - bar                                   # = R(angle) @ (0, 60) in block coordinates
    angle = np.arctan2(-d[:, 0], d[:, 1])
    origin = bar - np.stack([-np.sin(angle), np.cos(angle)], axis=1) * 15.0  # the bar's center is (0, 15)
    return np.concatenate([origin, angle[:, None]], axis=1)


def agent_velocities(positions, targets, episode_ids):
    """Replay the PD controller from rest; returns velocities and the worst position mismatch."""
    dt, n_sub = 1.0 / SIM_HZ, SIM_HZ // CONTROL_HZ
    vel, worst = np.zeros_like(positions), 0.0
    for e in np.unique(episode_ids):
        idx = np.nonzero(episode_ids == e)[0]
        v = np.zeros(2)
        for i, nxt in zip(idx, list(idx[1:]) + [None]):
            vel[i] = v
            p = positions[i].copy()
            for _ in range(n_sub):
                v = v + (KP * (targets[i] - p) - KV * v) * dt
                p = p + v * dt
            if nxt is not None:
                worst = max(worst, float(np.abs(p - positions[nxt]).max()))
    return vel, worst


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    frames, episodes, info = load_lerobot(DATASET)
    corners, _, _ = load_lerobot(KEYPOINTS)
    assert np.array_equal(frames["index"], corners["index"]), "the two datasets list different frames"
    images = decode_frames(info, episodes, args.image_size)
    assert len(images) == len(frames["index"]), "video and data disagree on the number of frames"

    episode_ids = frames["episode_index"].astype(np.int64)
    agent, target = frames["observation.state"], frames["action"]
    vel, replay_error = agent_velocities(agent, target, episode_ids)
    pose = block_poses(corners["observation.environment_state"])
    states = np.stack([to_state(np.concatenate([a, v, p])) for a, v, p in zip(agent, vel, pose)])

    offset = (target - agent) / STEP
    actions = np.clip(np.stack([offset[:, 0], -offset[:, 1]], axis=1), -1.0, 1.0).astype(np.float32)  # y up
    clipped = int((np.abs(offset) > 1.0).any(axis=1).sum())

    # transitions (o_t, a_t, o_t+1) inside one episode
    keep = np.nonzero(episode_ids[:-1] == episode_ids[1:])[0]
    if args.episodes is not None:
        keep = keep[episode_ids[keep] < np.unique(episode_ids)[args.episodes - 1] + 1]

    tasks = read_parquet([fetch(DATASET, "meta/tasks.parquet")])
    task = str(tasks["__index_level_0__"][0]) if "__index_level_0__" in tasks else PushTEnv.instruction
    if task != PushTEnv.instruction:
        print(f"[warning] dataset task {task!r} differs from the env's instruction {PushTEnv.instruction!r}")
    texts = np.asarray([PushTEnv.instruction] * len(keep))
    text_ids, vocab = tokenize_texts(texts)

    np.savez_compressed(
        args.output_path,
        images=images[keep],
        states=states[keep],
        actions=actions[keep],
        next_images=images[keep + 1],
        next_states=states[keep + 1],
        is_expert=np.ones(len(keep), dtype=np.float32),
        episode_ids=episode_ids[keep],
        timesteps=frames["frame_index"][keep].astype(np.int64),
        text_ids=text_ids,
        texts=texts,
        vocab=vocab,
        state_labels=np.asarray(STATE_LABELS),
        env=np.asarray("pusht"),
        source=np.asarray(f"{DATASET[0]}@{DATASET[1]} + {KEYPOINTS[0]}@{KEYPOINTS[1]}"),
    )

    lengths = np.bincount(episode_ids[keep])
    lengths = lengths[lengths > 0]
    print("Saved WAM-compatible dataset to:", args.output_path)
    print("  source:      ", f"{DATASET[0]}@{DATASET[1][:7]}, {KEYPOINTS[0]}@{KEYPOINTS[1][:7]}")
    print("  episodes:    ", len(lengths), f"(transitions per episode: mean {lengths.mean():.0f}, "
          f"min {lengths.min()}, max {lengths.max()})")
    print("  images:      ", images[keep].shape)
    print("  states:      ", states[keep].shape)
    print("  actions:     ", actions[keep].shape, f"({clipped} of {len(actions)} targets clipped to STEP)")
    print("  PD replay:   ", f"worst agent position mismatch {replay_error:.1e} px")
    print("  vocab size:  ", len(vocab))


if __name__ == "__main__":
    main()
