"""Measure where pico-WAM spends parameters and time.

Uses one trained baseline checkpoint pair (wam.pt, vla.pt and the run's
<env>.npz, from scripts/run.sh) and writes efficiency.json (by default to
docs/results/efficiency/):

    parameters   per module, WAM and VLA
    latency      ms per call (batch 1) for act(), one latent dynamics step,
                 the linear state readout, the image decoder, and imagine(k)
    train_step   ms per optimizer step (batch 64) for WAM (H=5), the same WAM
                 without its image decoder, and the VLA
    batching     ms to assemble one training batch with a per-sample
                 DataLoader-style loop vs. tensor indexing (the change made
                 in scripts/train.py)

Timings are medians over repeated calls on the machine that runs the script
(recorded in the output); they compare components against each other, not
hardware.

    python docs/figures/benchmark_efficiency.py --run-dir /path/to/baseline/seed0
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from models.wam_policy import PicoWAMPolicy  # noqa: E402
from models.losses import LossWeights, compute_wam_loss  # noqa: E402
from scripts.train import TrainingDataset  # noqa: E402

OUT = os.path.join(HERE, "..", "results", "efficiency")


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


def timeit(fn, repeats=200, warmup=20):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def per_sample_batch(ds, idx):
    """The old per-sample path: numpy -> torch per item, then stack (as a DataLoader collate does)."""
    items = []
    for i in idx:
        w = slice(i, i + ds.horizon)
        items.append({
            "image": torch.from_numpy(np.ascontiguousarray(ds.images[i])).permute(2, 0, 1).float() / 255.0,
            "state": torch.from_numpy(ds.states[i]).float(),
            "action": torch.from_numpy(ds.actions[i]).float(),
            "text_ids": torch.from_numpy(ds.text_ids[i]).long(),
            "actions": torch.from_numpy(ds.actions[w]).float(),
            "future_states": torch.from_numpy(ds.next_states[w]).float(),
            "future_images": torch.stack([torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0
                                          for im in ds.next_images[w]]),
        })
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Directory with wam.pt, vla.pt and the run's dataset (.npz)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    wam, _ = PicoWAMPolicy.from_checkpoint(os.path.join(args.run_dir, "wam.pt"))
    vla, _ = PicoWAMPolicy.from_checkpoint(os.path.join(args.run_dir, "vla.pt"))

    parts = ["img_encoder", "txt_encoder", "state_encoder", "fusion", "policy_head", "dynamics", "state_readout", "image_decoder"]
    params = {
        "wam": {p: n_params(getattr(wam, p)) for p in parts if hasattr(wam, p)},
        "vla": {p: n_params(getattr(vla, p)) for p in parts if hasattr(vla, p)},
    }
    params["wam_total"], params["vla_total"] = n_params(wam), n_params(vla)

    data_path = [os.path.join(args.run_dir, f) for f in sorted(os.listdir(args.run_dir)) if f.endswith(".npz")][0]
    ds = TrainingDataset(data_path, horizon=5, action_chunk=wam.config.get("action_chunk", 1))
    ds.to(torch.device("cpu"))
    i0 = int(ds.starts[0])
    img = ds.t_images[i0 : i0 + 1].float() / 255.0
    txt = ds.t_text_ids[i0 : i0 + 1]
    st = ds.t_states[i0 : i0 + 1]
    a = ds.t_actions[i0 : i0 + 1]

    with torch.no_grad():
        z = wam.encode_obs(img, txt, st)
        latency = {
            "act_wam": timeit(lambda: wam.act(img, txt, st)),
            "act_vla": timeit(lambda: vla.act(img, txt, st)),
            "encode": timeit(lambda: wam.encode_obs(img, txt, st)),
            "dynamics_step": timeit(lambda: wam.dynamics(z, a)),
            "state_readout": timeit(lambda: wam.state_readout(z)),
        }
        if wam.predict_image:
            latency["image_decoder"] = timeit(lambda: wam.image_decoder(z))
        for k in [1, 5, 10, 20]:
            acts = a[:, None].expand(1, k, -1).contiguous()
            latency[f"imagine_{k}_states"] = timeit(
                lambda acts=acts, k=k: [wam.state_readout(zz) for zz in wam.dynamics.rollout(z, acts).unbind(1)], repeats=100)
            if wam.predict_image:
                latency[f"imagine_{k}_states_and_images"] = timeit(
                    lambda acts=acts, k=k: wam.image_decoder(wam.dynamics.rollout(z, acts)), repeats=50)

    weights = LossWeights()
    starts = ds.starts[:64]
    batch = next(ds.batches(starts, 64, shuffle=False))
    # a WAM that predicts states but no pixels (same config, no image decoder;
    # the same as `wam` if it was trained without one)
    cfg = dict(wam.config, predict_image=False)
    wam_state_only = PicoWAMPolicy(**cfg)
    wam_state_only.load_state_dict({k: v for k, v in wam.state_dict().items() if not k.startswith("image_decoder")}, strict=False)
    params["wam_state_only_total"] = n_params(wam_state_only)

    train_step = {}
    for name, model in [("wam", wam), ("wam_state_only", wam_state_only), ("vla", vla)]:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def step(model=model, opt=opt):
            out = model(batch)
            loss, _ = compute_wam_loss(out, batch, model, weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_step[name] = timeit(step, repeats=30, warmup=5)

    batching = {
        "per_sample": timeit(lambda: per_sample_batch(ds, starts), repeats=50, warmup=5),
        "tensor_index": timeit(lambda: next(ds.batches(starts, 64, shuffle=False)), repeats=200),
    }

    result = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(), "torch": torch.__version__,
                    "threads": torch.get_num_threads(), "device": "cpu"},
        "parameters": params, "latency_ms": latency, "train_step_ms": train_step, "batch_assembly_ms": batching,
        "notes": "Median wall time per call; batch 1 for inference, batch 64 for training steps and batch assembly.",
    }
    with open(os.path.join(args.out, "efficiency.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
