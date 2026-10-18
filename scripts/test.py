"""Test a pico-WAM (or mini-VLA baseline) policy in closed loop (Push-T, toy env or Meta-World MT1).

With --save-video, every frame of a WAM rollout shows what the model sees
and what it is thinking:

    | real observation | decode(z_t): what it believes now | decode(z_t+k): what it expects k steps ahead |

so you can watch the world model's predictions succeed (or fail) live.

Usage:
    python -m scripts.test --checkpoint checkpoints/wam.pt --episodes 20 --save-video
"""

import os
import json
import argparse

import numpy as np
import torch

from models.wam_policy import PicoWAMPolicy
from utils.tokenizer import SimpleTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Test a pico-WAM policy in closed loop")

    parser.add_argument("--checkpoint", type=str, default="checkpoints/model.pt")
    parser.add_argument("--env", type=str, default=None, choices=["pusht", "toy", "metaworld"],
                        help="Defaults to the env the checkpoint was trained on.")
    parser.add_argument("--env-name", type=str, default="push-v3", help="Meta-World task")
    parser.add_argument("--camera-name", type=str, default="topview")
    parser.add_argument("--seed", type=int, default=10_000, help="Use seeds not seen in training.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=None, help="Default: 300 for pusht, 80 otherwise.")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Meta-World: language instruction (pusht and toy have their own).",
    )
    parser.add_argument("--lookahead", type=int, default=5, help="Imagination steps shown in the video.")
    parser.add_argument("--execute-steps", type=int, default=None,
                        help="With action chunks of K actions: how many to execute before predicting again (default K // 2).")
    parser.add_argument("--planner", type=str, default="policy", choices=["policy", "mpc"],
                        help="policy: act with the action head; mpc: plan with the world model (scripts/plan.py, Push-T).")
    parser.add_argument("--planner-config", type=str, default="{}",
                        help='JSON overrides for the planner, e.g. \'{"warm_start": true}\' or \'{"iterations": 3, "noise": 0.3}\'.')
    parser.add_argument("--check-plans-every", type=int, default=10,
                        help="mpc on Push-T: every N steps, compare the chosen plan's imagined outcome with the simulator's (0 = never).")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="videos")
    parser.add_argument("--json-out", type=str, default=None, help="Write success rate / steps here.")

    return parser.parse_args()


def make_env(args, env_name, image_size):
    if env_name == "pusht":
        from envs.pusht_env import PushTEnv
        return PushTEnv(seed=args.seed, image_size=image_size)

    if env_name == "toy":
        from envs.toy_point_env import ToyPointEnv
        return ToyPointEnv(seed=args.seed, image_size=image_size)

    from envs.metaworld_env import MetaWorldMT1Wrapper
    return MetaWorldMT1Wrapper(
        env_name=args.env_name,
        seed=args.seed,
        render_mode="rgb_array",
        camera_name=args.camera_name,
    )


def to_tensor(img, size, device):
    if img.shape[0] != size:
        import cv2
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0).div(255.0).to(device)


def to_np_img(x, size=192):
    import cv2
    img = (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)


def caption(img, text):
    import cv2
    return cv2.putText(img.copy(), text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


@torch.no_grad()
def mind_frame(model, image_t, text_ids, state_t, lookahead):
    """Real frame | what the WAM believes now | what it expects `lookahead` steps ahead."""
    panels = [caption(to_np_img(image_t[0]), "real")]
    if model.predict_image:
        dream = model.imagine(image_t, text_ids, state_t, horizon=lookahead)
        panels.append(caption(to_np_img(dream["image"][0, 0]), "believes now"))
        panels.append(caption(to_np_img(dream["image"][0, -1]), f"expects t+{lookahead}"))
    return np.concatenate(panels, axis=1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[test] Loading checkpoint from {args.checkpoint}")
    model, ckpt = PicoWAMPolicy.from_checkpoint(args.checkpoint, device=device)
    tokenizer = SimpleTokenizer(vocab=ckpt["vocab"])
    size = model.config["image_size"]

    env_name = args.env or ckpt.get("env", "toy")
    env = make_env(args, env_name, size)
    max_steps = args.max_steps or (300 if env_name == "pusht" else 80)
    print(f"[test] env={env_name} state_dim={env.state_dim} action_dim={env.action_dim} "
          f"model_type={model.config['model_type']}")

    if args.save_video:
        import imageio.v2 as imageio
        os.makedirs(args.video_dir, exist_ok=True)

    planner, plan_checks = None, []
    if args.planner == "mpc":
        from scripts.plan import LatentPlanner
        planner = LatentPlanner(model, **json.loads(args.planner_config))
        sim = make_env(args, env_name, size)

    successes, lengths, coverages = 0, [], []

    for ep in range(args.episodes):
        img, state, info = env.reset(seed=args.seed + ep)
        instruction = info.get("instruction") or args.instruction or "push the object to the goal"
        text_ids = torch.tensor([tokenizer.encode(instruction)], dtype=torch.long, device=device)

        frames, success, step, best_coverage = [], 0, 0, 0.0
        done = False
        if planner is not None:
            planner.reset()
        queue = []   # actions of the current chunk still to execute
        while not done and step < max_steps:
            img_t = to_tensor(img, size, device)
            state_t = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)

            if args.save_video:
                frames.append(mind_frame(model, img_t, text_ids, state_t, args.lookahead)
                              if model.is_wam else to_np_img(img_t[0]))

            if planner is None and model.config.get("action_chunk", 1) > 1:
                if not queue:
                    k = args.execute_steps or max(1, model.config["action_chunk"] // 2)
                    queue = list(model.act_chunk(img_t, text_ids, state_t)[0, :k].cpu().numpy())
                action = queue.pop(0)
            else:
                act = model.act if planner is None else planner.act
                action = act(img_t, text_ids, state_t).squeeze(0).cpu().numpy()
                if planner is not None and args.check_plans_every and env_name == "pusht" and step % args.check_plans_every == 0:
                    plan_checks.append(planner.check_last_plan(sim, np.asarray(state)))
            img, state, reward, done, info = env.step(action)
            success = max(success, int(info.get("success", 0)))
            best_coverage = max(best_coverage, info.get("coverage", 0.0))
            step += 1

            # Meta-World doesn't stop on success by itself
            done = done or bool(success)

        successes += success
        lengths.append(step)
        coverages.append(best_coverage)
        extra = f" max_coverage={best_coverage:.3f}" if "coverage" in info else ""
        print(f"[test] Episode {ep + 1}/{args.episodes}: \"{instruction}\" success={success} steps={step}{extra}")

        if args.save_video:
            path = os.path.join(args.video_dir, f"{env_name}_ep{ep + 1:03d}.mp4")
            with imageio.get_writer(path, fps=10) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"[test] Saved video to {path}")

    env.close()
    has_coverage = "coverage" in info
    print(f"[test] Success rate: {successes}/{args.episodes} = {successes / args.episodes:.0%} "
          f"| mean steps: {np.mean(lengths):.1f}"
          + (f" | mean max coverage: {np.mean(coverages):.3f}" if has_coverage else "")
          + (f" | chosen plans, block px: expected gain {np.mean([c['expected_gain_px'] for c in plan_checks]):+.1f}, "
             f"real gain {np.mean([c['real_gain_px'] for c in plan_checks]):+.1f}, "
             f"error {np.mean([c['plan_error_px'] for c in plan_checks]):.1f}" if plan_checks else ""))

    if args.json_out:
        out = {"success_rate": successes / args.episodes, "episodes": args.episodes,
               "mean_steps": float(np.mean(lengths)), "seed": args.seed, "planner": args.planner}
        if has_coverage:
            # Push-T's usual score: the best coverage reached in each episode, averaged
            out["mean_max_coverage"] = float(np.mean(coverages))
            out["max_coverage"] = [float(c) for c in coverages]
        if planner is not None:
            out["planner_config"] = {k: getattr(planner, k) for k in
                                     ["horizon", "samples", "elites", "iterations", "noise", "approach_weight", "warm_start", "prior"]}
            if plan_checks:
                out["plan_checks"] = {k: float(np.mean([c[k] for c in plan_checks])) for k in plan_checks[0]}
                out["plan_checks"]["n"] = len(plan_checks)
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
