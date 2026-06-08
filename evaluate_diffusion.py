"""Offline evaluation of trained Diffusion Policy on gesture mimic datasets.

Evaluates one or more checkpoints against a validation split,
reporting loss metrics, per-joint errors, and generating visualizations.

Unlike ACT evaluation (which reports L1 + KLD losses), Diffusion Policy
evaluation reports MSE noise-prediction loss and trajectory-level errors.

Usage:
    # Evaluate all checkpoints in a training run
    python evaluate_diffusion.py --training-dir outputs/train/diffusion_gesture

    # Evaluate a single checkpoint
    python evaluate_diffusion.py --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model

    # Use a specific dataset and validation episodes
    python evaluate_diffusion.py --training-dir outputs/train/diffusion_gesture \
        --dataset AmolSapale181284/multigesture-mimic \
        --val-episodes 45 46 47 48 49

    # Evaluate keypoint-mode diffusion policy
    python evaluate_diffusion.py --training-dir outputs/train/diffusion_gesture_kp \
        --dataset local/gesture_mimic_keypoints \
        --dataset-root data/local_gesture_mimic_keypoints
"""

import argparse
import os

os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ.setdefault("MIOPEN_FIND_ENFORCE", "5")

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_pre_post_processors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def load_dataset(repo_id, episodes, root=None, horizon=32, n_obs_steps=2):
    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    config = DiffusionConfig(horizon=horizon, n_obs_steps=n_obs_steps)
    delta_timestamps = resolve_delta_timestamps(config, ds_meta)
    kwargs = {"repo_id": repo_id, "episodes": episodes, "delta_timestamps": delta_timestamps, "tolerance_s": 0.0001}
    if root:
        kwargs["root"] = root
    return LeRobotDataset(**kwargs)


def load_policy(checkpoint_path):
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.to(DEVICE)
    policy.eval()
    return policy


def prepare_batch(batch, camera_keys, preprocessor):
    for cam_key in camera_keys:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0
    return preprocessor(batch)


def evaluate_loss(policy, dataloader, preprocessor, camera_keys):
    total_loss = 0.0
    n_batches = 0

    policy.train()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, camera_keys, preprocessor)
            loss, _ = policy.forward(batch)
            total_loss += loss.item()
            n_batches += 1
    policy.eval()

    return {
        "loss": total_loss / n_batches,
    }


def compute_per_joint_error(policy, dataloader, preprocessor, camera_keys):
    joint_errors = []

    policy.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = prepare_batch(batch, camera_keys, preprocessor)

            if policy.config.image_features:
                batch_for_select = dict(batch)
                batch_for_select["observation.images"] = torch.stack(
                    [batch_for_select[key] for key in policy.config.image_features], dim=-4
                )
            else:
                batch_for_select = batch

            obs_state = batch_for_select["observation.state"]
            batch_size = obs_state.shape[0]

            global_cond = policy.diffusion._prepare_global_conditioning(batch_for_select)
            actions_hat = policy.diffusion.conditional_sample(batch_size, global_cond=global_cond)

            gt_actions = batch["action"]
            n_obs = policy.config.n_obs_steps
            start = n_obs - 1
            end = start + policy.config.n_action_steps
            pred_slice = actions_hat[:, start:end]

            gt_len = gt_actions.shape[1]
            pred_len = pred_slice.shape[1]
            compare_len = min(gt_len, pred_len)

            abs_err = (pred_slice[:, :compare_len] - gt_actions[:, :compare_len]).abs()
            valid_mask = ~batch["action_is_pad"][:, :compare_len]
            for i in range(abs_err.shape[0]):
                valid = valid_mask[i]
                if valid.sum() > 0:
                    joint_errors.append(abs_err[i][valid].mean(dim=0).cpu().numpy())

    if not joint_errors:
        return np.zeros(len(JOINT_NAMES))
    joint_errors = np.stack(joint_errors)
    return joint_errors.mean(axis=0)


def predict_episode(policy, dataset, episode_idx, preprocessor, camera_keys, max_frames=200):
    """Predict actions for an episode using denoising inference."""
    predicted_actions = []
    gt_actions = []

    policy.eval()
    with torch.no_grad():
        for i in range(min(len(dataset), max_frames)):
            item = dataset[i]
            if item["episode_index"].item() != episode_idx:
                if len(gt_actions) > 0:
                    break
                continue

            batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else v for k, v in item.items()}
            batch = prepare_batch(batch, camera_keys, preprocessor)

            if policy.config.image_features:
                batch_input = dict(batch)
                for key in policy.config.image_features:
                    if key in batch_input and batch_input[key].ndim == 4:
                        batch_input[key] = batch_input[key].unsqueeze(1)
                batch_input["observation.images"] = torch.stack(
                    [batch_input[key] for key in policy.config.image_features], dim=-4
                )
            else:
                batch_input = batch

            obs_state = batch_input["observation.state"]
            if obs_state.ndim == 2:
                batch_input["observation.state"] = obs_state.unsqueeze(1)

            global_cond = policy.diffusion._prepare_global_conditioning(batch_input)
            actions_hat = policy.diffusion.conditional_sample(1, global_cond=global_cond)

            n_obs = policy.config.n_obs_steps
            start = n_obs - 1
            pred_action = actions_hat[0, start].cpu().numpy()
            predicted_actions.append(pred_action)

            gt_action = batch["action"]
            if gt_action.ndim == 3:
                gt_actions.append(gt_action[0, 0].cpu().numpy())
            else:
                gt_actions.append(gt_action[0].cpu().numpy())

    return np.array(predicted_actions), np.array(gt_actions)


def discover_checkpoints(training_dir):
    ckpt_base = Path(training_dir) / "checkpoints"
    if not ckpt_base.exists():
        return []
    checkpoints = []
    for d in sorted(ckpt_base.iterdir()):
        if d.is_dir() and (d / "pretrained_model").exists():
            checkpoints.append(str(d / "pretrained_model"))
    return checkpoints


def plot_results(results, joint_errors, pred_actions, gt_actions, eval_episode, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 16))

    if len(results) > 1:
        ax1 = fig.add_subplot(3, 2, 1)
        names = list(results.keys())
        losses = [r["loss"] for r in results.values()]
        try:
            x_vals = [int(Path(n).parent.name) for n in names]
        except ValueError:
            x_vals = list(range(len(names)))
        ax1.plot(x_vals, losses, "b-o", label="MSE Loss")
        ax1.set_xlabel("Training Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Validation Loss (Noise Prediction MSE)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if joint_errors is not None:
        ax2 = fig.add_subplot(3, 2, 2)
        colors = plt.cm.Set2(np.linspace(0, 1, len(JOINT_NAMES)))
        bars = ax2.bar(JOINT_NAMES, joint_errors, color=colors)
        ax2.set_ylabel("Mean Absolute Error")
        ax2.set_title("Per-Joint Trajectory Error (Validation Set)")
        ax2.tick_params(axis="x", rotation=30)
        for bar, err in zip(bars, joint_errors):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{err:.3f}",
                     ha="center", va="bottom", fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

    if pred_actions is not None and len(pred_actions) > 0:
        n_frames = min(len(pred_actions), len(gt_actions))
        for j in range(min(len(JOINT_NAMES), 4)):
            ax = fig.add_subplot(3, 2, j + 3)
            ax.plot(range(n_frames), gt_actions[:n_frames, j], "b-", alpha=0.7, label="Ground Truth")
            ax.plot(range(n_frames), pred_actions[:n_frames, j], "r--", alpha=0.7, label="Predicted")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Value")
            ax.set_title(f"{JOINT_NAMES[j]}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Diffusion Policy — Gesture Mimic Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    summary_path = output_dir / "eval_summary.png"
    plt.savefig(summary_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {summary_path}")

    if pred_actions is not None and len(pred_actions) > 0:
        fig2, axes2 = plt.subplots(3, 2, figsize=(14, 12))
        n_frames = min(len(pred_actions), len(gt_actions))
        for j in range(len(JOINT_NAMES)):
            ax = axes2[j // 2, j % 2]
            ax.plot(range(n_frames), gt_actions[:n_frames, j], "b-", linewidth=1.2, label="Ground Truth")
            ax.plot(range(n_frames), pred_actions[:n_frames, j], "r--", linewidth=1.2, label="Predicted")
            ax.set_xlabel("Frame")
            ax.set_ylabel("Value")
            ax.set_title(f"{JOINT_NAMES[j]}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig2.suptitle(f"Predicted vs Actual — Episode {eval_episode} (Diffusion Policy)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        actions_path = output_dir / "eval_actions.png"
        plt.savefig(actions_path, dpi=150)
        plt.close(fig2)
        print(f"Saved: {actions_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy for gesture mimic")
    parser.add_argument("--training-dir", type=str, help="Path to training output directory (evaluates all checkpoints)")
    parser.add_argument("--checkpoint", type=str, help="Path to a single checkpoint (or HuggingFace model ID)")
    parser.add_argument("--dataset", type=str, default="BlankHead/extended_gesture_mimic",
                        help="Dataset repo_id for evaluation")
    parser.add_argument("--dataset-root", type=str, default=None, help="Local dataset root path")
    parser.add_argument("--val-episodes", type=int, nargs="+", default=None,
                        help="Validation episode indices (default: last 10%% of episodes)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default=None, help="Where to save plots")
    _default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", type=str, default=_default_device)
    args = parser.parse_args()

    globals()["DEVICE"] = args.device

    if not args.training_dir and not args.checkpoint:
        parser.error("Provide either --training-dir or --checkpoint")

    ds_meta = LeRobotDatasetMetadata(args.dataset, root=args.dataset_root)
    total_episodes = ds_meta.total_episodes
    if args.val_episodes:
        val_episodes = args.val_episodes
    else:
        n_val = max(1, total_episodes // 10)
        val_episodes = list(range(total_episodes - n_val, total_episodes))

    print(f"Device: {DEVICE}")
    print(f"Dataset: {args.dataset} ({total_episodes} episodes)")
    print(f"Validation episodes: {val_episodes}")

    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    else:
        checkpoint_paths = discover_checkpoints(args.training_dir)
        if not checkpoint_paths:
            print(f"No checkpoints found in {args.training_dir}/checkpoints/")
            return

    # Load first checkpoint to read config (horizon, n_obs_steps)
    probe_policy = load_policy(checkpoint_paths[0])
    horizon = probe_policy.config.horizon
    n_obs_steps = probe_policy.config.n_obs_steps
    del probe_policy
    torch.cuda.empty_cache()

    print(f"Policy config: horizon={horizon}, n_obs_steps={n_obs_steps}")

    val_dataset = load_dataset(args.dataset, val_episodes, root=args.dataset_root,
                               horizon=horizon, n_obs_steps=n_obs_steps)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    camera_keys = list(val_dataset.meta.camera_keys)
    print(f"Validation samples: {len(val_dataset)}, Camera keys: {camera_keys}")

    results = {}
    print(f"\n{'Checkpoint':<60} {'MSE Loss':>10}")
    print("-" * 75)

    for ckpt_path in checkpoint_paths:
        policy = load_policy(ckpt_path)
        preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(ckpt_path))
        metrics = evaluate_loss(policy, val_loader, preprocessor, camera_keys)
        results[ckpt_path] = metrics
        label = ckpt_path if len(ckpt_path) < 58 else f"...{ckpt_path[-55:]}"
        print(f"  {label:<58} {metrics['loss']:>10.6f}")
        del policy, preprocessor
        torch.cuda.empty_cache()

    last_ckpt = checkpoint_paths[-1]
    print(f"\n=== Per-Joint Trajectory Error ({Path(last_ckpt).parent.name}) ===")
    policy = load_policy(last_ckpt)
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, pretrained_path=str(last_ckpt))
    joint_errors = compute_per_joint_error(policy, val_loader, preprocessor, camera_keys)
    for name, err in zip(JOINT_NAMES, joint_errors):
        print(f"  {name:<20} {err:.4f}")
    print(f"  {'MEAN':<20} {joint_errors.mean():.4f}")

    eval_episode = val_episodes[0]
    ep_dataset = load_dataset(args.dataset, [eval_episode], root=args.dataset_root,
                              horizon=horizon, n_obs_steps=n_obs_steps)
    pred_actions, gt_actions = predict_episode(policy, ep_dataset, eval_episode, preprocessor, camera_keys)
    print(f"\nPredicted vs actual: episode {eval_episode}, {len(pred_actions)} frames")

    output_dir = args.output_dir
    if not output_dir:
        if args.training_dir:
            output_dir = str(Path(args.training_dir) / "eval")
        else:
            output_dir = str(Path(last_ckpt).parent.parent.parent / "eval")
    plot_results(results, joint_errors, pred_actions, gt_actions, eval_episode, output_dir)

    metrics_path = Path(output_dir) / "eval_metrics.json"
    serializable = {}
    for k, v in results.items():
        serializable[str(k)] = v
    serializable["per_joint_error"] = {name: float(err) for name, err in zip(JOINT_NAMES, joint_errors)}
    serializable["mean_joint_error"] = float(joint_errors.mean())
    serializable["policy_type"] = "diffusion"
    with open(metrics_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Saved: {metrics_path}")

    del policy
    torch.cuda.empty_cache()
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
