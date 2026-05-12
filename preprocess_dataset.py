"""Preprocess a gesture mimic dataset: extract pose keypoints from video frames.

Takes an existing LeRobot dataset with RGB video (observation.images.human)
and creates a new dataset where keypoints replace the image features.

The new dataset's observation.state is extended from (6,) robot joints to
(6 + K,) where K is the keypoint feature dimension (positions + velocities
+ accelerations of normalized upper-body landmarks).

The resulting dataset can be trained with ACT without the vision backbone,
making training much faster and more data-efficient for small datasets.

Usage:
    # Basic: preprocess the extended gesture mimic dataset
    python preprocess_dataset.py \
        --source BlankHead/extended_gesture_mimic \
        --target local/extended_gesture_mimic_keypoints

    # With custom landmark set (right arm only)
    python preprocess_dataset.py \
        --source AmolSapale181284/multigesture-mimic \
        --target local/multigesture_mimic_keypoints \
        --landmarks right_arm

    # Keep images alongside keypoints (hybrid mode)
    python preprocess_dataset.py \
        --source BlankHead/extended_gesture_mimic \
        --target local/extended_gesture_mimic_hybrid \
        --keep-images

    # Push to HuggingFace Hub
    python preprocess_dataset.py \
        --source BlankHead/extended_gesture_mimic \
        --target myuser/extended_gesture_mimic_keypoints \
        --push-to-hub
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from pose_estimator import PoseEstimator, LANDMARK_PRESETS


def image_tensor_to_rgb(tensor):
    """Convert a LeRobot image tensor to uint8 RGB numpy array."""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        arr = tensor.permute(1, 2, 0).numpy()
    else:
        arr = tensor.numpy()
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def preprocess_dataset(
    source_repo_id,
    target_repo_id,
    target_root=None,
    source_root=None,
    landmarks="upper_body",
    include_velocity=True,
    include_acceleration=True,
    smoothing_alpha=0.3,
    use_3d=True,
    keep_images=False,
    push_to_hub=False,
    batch_size_display=100,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    # --- Load source dataset metadata ---
    print(f"Loading source dataset: {source_repo_id}")
    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    print(f"  Episodes: {source_meta.total_episodes}")
    print(f"  Frames: {source_meta.total_frames}")
    print(f"  FPS: {source_meta.fps}")
    print(f"  Features: {list(source_meta.features.keys())}")

    # --- Initialize pose estimator ---
    estimator = PoseEstimator(
        landmarks=landmarks,
        include_velocity=include_velocity,
        include_acceleration=include_acceleration,
        smoothing_alpha=smoothing_alpha,
        use_3d=use_3d,
    )
    keypoint_dim = estimator.feature_dim
    keypoint_names = estimator.feature_names

    robot_state_shape = source_meta.features["observation.state"]["shape"][0]
    new_state_dim = robot_state_shape + keypoint_dim

    print(f"\nKeypoint configuration:")
    print(f"  Landmarks: {landmarks} ({len(estimator._sorted_names)} points)")
    print(f"  Coordinates: {'3D' if use_3d else '2D'}")
    print(f"  Velocity: {include_velocity}")
    print(f"  Acceleration: {include_acceleration}")
    print(f"  Keypoint features: {keypoint_dim}")
    print(f"  Robot state: {robot_state_shape}")
    print(f"  New state dim: {new_state_dim}")

    # --- Build feature spec for target dataset ---
    robot_state_names = source_meta.features["observation.state"].get("names", [])
    if isinstance(robot_state_names, list) and len(robot_state_names) > 0:
        all_state_names = list(robot_state_names) + keypoint_names
    else:
        all_state_names = [f"joint_{i}" for i in range(robot_state_shape)] + keypoint_names

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [new_state_dim],
            "names": all_state_names,
        },
        "action": source_meta.features["action"],
    }

    if keep_images:
        for key in source_meta.features:
            if key.startswith("observation.images"):
                features[key] = source_meta.features[key]

    # --- Load source dataset (all episodes) ---
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.datasets.factory import resolve_delta_timestamps

    config = ACTConfig(chunk_size=100)
    delta_timestamps = resolve_delta_timestamps(config, source_meta)

    source_kwargs = {
        "repo_id": source_repo_id,
        "delta_timestamps": delta_timestamps,
        "tolerance_s": 0.0001,
    }
    if source_root:
        source_kwargs["root"] = source_root
    source_dataset = LeRobotDataset(**source_kwargs)

    camera_key = None
    for key in source_meta.features:
        if key.startswith("observation.images"):
            camera_key = key
            break

    if camera_key is None:
        print("ERROR: No image feature found in source dataset.")
        sys.exit(1)
    print(f"  Camera key: {camera_key}")

    # --- Create target dataset ---
    if target_root is None:
        target_root = str(Path("data") / target_repo_id.replace("/", "_"))

    print(f"\nCreating target dataset: {target_repo_id}")
    print(f"  Root: {target_root}")

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        root=target_root,
        fps=source_meta.fps,
        features=features,
        robot_type=source_meta.robot_type,
    )

    # --- Process each episode ---
    total_frames = 0
    total_failures = 0

    for ep_idx in range(source_meta.total_episodes):
        estimator.reset()

        ep_start = source_dataset.episode_data_index["from"][ep_idx].item()
        ep_end = source_dataset.episode_data_index["to"][ep_idx].item()
        ep_length = ep_end - ep_start

        ep_failures = 0

        for frame_offset, global_idx in enumerate(range(ep_start, ep_end)):
            sample = source_dataset[global_idx]

            robot_state = sample["observation.state"].numpy()
            if robot_state.ndim > 1:
                robot_state = robot_state[0]

            image_tensor = sample[camera_key]
            rgb = image_tensor_to_rgb(image_tensor)
            keypoint_features = estimator.process_frame(rgb)

            extended_state = np.concatenate([robot_state[:robot_state_shape], keypoint_features])

            frame = {
                "observation.state": torch.from_numpy(extended_state).float(),
                "action": sample["action"],
            }

            if keep_images:
                for key in features:
                    if key.startswith("observation.images") and key in sample:
                        frame[key] = sample[key]

            target_dataset.add_frame(frame)

            if frame_offset % batch_size_display == 0 and frame_offset > 0:
                print(f"    Episode {ep_idx}: {frame_offset}/{ep_length} frames", end="\r")

        ep_failures = estimator.detection_failure_count - total_failures
        total_failures = estimator.detection_failure_count

        task_str = "Gesture mimic"
        if hasattr(source_meta, "tasks") and source_meta.tasks:
            if isinstance(source_meta.tasks, dict):
                task_str = list(source_meta.tasks.values())[0]
            elif isinstance(source_meta.tasks, list) and len(source_meta.tasks) > 0:
                task_str = source_meta.tasks[0]

        target_dataset.save_episode(task=task_str)
        total_frames += ep_length

        fail_pct = (ep_failures / ep_length * 100) if ep_length > 0 else 0
        status = "OK" if fail_pct < 20 else "WARN"
        print(f"  Episode {ep_idx:3d}: {ep_length:5d} frames, {ep_failures:3d} detection failures ({fail_pct:.1f}%) [{status}]")

    # --- Summary ---
    print(f"\nPreprocessing complete!")
    print(f"  Total frames: {total_frames}")
    print(f"  Total detection failures: {total_failures} ({total_failures / max(total_frames, 1) * 100:.1f}%)")
    print(f"  Target dataset: {target_root}")

    # Save preprocessing config for reproducibility
    config_path = Path(target_root) / "meta" / "keypoint_config.json"
    config_data = {
        "source_repo_id": source_repo_id,
        "landmarks": landmarks if isinstance(landmarks, str) else dict(landmarks),
        "include_velocity": include_velocity,
        "include_acceleration": include_acceleration,
        "smoothing_alpha": smoothing_alpha,
        "use_3d": use_3d,
        "keep_images": keep_images,
        "keypoint_dim": keypoint_dim,
        "robot_state_dim": robot_state_shape,
        "total_state_dim": new_state_dim,
        "feature_names": all_state_names,
        "detection_failures": total_failures,
        "total_frames": total_frames,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Keypoint config: {config_path}")

    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {target_repo_id}")
        target_dataset.push_to_hub(repo_id=target_repo_id)
        print("  Done!")

    estimator.close()
    return target_root


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess gesture mimic dataset: extract pose keypoints from video"
    )
    parser.add_argument("--source", required=True, help="Source dataset repo_id (e.g. BlankHead/extended_gesture_mimic)")
    parser.add_argument("--target", required=True, help="Target dataset repo_id (e.g. local/gesture_mimic_keypoints)")
    parser.add_argument("--target-root", default=None, help="Local root path for target dataset")
    parser.add_argument("--source-root", default=None, help="Local root path for source dataset")

    parser.add_argument("--landmarks", default="upper_body",
                        choices=list(LANDMARK_PRESETS.keys()),
                        help="Landmark preset to use")
    parser.add_argument("--no-velocity", action="store_true", help="Disable velocity features")
    parser.add_argument("--no-acceleration", action="store_true", help="Disable acceleration features")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha (0=off)")
    parser.add_argument("--use-2d", action="store_true", help="Use 2D coordinates instead of 3D")
    parser.add_argument("--keep-images", action="store_true", help="Keep image features alongside keypoints")
    parser.add_argument("--push-to-hub", action="store_true", help="Push target dataset to HuggingFace Hub")

    args = parser.parse_args()

    preprocess_dataset(
        source_repo_id=args.source,
        target_repo_id=args.target,
        target_root=args.target_root,
        source_root=args.source_root,
        landmarks=args.landmarks,
        include_velocity=not args.no_velocity,
        include_acceleration=not args.no_acceleration,
        smoothing_alpha=args.smoothing,
        use_3d=not args.use_2d,
        keep_images=args.keep_images,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
