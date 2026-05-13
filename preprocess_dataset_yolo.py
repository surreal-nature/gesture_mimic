"""Preprocess a gesture mimic dataset: segment person + overlay arm skeleton.

Takes an existing LeRobot RGB dataset and creates a new dataset where
every image frame has the background removed (YOLOv26n-seg) and 6 arm
keypoints overlaid as colored circles with an inverted-U skeleton
(YOLOv26n-pose).

Unlike preprocess_dataset.py (MediaPipe) which removes images and extends
state, this script keeps the exact same dataset schema — only the image
pixel content changes. The output dataset trains with the standard RGB
ACT pipeline (ResNet-18 vision backbone).

Usage:
    # Basic: preprocess the multigesture-mimic dataset
    python preprocess_dataset_yolo.py \
        --source AmolSapale181284/multigesture-mimic \
        --target local/gesture_mimic_yolo

    # With custom model paths
    python preprocess_dataset_yolo.py \
        --source BlankHead/extended_gesture_mimic \
        --target local/extended_gesture_mimic_yolo \
        --seg-model /path/to/yolo26n-seg.pt \
        --pose-model /path/to/yolo26n-pose.pt

    # On CPU (slower but no GPU needed)
    python preprocess_dataset_yolo.py \
        --source AmolSapale181284/multigesture-mimic \
        --target local/gesture_mimic_yolo \
        --device cpu

    # Push to HuggingFace Hub
    python preprocess_dataset_yolo.py \
        --source AmolSapale181284/multigesture-mimic \
        --target myuser/gesture_mimic_yolo \
        --push-to-hub
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from yolo_preprocessor import YoloPosePreprocessor


def image_tensor_to_rgb(tensor):
    """Convert a LeRobot image tensor to uint8 RGB numpy array."""
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
        arr = tensor.permute(1, 2, 0).numpy()
    else:
        arr = tensor.numpy()
    if arr.dtype in (np.float32, np.float64):
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def rgb_to_image_tensor(rgb):
    """Convert uint8 RGB numpy array to HWC uint8 tensor (LeRobot v3 format)."""
    return torch.from_numpy(rgb.copy())


def preprocess_dataset_yolo(
    source_repo_id,
    target_repo_id,
    target_root=None,
    source_root=None,
    seg_model="yolo26n-seg.pt",
    pose_model="yolo26n-pose.pt",
    device="cuda",
    confidence_threshold=0.5,
    push_to_hub=False,
):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"Loading source dataset: {source_repo_id}")
    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    print(f"  Episodes: {source_meta.total_episodes}")
    print(f"  Frames: {source_meta.total_frames}")
    print(f"  FPS: {source_meta.fps}")
    print(f"  Features: {list(source_meta.features.keys())}")

    camera_key = None
    for key in source_meta.features:
        if key.startswith("observation.images"):
            camera_key = key
            break
    if camera_key is None:
        raise ValueError("No image feature found in source dataset")
    print(f"  Camera key: {camera_key}")

    preprocessor = YoloPosePreprocessor(
        seg_model=seg_model,
        pose_model=pose_model,
        device=device,
        confidence_threshold=confidence_threshold,
    )

    features = {}
    for key, feat in source_meta.features.items():
        features[key] = dict(feat)

    source_kwargs = {
        "repo_id": source_repo_id,
    }
    if source_root:
        source_kwargs["root"] = source_root
    source_dataset = LeRobotDataset(**source_kwargs)

    if target_root is None:
        target_root = str(Path("data") / target_repo_id.replace("/", "_"))

    print(f"\nCreating target dataset: {target_repo_id}")
    print(f"  Root: {target_root}")
    print(f"  Seg model: {seg_model}")
    print(f"  Pose model: {pose_model}")
    print(f"  Device: {device}")

    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        root=target_root,
        fps=source_meta.fps,
        features=features,
        robot_type=source_meta.robot_type,
    )

    total_frames = 0
    total_failures = 0

    episodes_info = source_meta.episodes
    for ep_idx in range(source_meta.total_episodes):
        ep_start = episodes_info[ep_idx]["dataset_from_index"]
        ep_end = episodes_info[ep_idx]["dataset_to_index"]
        ep_length = ep_end - ep_start
        failures_before = preprocessor.detection_failure_count

        task_str = "Gesture mimic"
        ep_tasks = episodes_info[ep_idx].get("tasks")
        if ep_tasks and isinstance(ep_tasks, list) and len(ep_tasks) > 0:
            task_str = ep_tasks[0]

        for frame_offset, global_idx in enumerate(range(ep_start, ep_end)):
            sample = source_dataset[global_idx]

            image_tensor = sample[camera_key]
            rgb = image_tensor_to_rgb(image_tensor)
            modified_rgb = preprocessor.process_frame(rgb)
            modified_tensor = rgb_to_image_tensor(modified_rgb)

            frame = {
                "task": task_str,
                "observation.state": sample["observation.state"],
                "action": sample["action"],
                camera_key: modified_tensor,
            }
            target_dataset.add_frame(frame)

            if frame_offset % 100 == 0 and frame_offset > 0:
                print(f"    Episode {ep_idx}: {frame_offset}/{ep_length} frames", end="\r")

        target_dataset.save_episode()

        ep_failures = preprocessor.detection_failure_count - failures_before
        total_failures = preprocessor.detection_failure_count
        total_frames += ep_length

        fail_pct = (ep_failures / ep_length * 100) if ep_length > 0 else 0
        status = "OK" if fail_pct < 20 else "WARN"
        print(f"  Episode {ep_idx:3d}: {ep_length:5d} frames, "
              f"{ep_failures:3d} detection failures ({fail_pct:.1f}%) [{status}]")

    print(f"\nPreprocessing complete!")
    print(f"  Total frames: {total_frames}")
    print(f"  Detection failures: {total_failures} "
          f"({total_failures / max(total_frames, 1) * 100:.1f}%)")
    print(f"  Target dataset: {target_root}")

    config_path = Path(target_root) / "meta" / "yolo_preprocess_config.json"
    config_data = {
        "source_repo_id": source_repo_id,
        "seg_model": seg_model,
        "pose_model": pose_model,
        "confidence_threshold": confidence_threshold,
        "detection_failures": total_failures,
        "total_frames": total_frames,
        "device": device,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Config saved: {config_path}")

    if push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {target_repo_id}")
        target_dataset.push_to_hub(repo_id=target_repo_id)
        print("  Done!")

    preprocessor.close()
    return target_root


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess gesture mimic dataset: YOLO segmentation + arm skeleton overlay"
    )
    parser.add_argument("--source", required=True,
                        help="Source dataset repo_id (e.g. AmolSapale181284/multigesture-mimic)")
    parser.add_argument("--target", required=True,
                        help="Target dataset repo_id (e.g. local/gesture_mimic_yolo)")
    parser.add_argument("--target-root", default=None,
                        help="Local root path for target dataset")
    parser.add_argument("--source-root", default=None,
                        help="Local root path for source dataset")
    parser.add_argument("--seg-model", default="yolo26n-seg.pt",
                        help="YOLOv26n-seg model name or path")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt",
                        help="YOLOv26n-pose model name or path")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device for YOLO inference")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Minimum keypoint confidence threshold")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push target dataset to HuggingFace Hub")

    args = parser.parse_args()

    preprocess_dataset_yolo(
        source_repo_id=args.source,
        target_repo_id=args.target,
        target_root=args.target_root,
        source_root=args.source_root,
        seg_model=args.seg_model,
        pose_model=args.pose_model,
        device=args.device,
        confidence_threshold=args.confidence,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
