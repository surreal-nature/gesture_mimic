"""Resume YOLO preprocessing from a specific episode.

Reuses already-processed video frames from a partially-completed target
dataset, and only runs YOLO inference on remaining episodes.

Usage:
    python resume_yolo_preprocess.py \
        --source local/gesture_mimic_merged_v2 \
        --source-root data/local_gesture_mimic_merged_v2 \
        --existing-target-root data/local_gesture_mimic_merged_v2_yolo \
        --new-target local/gesture_mimic_merged_v2_yolo \
        --new-target-root data/local_gesture_mimic_merged_v2_yolo_full \
        --start-episode 566
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch

from yolo_preprocessor import YoloPosePreprocessor


def image_tensor_to_rgb(tensor):
    if tensor.dtype == torch.uint8:
        return tensor.numpy()
    arr = tensor.permute(1, 2, 0).numpy()
    if arr.max() <= 1.0:
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def rgb_to_image_tensor(rgb):
    return torch.from_numpy(rgb.copy())


class ExistingVideoFrameReader:
    """Sequentially reads frames from existing mp4 files in a target dataset."""

    def __init__(self, video_dir):
        self.video_dir = Path(video_dir)
        self._files = sorted(self.video_dir.glob("*.mp4"))
        self._current_file_idx = 0
        self._container = None
        self._decoder = None
        self._open_next_file()

    def _open_next_file(self):
        if self._container:
            self._container.close()
        if self._current_file_idx < len(self._files):
            path = self._files[self._current_file_idx]
            self._container = av.open(str(path))
            self._decoder = self._container.decode(video=0)
        else:
            self._container = None
            self._decoder = None

    def read_frame(self):
        while self._decoder is not None:
            try:
                frame = next(self._decoder)
                return frame.to_ndarray(format="rgb24")
            except StopIteration:
                self._current_file_idx += 1
                self._open_next_file()
        raise RuntimeError("No more frames in existing video files")

    def close(self):
        if self._container:
            self._container.close()


def main():
    parser = argparse.ArgumentParser(description="Resume YOLO preprocessing from a specific episode")
    parser.add_argument("--source", required=True, help="Source dataset repo_id")
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--existing-target-root", required=True,
                        help="Root of the partially-completed YOLO dataset (with intact video files)")
    parser.add_argument("--new-target", required=True, help="New target dataset repo_id")
    parser.add_argument("--new-target-root", required=True, help="Root for the new complete target dataset")
    parser.add_argument("--start-episode", type=int, required=True,
                        help="First episode that needs YOLO processing (all before are read from existing videos)")
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"Loading source dataset: {args.source}")
    source_kwargs = {"repo_id": args.source}
    if args.source_root:
        source_kwargs["root"] = args.source_root
    source_meta = LeRobotDatasetMetadata(args.source, root=args.source_root)
    source_dataset = LeRobotDataset(**source_kwargs)
    print(f"  Episodes: {source_meta.total_episodes}, Frames: {source_meta.total_frames}")

    camera_key = None
    for key in source_meta.features:
        if key.startswith("observation.images"):
            camera_key = key
            break
    if camera_key is None:
        raise ValueError("No image feature found in source dataset")
    print(f"  Camera key: {camera_key}")

    features = {}
    for key, feat in source_meta.features.items():
        features[key] = dict(feat)

    existing_video_dir = Path(args.existing_target_root) / "videos" / camera_key / "chunk-000"
    print(f"\nExisting video dir: {existing_video_dir}")
    video_reader = ExistingVideoFrameReader(existing_video_dir)

    preprocessor = YoloPosePreprocessor(
        seg_model=args.seg_model,
        pose_model=args.pose_model,
        device=args.device,
        confidence_threshold=args.confidence,
    )

    print(f"\nCreating new target dataset: {args.new_target}")
    print(f"  Root: {args.new_target_root}")
    target_dataset = LeRobotDataset.create(
        repo_id=args.new_target,
        root=args.new_target_root,
        fps=source_meta.fps,
        features=features,
        robot_type=source_meta.robot_type,
    )

    start_ep = args.start_episode
    total_episodes = source_meta.total_episodes
    total_frames = 0
    total_failures = 0

    episodes_info = source_meta.episodes

    # Phase 1: Copy frames using existing processed video files (no YOLO needed)
    print(f"\n=== Phase 1: Copying {start_ep} episodes from existing videos ===")
    for ep_idx in range(start_ep):
        ep_start = episodes_info[ep_idx]["dataset_from_index"]
        ep_end = episodes_info[ep_idx]["dataset_to_index"]
        ep_length = ep_end - ep_start

        task_str = "Gesture mimic"
        ep_tasks = episodes_info[ep_idx].get("tasks")
        if ep_tasks and isinstance(ep_tasks, list) and len(ep_tasks) > 0:
            task_str = ep_tasks[0]

        for frame_offset, global_idx in enumerate(range(ep_start, ep_end)):
            sample = source_dataset[global_idx]
            existing_rgb = video_reader.read_frame()
            modified_tensor = rgb_to_image_tensor(existing_rgb)

            frame = {
                "task": task_str,
                "observation.state": sample["observation.state"],
                "action": sample["action"],
                camera_key: modified_tensor,
            }
            target_dataset.add_frame(frame)

        target_dataset.save_episode()
        total_frames += ep_length

        if (ep_idx + 1) % 50 == 0 or ep_idx == start_ep - 1:
            print(f"  Copied episode {ep_idx + 1}/{start_ep} ({total_frames} frames)")

    video_reader.close()

    # Phase 2: YOLO processing for remaining episodes
    print(f"\n=== Phase 2: YOLO processing episodes {start_ep}-{total_episodes - 1} ===")
    for ep_idx in range(start_ep, total_episodes):
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

    preprocessor.close()

    print(f"\nPreprocessing complete!")
    print(f"  Total frames: {total_frames}")
    print(f"  Detection failures (YOLO phase): {total_failures} "
          f"({total_failures / max(total_frames - episodes_info[start_ep - 1]['dataset_to_index'], 1) * 100:.1f}%)")
    print(f"  Target dataset: {args.new_target_root}")

    config_path = Path(args.new_target_root) / "meta" / "yolo_preprocess_config.json"
    config_data = {
        "source_repo_id": args.source,
        "seg_model": args.seg_model,
        "pose_model": args.pose_model,
        "confidence_threshold": args.confidence,
        "detection_failures": total_failures,
        "total_frames": total_frames,
        "device": args.device,
        "resumed_from_episode": start_ep,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Config saved: {config_path}")


if __name__ == "__main__":
    main()
