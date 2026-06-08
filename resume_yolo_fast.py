"""Fast resume of YOLO preprocessing using direct file copy + LeRobot resume API.

Step 1: Repair the corrupted target dataset by rebuilding parquets from
        source data (state/action are identical — YOLO only changes images).
Step 2: Use LeRobotDataset.resume() to append remaining episodes with YOLO.

Usage:
    python resume_yolo_fast.py \
        --source local/gesture_mimic_merged_v2 \
        --source-root data/local_gesture_mimic_merged_v2 \
        --target local/gesture_mimic_merged_v2_yolo \
        --target-root data/local_gesture_mimic_merged_v2_yolo \
        --start-episode 566
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def repair_target_parquets(source_root, target_root, start_episode):
    """Rebuild target data + episodes parquets from source (for episodes 0..start_episode-1)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    source_root = Path(source_root)
    target_root = Path(target_root)

    print("=== Step 1: Repairing target dataset parquets ===")

    source_meta = LeRobotDatasetMetadata(
        "local/source_tmp", root=source_root
    )
    last_frame = source_meta.episodes[start_episode - 1]["dataset_to_index"]
    print(f"  Source episodes: {source_meta.total_episodes}")
    print(f"  Target episodes (to repair): {start_episode}")
    print(f"  Target frames: {last_frame}")

    # --- Repair data parquet ---
    # Read source data parquet, filter to episodes 0..start_episode-1
    source_data_parquet = source_root / "data" / "chunk-000" / "file-000.parquet"
    print(f"  Reading source data: {source_data_parquet}")
    source_table = pq.read_table(source_data_parquet)
    source_df = source_table.to_pandas()

    target_df = source_df[source_df["episode_index"] < start_episode].copy()
    target_df = target_df.reset_index(drop=True)
    target_df["index"] = range(len(target_df))
    print(f"  Filtered data: {len(target_df)} frames (episodes 0-{start_episode - 1})")

    target_data_dir = target_root / "data" / "chunk-000"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    target_data_parquet = target_data_dir / "file-000.parquet"

    # Back up corrupted file
    if target_data_parquet.exists():
        backup = target_data_parquet.with_suffix(".parquet.corrupt_bak")
        shutil.move(str(target_data_parquet), str(backup))
        print(f"  Backed up corrupted data parquet to {backup.name}")

    target_table = pa.Table.from_pandas(target_df, preserve_index=False)
    pq.write_table(target_table, target_data_parquet)
    print(f"  Wrote repaired data parquet: {target_data_parquet}")

    # --- Repair episodes parquet ---
    source_episodes_parquet = source_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    print(f"  Reading source episodes: {source_episodes_parquet}")
    source_ep_table = pq.read_table(source_episodes_parquet)
    source_ep_df = source_ep_table.to_pandas()

    target_ep_df = source_ep_df[source_ep_df["episode_index"] < start_episode].copy()

    # Fix dataset_from/to indices and video file references
    # The target dataset had its own video file structure (5 files for 566 episodes)
    # We need to reconstruct the episode metadata to match the target's video layout
    # For now, copy the video file mapping from the target's existing structure
    # The video files in the target have their own chunk/file indices
    # Since we can't read the corrupted episodes parquet, we'll reconstruct from scratch

    # Read target video file structure to figure out episode-to-video mapping
    target_video_dir = target_root / "videos" / "observation.images.human" / "chunk-000"
    import av
    video_frames = {}
    for vf in sorted(target_video_dir.glob("*.mp4")):
        container = av.open(str(vf))
        n = container.streams.video[0].frames
        fi = int(vf.stem.split("-")[1])
        video_frames[fi] = n
        container.close()
        print(f"  Target video file-{fi:03d}.mp4: {n} frames")

    # Map episodes to video files based on cumulative frame counts
    cumulative = 0
    video_boundaries = []
    for fi in sorted(video_frames.keys()):
        start_frame = cumulative
        cumulative += video_frames[fi]
        video_boundaries.append((fi, start_frame, cumulative))

    def find_video_file(frame_from, frame_to):
        for fi, vstart, vend in video_boundaries:
            if frame_from >= vstart and frame_from < vend:
                return fi
        return video_boundaries[-1][0]

    # Rebuild episode metadata with proper video references
    # The target episodes have sequential frame indices starting from 0
    frame_offset = 0
    for idx in range(len(target_ep_df)):
        row_idx = target_ep_df.index[idx]
        ep_length = source_meta.episodes[idx]["dataset_to_index"] - source_meta.episodes[idx]["dataset_from_index"]
        ep_from = frame_offset
        ep_to = frame_offset + ep_length

        target_ep_df.loc[row_idx, "dataset_from_index"] = ep_from
        target_ep_df.loc[row_idx, "dataset_to_index"] = ep_to

        vid_fi = find_video_file(ep_from, ep_to)
        target_ep_df.loc[row_idx, "videos/observation.images.human/chunk_index"] = 0
        target_ep_df.loc[row_idx, "videos/observation.images.human/file_index"] = vid_fi

        frame_offset = ep_to

    target_ep_dir = target_root / "meta" / "episodes" / "chunk-000"
    target_ep_dir.mkdir(parents=True, exist_ok=True)
    target_ep_parquet = target_ep_dir / "file-000.parquet"

    if target_ep_parquet.exists():
        backup = target_ep_parquet.with_suffix(".parquet.corrupt_bak")
        shutil.move(str(target_ep_parquet), str(backup))
        print(f"  Backed up corrupted episodes parquet to {backup.name}")

    target_ep_table = pa.Table.from_pandas(target_ep_df, preserve_index=False)
    pq.write_table(target_ep_table, target_ep_parquet)
    print(f"  Wrote repaired episodes parquet: {target_ep_parquet}")

    # --- Fix info.json ---
    info_path = target_root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_episodes"] = start_episode
    info["total_frames"] = last_frame
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Updated info.json: {start_episode} episodes, {last_frame} frames")

    # Verify the repaired dataset loads
    print("\n  Verifying repaired dataset...")
    target_meta = LeRobotDatasetMetadata(
        "local/gesture_mimic_merged_v2_yolo", root=target_root
    )
    print(f"  Verified: {target_meta.total_episodes} episodes, {target_meta.total_frames} frames")
    return target_meta


def resume_yolo_processing(source_repo_id, source_root, target_repo_id, target_root,
                           start_episode, seg_model, pose_model, device, confidence):
    """Use LeRobotDataset.resume() to append YOLO-processed episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    print(f"\n=== Step 2: YOLO processing episodes {start_episode}+ ===")

    source_kwargs = {"repo_id": source_repo_id}
    if source_root:
        source_kwargs["root"] = source_root
    source_meta = LeRobotDatasetMetadata(source_repo_id, root=source_root)
    source_dataset = LeRobotDataset(**source_kwargs)
    total_episodes = source_meta.total_episodes

    camera_key = None
    for key in source_meta.features:
        if key.startswith("observation.images"):
            camera_key = key
            break
    print(f"  Camera key: {camera_key}")
    print(f"  Processing episodes {start_episode} to {total_episodes - 1}")

    preprocessor = YoloPosePreprocessor(
        seg_model=seg_model,
        pose_model=pose_model,
        device=device,
        confidence_threshold=confidence,
    )

    target_dataset = LeRobotDataset.resume(
        repo_id=target_repo_id,
        root=str(target_root),
    )
    print(f"  Resumed target dataset: {target_dataset.meta.total_episodes} episodes, "
          f"{target_dataset.meta.total_frames} frames")

    total_frames = 0
    total_failures = 0
    episodes_info = source_meta.episodes

    for ep_idx in range(start_episode, total_episodes):
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

    print(f"\nYOLO processing complete!")
    print(f"  New frames: {total_frames}")
    print(f"  Detection failures: {total_failures} "
          f"({total_failures / max(total_frames, 1) * 100:.1f}%)")
    print(f"  Total dataset: {target_dataset.meta.total_episodes} episodes, "
          f"{target_dataset.meta.total_frames} frames")

    config_path = Path(target_root) / "meta" / "yolo_preprocess_config.json"
    config_data = {
        "source_repo_id": source_repo_id,
        "seg_model": seg_model,
        "pose_model": pose_model,
        "confidence_threshold": confidence,
        "detection_failures": total_failures,
        "total_frames": target_dataset.meta.total_frames,
        "device": device,
        "resumed_from_episode": start_episode,
    }
    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=2)
    print(f"  Config saved: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Fast resume YOLO preprocessing")
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--target", required=True, help="Target dataset repo_id")
    parser.add_argument("--target-root", required=True, help="Existing target root with intact videos")
    parser.add_argument("--start-episode", type=int, required=True)
    parser.add_argument("--seg-model", default="yolo26n-seg.pt")
    parser.add_argument("--pose-model", default="yolo26n-pose.pt")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    # Step 1: Repair corrupted parquets
    repair_target_parquets(
        source_root=args.source_root,
        target_root=args.target_root,
        start_episode=args.start_episode,
    )

    # Step 2: Resume YOLO processing
    resume_yolo_processing(
        source_repo_id=args.source,
        source_root=args.source_root,
        target_repo_id=args.target,
        target_root=args.target_root,
        start_episode=args.start_episode,
        seg_model=args.seg_model,
        pose_model=args.pose_model,
        device=args.device,
        confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
