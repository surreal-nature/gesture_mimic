"""Merge multiple LeRobot datasets into a single dataset for training.

Since lerobot-train only accepts a single --dataset.repo_id, this script
merges multiple compatible datasets (same fps, robot_type, features) into
one local dataset that can be used directly for training.

Uses lerobot.datasets.aggregate.aggregate_datasets() under the hood.

Usage:
    # Merge the two new AMD-PAVS-AI datasets with the original dataset
    python merge_datasets.py \
        --sources AmolSapale181284/multigesture-mimic \
                  AMD-PAVS-AI/multigesture_mimic_test \
                  AMD-PAVS-AI/Action-per-video-multigesture-mimic \
        --target local/gesture_mimic_merged

    # Merge all known gesture mimic datasets
    python merge_datasets.py \
        --sources AmolSapale181284/multigesture-mimic \
                  BlankHead/extended_gesture_mimic \
                  AMD-PAVS-AI/multigesture_mimic_test \
                  AMD-PAVS-AI/Action-per-video-multigesture-mimic \
        --target local/gesture_mimic_all

    # Custom output location
    python merge_datasets.py \
        --sources AMD-PAVS-AI/multigesture_mimic_test \
                  AMD-PAVS-AI/Action-per-video-multigesture-mimic \
        --target local/amd_pavs_merged \
        --target-root ./data/amd_pavs_merged

    # Push merged dataset to HuggingFace Hub
    python merge_datasets.py \
        --sources AmolSapale181284/multigesture-mimic \
                  AMD-PAVS-AI/Action-per-video-multigesture-mimic \
        --target myuser/gesture_mimic_combined \
        --push-to-hub
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def ensure_dataset_downloaded(repo_id):
    """Download a dataset from HuggingFace Hub, handling datasets without version tags.

    LeRobot's aggregate_datasets expects datasets to have version tags (e.g. v3.0).
    Some datasets on HuggingFace don't have these tags. This function downloads them
    using revision="main" as a fallback and returns the local root path.
    """
    from huggingface_hub import HfApi, snapshot_download
    from lerobot.utils.constants import HF_LEROBOT_HOME

    local_root = HF_LEROBOT_HOME / repo_id
    if (local_root / "meta" / "info.json").exists():
        logger.info("  Already cached: %s", repo_id)
        return local_root

    api = HfApi()
    refs = api.list_repo_refs(repo_id, repo_type="dataset")
    tags = [t.name for t in refs.tags]

    if tags:
        revision = tags[0]
    else:
        logger.info("  No version tags found for %s, using revision='main'", repo_id)
        revision = "main"

    logger.info("  Downloading %s (revision=%s)...", repo_id, revision)
    snapshot_download(
        repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_root,
    )
    return local_root


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple LeRobot datasets into a single dataset for training.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source dataset repo_ids to merge",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Target dataset repo_id (e.g. local/gesture_mimic_merged)",
    )
    parser.add_argument(
        "--target-root",
        default=None,
        help="Local root path for the merged dataset (default: data/<target_slug>)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to HuggingFace Hub after creation",
    )

    args = parser.parse_args()

    if len(args.sources) < 2:
        parser.error("At least 2 source datasets are required for merging")

    target_root = args.target_root
    if target_root is None:
        slug = args.target.replace("/", "_")
        target_root = SCRIPT_DIR / "data" / slug
    target_root = Path(target_root)

    if target_root.exists():
        logger.warning("Target directory already exists: %s", target_root)
        response = input("Delete and recreate? [y/N] ").strip().lower()
        if response == "y":
            shutil.rmtree(target_root)
        else:
            logger.info("Aborted.")
            sys.exit(0)

    logger.info("Merging %d datasets into %s", len(args.sources), args.target)
    for src in args.sources:
        logger.info("  Source: %s", src)
    logger.info("  Target: %s (root: %s)", args.target, target_root)

    logger.info("Downloading source datasets...")
    source_roots = []
    for repo_id in args.sources:
        root = ensure_dataset_downloaded(repo_id)
        source_roots.append(root)

    from lerobot.datasets.aggregate import aggregate_datasets

    aggregate_datasets(
        repo_ids=args.sources,
        roots=source_roots,
        aggr_repo_id=args.target,
        aggr_root=target_root,
    )

    logger.info("Merged dataset created at: %s", target_root)
    logger.info("")
    logger.info("To train on the merged dataset:")
    logger.info("  DATASET=%s DATASET_ROOT=%s bash train.sh", args.target, target_root)
    logger.info("")
    logger.info("To preprocess for keypoint mode:")
    logger.info(
        "  python preprocess_dataset.py --source %s --source-root %s --target local/merged_keypoints",
        args.target,
        target_root,
    )
    logger.info("To preprocess for YOLO keypoint mode:")
    logger.info(
        "  python preprocess_dataset_yolo.py --source %s --source-root %s --target local/merged_yolo",
        args.target,
        target_root,
    )

    if args.push_to_hub:
        logger.info("Pushing merged dataset to HuggingFace Hub: %s", args.target)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(args.target, root=target_root)
        dataset.push_to_hub(repo_id=args.target)
        logger.info("Pushed to Hub: https://huggingface.co/datasets/%s", args.target)


if __name__ == "__main__":
    main()
