#!/bin/bash
# Training script for gesture mimic ACT policy.
#
# Trains an ACT policy to mimic human gestures observed via camera.
# The robot (SO-101 follower, 6-DOF) learns to reproduce arm/hand
# movements demonstrated by a human in the camera frame.
#
# Usage:
#   bash train.sh                                    # defaults (extended dataset)
#   bash train.sh --steps 50000                      # override steps
#   DATASET=AmolSapale181284/multigesture-mimic bash train.sh  # use original dataset
#   SLURM: sbatch train.sh
#
# Fine-tuning from a previous checkpoint:
#   PRETRAINED=outputs/train/act_gesture/checkpoints/last/pretrained_model bash train.sh
#
# RGB mode with ROCm (AMD GPU):
#   export MIOPEN_FIND_ENFORCE=5 MIOPEN_FIND_MODE=2
#   bash train.sh
#
# Keypoint mode — MediaPipe (train on preprocessed keypoint dataset):
#   First preprocess:  python preprocess_dataset.py --source BlankHead/extended_gesture_mimic \
#                          --target local/gesture_kp
#   Then train:        USE_KEYPOINTS=true bash train.sh
#   Or manually:       DATASET=local/gesture_kp DATASET_ROOT=data/local_gesture_kp bash train.sh
#
# Keypoint mode — YOLO (segmented RGB + arm skeleton overlay):
#   First preprocess:  python preprocess_dataset_yolo.py --source AmolSapale181284/multigesture-mimic \
#                          --target local/gesture_mimic_yolo
#   Then train:        USE_KEYPOINTS=true POSE_BACKEND=yolo bash train.sh

#SBATCH --job-name=act_gesture
#SBATCH --partition=defq
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Configuration (override via environment variables) ---
DATASET="${DATASET:-BlankHead/extended_gesture_mimic}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/train/act_gesture}"
PRETRAINED="${PRETRAINED:-}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# ACT architecture
CHUNK_SIZE="${CHUNK_SIZE:-100}"
N_ACTION_STEPS="${N_ACTION_STEPS:-100}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
DIM_MODEL="${DIM_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-3200}"
N_ENCODER_LAYERS="${N_ENCODER_LAYERS:-4}"
N_DECODER_LAYERS="${N_DECODER_LAYERS:-1}"
LATENT_DIM="${LATENT_DIM:-32}"
N_VAE_ENCODER_LAYERS="${N_VAE_ENCODER_LAYERS:-4}"
DROPOUT="${DROPOUT:-0.1}"
KL_WEIGHT="${KL_WEIGHT:-10.0}"

# Optimizer
LR="${LR:-1e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
DECAY_LR="${DECAY_LR:-1e-7}"

# Logging / saving
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-${STEPS}}"
WANDB="${WANDB:-false}"

# Push to Hub
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
HUB_REPO="${HUB_REPO:-}"

# Keypoint mode
USE_KEYPOINTS="${USE_KEYPOINTS:-false}"
POSE_BACKEND="${POSE_BACKEND:-mediapipe}"
KP_DATASET="${KP_DATASET:-local/gesture_mimic_keypoints}"
KP_DATASET_ROOT="${KP_DATASET_ROOT:-${SCRIPT_DIR}/data/local_gesture_mimic_keypoints}"

# YOLO keypoint mode
YOLO_DATASET="${YOLO_DATASET:-local/gesture_mimic_yolo}"
YOLO_DATASET_ROOT="${YOLO_DATASET_ROOT:-${SCRIPT_DIR}/data/local_gesture_mimic_yolo}"

# --- Environment setup ---
conda activate act 2>/dev/null || true

# AMD ROCm optimizations
if command -v rocminfo &>/dev/null; then
    export MIOPEN_FIND_ENFORCE=5
    export MIOPEN_FIND_MODE=2
fi

mkdir -p "${SCRIPT_DIR}/logs"

# --- Build command ---
# --- Keypoint mode overrides ---
if [[ "${USE_KEYPOINTS}" == "true" ]]; then
    if [[ "${POSE_BACKEND}" == "yolo" ]]; then
        DATASET="${YOLO_DATASET}"
        DATASET_ROOT="${YOLO_DATASET_ROOT}"
        OUTPUT_DIR="${SCRIPT_DIR}/outputs/train/act_gesture_yolo"
    else
        DATASET="${KP_DATASET}"
        DATASET_ROOT="${KP_DATASET_ROOT}"
        OUTPUT_DIR="${SCRIPT_DIR}/outputs/train/act_gesture_kp"
    fi
fi

echo "=== Gesture Mimic ACT Training ==="
echo "Started: $(date)"
echo "Node: $(hostname)"
if [[ "${USE_KEYPOINTS}" == "true" && "${POSE_BACKEND}" == "yolo" ]]; then
    echo "Mode: YOLO (segmented RGB + skeleton overlay, vision backbone)"
elif [[ "${USE_KEYPOINTS}" == "true" ]]; then
    echo "Mode: KEYPOINT/MediaPipe (state-only, no vision backbone)"
else
    echo "Mode: RGB (vision backbone)"
fi
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "Steps: ${STEPS}, Batch: ${BATCH_SIZE}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>/dev/null || true

CMD=(
    lerobot-train
    --policy.type=act
    --dataset.repo_id="${DATASET}"
    --dataset.image_transforms.enable=true

    # ACT architecture
    --policy.chunk_size="${CHUNK_SIZE}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.vision_backbone="${VISION_BACKBONE}"
    --policy.dim_model="${DIM_MODEL}"
    --policy.n_heads="${N_HEADS}"
    --policy.dim_feedforward="${DIM_FEEDFORWARD}"
    --policy.n_encoder_layers="${N_ENCODER_LAYERS}"
    --policy.n_decoder_layers="${N_DECODER_LAYERS}"
    --policy.use_vae=true
    --policy.latent_dim="${LATENT_DIM}"
    --policy.n_vae_encoder_layers="${N_VAE_ENCODER_LAYERS}"
    --policy.dropout="${DROPOUT}"
    --policy.kl_weight="${KL_WEIGHT}"

    # Training
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --use_policy_training_preset=false
    --output_dir="${OUTPUT_DIR}"

    # Optimizer
    --optimizer.type=adamw
    --optimizer.lr="${LR}"
    --optimizer.weight_decay="${WEIGHT_DECAY}"
    --optimizer.grad_clip_norm="${GRAD_CLIP_NORM}"

    # Scheduler
    --scheduler.type=cosine_decay_with_warmup
    --scheduler.num_warmup_steps="${WARMUP_STEPS}"
    --scheduler.num_decay_steps="${STEPS}"
    --scheduler.peak_lr="${LR}"
    --scheduler.decay_lr="${DECAY_LR}"

    # Logging
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --wandb.enable="${WANDB}"

    # Disable push to hub by default (overridden below if requested)
    --policy.push_to_hub=false
)

# Optional: dataset root (for local/cached datasets)
if [[ -n "${DATASET_ROOT}" ]]; then
    CMD+=(--dataset.root="${DATASET_ROOT}")
fi

# Optional: fine-tune from pretrained checkpoint
if [[ -n "${PRETRAINED}" ]]; then
    echo "Fine-tuning from: ${PRETRAINED}"
    CMD+=(--policy.pretrained_path="${PRETRAINED}")
fi

# Optional: push to Hub
if [[ "${PUSH_TO_HUB}" == "true" && -n "${HUB_REPO}" ]]; then
    echo "Will push to: ${HUB_REPO}"
    CMD+=(--policy.push_to_hub=true --policy.repo_id="${HUB_REPO}")
fi

# Pass through any extra CLI arguments
CMD+=("$@")

echo "---"
"${CMD[@]}"
echo "=== Finished: $(date) ==="
