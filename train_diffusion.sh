#!/bin/bash
# Training script for gesture mimic Diffusion Policy.
#
# Trains a Diffusion Policy to mimic human gestures observed via camera.
# Uses a 1D conditional UNet with DDPM/DDIM noise scheduling to generate
# smooth future action trajectories via iterative denoising.
#
# The same datasets used for ACT training work here — no data recollection needed.
#
# Usage:
#   bash train_diffusion.sh                              # defaults (extended dataset, RGB mode)
#   bash train_diffusion.sh --steps 50000                # override steps
#   DATASET=AmolSapale181284/multigesture-mimic bash train_diffusion.sh
#   DATASET=AMD-PAVS-AI/Action-per-video-multigesture-mimic bash train_diffusion.sh
#
# Merged dataset:
#   DATASET=local/gesture_mimic_merged DATASET_ROOT=data/local_gesture_mimic_merged bash train_diffusion.sh
#
# Fine-tuning from a previous checkpoint:
#   PRETRAINED=outputs/train/diffusion_gesture/checkpoints/last/pretrained_model bash train_diffusion.sh
#
# MediaPipe keypoint mode (train on preprocessed keypoint dataset):
#   First preprocess:  python preprocess_dataset.py --source BlankHead/extended_gesture_mimic \
#                          --target local/gesture_kp
#   Then train:        USE_KEYPOINTS=true bash train_diffusion.sh
#
# YOLO keypoint mode (segmented RGB + arm skeleton overlay):
#   First preprocess:  python preprocess_dataset_yolo.py --source AmolSapale181284/multigesture-mimic \
#                          --target local/gesture_mimic_yolo
#   Then train:        USE_KEYPOINTS=true POSE_BACKEND=yolo bash train_diffusion.sh
#
# SLURM: sbatch train_diffusion.sh

#SBATCH --job-name=diff_gesture
#SBATCH --partition=defq
#SBATCH --gres=gpu:gfx942-mi300x:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/train_diffusion_%j.log
#SBATCH --error=logs/train_diffusion_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- Configuration (override via environment variables) ---
DATASET="${DATASET:-BlankHead/extended_gesture_mimic}"
DATASET_ROOT="${DATASET_ROOT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/train/diffusion_gesture}"
PRETRAINED="${PRETRAINED:-}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-1000}"

# Diffusion architecture
N_OBS_STEPS="${N_OBS_STEPS:-2}"
HORIZON="${HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
DOWN_DIMS="${DOWN_DIMS:-[256,512,1024]}"
KERNEL_SIZE="${KERNEL_SIZE:-5}"
N_GROUPS="${N_GROUPS:-8}"
DIFFUSION_STEP_EMBED_DIM="${DIFFUSION_STEP_EMBED_DIM:-128}"
USE_FILM_SCALE_MODULATION="${USE_FILM_SCALE_MODULATION:-true}"
SPATIAL_SOFTMAX_NUM_KEYPOINTS="${SPATIAL_SOFTMAX_NUM_KEYPOINTS:-32}"
USE_GROUP_NORM="${USE_GROUP_NORM:-true}"

# Noise scheduler
NOISE_SCHEDULER_TYPE="${NOISE_SCHEDULER_TYPE:-DDPM}"
NUM_TRAIN_TIMESTEPS="${NUM_TRAIN_TIMESTEPS:-100}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
BETA_SCHEDULE="${BETA_SCHEDULE:-squaredcos_cap_v2}"
BETA_START="${BETA_START:-0.0001}"
BETA_END="${BETA_END:-0.02}"
PREDICTION_TYPE="${PREDICTION_TYPE:-epsilon}"
CLIP_SAMPLE="${CLIP_SAMPLE:-true}"
CLIP_SAMPLE_RANGE="${CLIP_SAMPLE_RANGE:-1.0}"

# Optimizer
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-6}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"

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

# --- Keypoint mode overrides ---
if [[ "${USE_KEYPOINTS}" == "true" ]]; then
    if [[ "${POSE_BACKEND}" == "yolo" ]]; then
        DATASET="${YOLO_DATASET}"
        DATASET_ROOT="${YOLO_DATASET_ROOT}"
        OUTPUT_DIR="${SCRIPT_DIR}/outputs/train/diffusion_gesture_yolo"
    else
        DATASET="${KP_DATASET}"
        DATASET_ROOT="${KP_DATASET_ROOT}"
        OUTPUT_DIR="${SCRIPT_DIR}/outputs/train/diffusion_gesture_kp"
    fi
fi

echo "=== Gesture Mimic Diffusion Policy Training ==="
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
echo "Horizon: ${HORIZON}, N_action_steps: ${N_ACTION_STEPS}, N_obs_steps: ${N_OBS_STEPS}"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA/ROCm: {torch.version.hip if hasattr(torch.version, \"hip\") else torch.version.cuda}'); print(f'GPUs: {torch.cuda.device_count()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')" 2>/dev/null || true

CMD=(
    lerobot-train
    --policy.type=diffusion
    --dataset.repo_id="${DATASET}"
    --dataset.image_transforms.enable=true

    # Diffusion architecture
    --policy.n_obs_steps="${N_OBS_STEPS}"
    --policy.horizon="${HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.vision_backbone="${VISION_BACKBONE}"
    --policy.down_dims="${DOWN_DIMS}"
    --policy.kernel_size="${KERNEL_SIZE}"
    --policy.n_groups="${N_GROUPS}"
    --policy.diffusion_step_embed_dim="${DIFFUSION_STEP_EMBED_DIM}"
    --policy.use_film_scale_modulation="${USE_FILM_SCALE_MODULATION}"
    --policy.spatial_softmax_num_keypoints="${SPATIAL_SOFTMAX_NUM_KEYPOINTS}"
    --policy.use_group_norm="${USE_GROUP_NORM}"

    # Noise scheduler
    --policy.noise_scheduler_type="${NOISE_SCHEDULER_TYPE}"
    --policy.num_train_timesteps="${NUM_TRAIN_TIMESTEPS}"
    --policy.beta_schedule="${BETA_SCHEDULE}"
    --policy.beta_start="${BETA_START}"
    --policy.beta_end="${BETA_END}"
    --policy.prediction_type="${PREDICTION_TYPE}"
    --policy.clip_sample="${CLIP_SAMPLE}"
    --policy.clip_sample_range="${CLIP_SAMPLE_RANGE}"

    # Inference (DDIM with fewer steps for speed)
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}"

    # Training
    --steps="${STEPS}"
    --batch_size="${BATCH_SIZE}"
    --num_workers="${NUM_WORKERS}"
    --seed="${SEED}"
    --use_policy_training_preset=true
    --output_dir="${OUTPUT_DIR}"

    # Optimizer
    --optimizer.type=adam
    --optimizer.lr="${LR}"
    --optimizer.weight_decay="${WEIGHT_DECAY}"
    --optimizer.grad_clip_norm="${GRAD_CLIP_NORM}"

    # Logging
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --wandb.enable="${WANDB}"

    # Disable push to hub by default
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
