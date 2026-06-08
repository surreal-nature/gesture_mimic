# Gesture Mimic — ACT & Diffusion Policy Training & Deployment

Train and deploy gesture mimic policies that watch a human via camera and drive an SO-101 robot arm to mimic their gestures in real-time. Supports two policy architectures:

- **ACT** (Action Chunking with Transformers) — predicts action chunks directly via a transformer + VAE
- **Diffusion Policy** — generates smooth future trajectories via iterative denoising, producing more natural robot motions

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start: RGB vs Keypoint Mode](#quick-start-rgb-vs-keypoint-mode)
- [ACT Policy](#act-policy)
- [Diffusion Policy](#diffusion-policy)
- [Keypoint Mode Details](#keypoint-mode-details)
- [Policy Architecture Comparison](#policy-architecture-comparison)
- [Training Tips](#training-tips)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
gesture_mimic/
├── train.sh                   # ACT policy training script
├── train_diffusion.sh         # Diffusion Policy training script
├── evaluate.py                # ACT offline evaluation
├── evaluate_diffusion.py      # Diffusion Policy offline evaluation
├── deploy_policy.py           # ACT real-time deployment
├── deploy_diffusion.py        # Diffusion Policy real-time deployment
├── pose_estimator.py          # MediaPipe pose estimation + keypoint normalization
├── yolo_preprocessor.py       # YOLO segmentation + arm skeleton overlay
├── merge_datasets.py          # Merge multiple datasets into one for training
├── preprocess_dataset.py      # Convert RGB dataset to MediaPipe keypoint dataset
├── preprocess_dataset_yolo.py # Preprocess RGB dataset with YOLO seg + pose overlay
├── requirements.txt           # Python dependencies
├── logs/                      # (created at runtime) SLURM/training logs
├── data/                      # (created at runtime) Preprocessed datasets
└── outputs/                   # (created at runtime) Checkpoints and eval results
    ├── train/act_gesture/              # ACT RGB mode
    ├── train/act_gesture_kp/           # ACT MediaPipe keypoint mode
    ├── train/act_gesture_yolo/         # ACT YOLO keypoint mode
    ├── train/diffusion_gesture/        # Diffusion RGB mode
    ├── train/diffusion_gesture_kp/     # Diffusion MediaPipe keypoint mode
    └── train/diffusion_gesture_yolo/   # Diffusion YOLO keypoint mode
```

No files inside the `lerobot/` repository are modified. Everything is standalone.

---

## Prerequisites

- **Python 3.12+**
- **Conda** (Miniconda or Anaconda)
- **GPU** recommended — AMD MI210/MI300X (ROCm) or NVIDIA A100/H100 (CUDA)
- **Webcam** for real-time deployment
- **SO-101 robot** (optional — deploy works in `--no-robot` mode)

---

## Installation

### 1. Install Conda (if not already installed)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts, then restart your shell or run:
source ~/miniconda3/etc/profile.d/conda.sh
```

### 2. Create and activate the conda environment

```bash
conda create -n act python=3.12 -y
conda activate act
```

### 3. Install PyTorch

```bash
# NVIDIA GPU (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# AMD GPU (ROCm 6.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
```

### 4. Install LeRobot

```bash
# Recommended: install from source for latest features
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[all]"
cd ..

# Or install from PyPI (may lag behind source)
pip install lerobot
```

### 5. Clone this repo and install dependencies

```bash
git clone https://github.com/surreal-nature/gesture_mimic.git
cd gesture_mimic
pip install -r requirements.txt
```

### 6. Verify installation

```bash
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
print(f'Device:   {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')
import lerobot
print(f'LeRobot:  {lerobot.__version__}')
"
lerobot-train --help | head -3
```

### AMD ROCm notes

On AMD GPUs (MI210, MI300X), set these environment variables before training for stable MIOpen kernel tuning:

```bash
export MIOPEN_FIND_ENFORCE=5
export MIOPEN_FIND_MODE=2
```

The `train.sh` script sets these automatically when `rocminfo` is detected, but exporting them in your shell ensures they apply to all commands (evaluate, deploy, etc.).

---

## Datasets

Multiple LeRobot v3.0 datasets are available on HuggingFace Hub (auto-downloaded on first use):

| Dataset | Episodes | Frames | Description |
|---------|----------|--------|-------------|
| [AmolSapale181284/multigesture-mimic](https://huggingface.co/datasets/AmolSapale181284/multigesture-mimic) | 50 | 19,329 | Original multi-gesture dataset |
| [BlankHead/extended_gesture_mimic](https://huggingface.co/datasets/BlankHead/extended_gesture_mimic) | — | — | Extended dataset with more demonstrations |
| [AMD-PAVS-AI/multigesture_mimic_test](https://huggingface.co/datasets/AMD-PAVS-AI/multigesture_mimic_test) | 10 | 3,868 | Test subset (shorter evaluation runs) |
| [AMD-PAVS-AI/Action-per-video-multigesture-mimic](https://huggingface.co/datasets/AMD-PAVS-AI/Action-per-video-multigesture-mimic) | 450 | 107,760 | Large dataset with one action per video (~8s episodes) |

### Dataset features

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `observation.state` | (6,) | float32 | Current joint angles |
| `observation.images.human` | (480, 640, 3) | video | Camera view of human demonstrator |
| `action` | (6,) | float32 | Target joint positions |

Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Robot type: `so_follower` (single SO-101 arm, 6-DOF)

### Merging datasets

Since `lerobot-train` accepts only a single dataset, use `merge_datasets.py` to combine multiple datasets before training:

```bash
# Merge the original + AMD-PAVS-AI datasets into one (~510 episodes, ~130K frames)
python merge_datasets.py \
    --sources AmolSapale181284/multigesture-mimic \
              AMD-PAVS-AI/multigesture_mimic_test \
              AMD-PAVS-AI/Action-per-video-multigesture-mimic \
    --target local/gesture_mimic_merged

# Train on the merged dataset
DATASET=local/gesture_mimic_merged \
    DATASET_ROOT=data/local_gesture_mimic_merged \
    bash train.sh

# Merge all known datasets (including extended)
python merge_datasets.py \
    --sources AmolSapale181284/multigesture-mimic \
              BlankHead/extended_gesture_mimic \
              AMD-PAVS-AI/multigesture_mimic_test \
              AMD-PAVS-AI/Action-per-video-multigesture-mimic \
    --target local/gesture_mimic_all

# Push merged dataset to HuggingFace Hub
python merge_datasets.py \
    --sources AmolSapale181284/multigesture-mimic \
              AMD-PAVS-AI/Action-per-video-multigesture-mimic \
    --target myuser/gesture_mimic_combined \
    --push-to-hub
```

The merged dataset can also be preprocessed for keypoint modes:

```bash
# MediaPipe keypoint preprocessing on merged dataset
python preprocess_dataset.py \
    --source local/gesture_mimic_merged \
    --source-root data/local_gesture_mimic_merged \
    --target local/merged_keypoints

# YOLO keypoint preprocessing on merged dataset
python preprocess_dataset_yolo.py \
    --source local/gesture_mimic_merged \
    --source-root data/local_gesture_mimic_merged \
    --target local/merged_yolo
```

---

## Quick Start: RGB vs Keypoint Mode

This pipeline supports three input modes:

| Mode | Train Command | Deploy Flag | What ACT Sees |
|------|--------------|-------------|---------------|
| **RGB** (default) | `bash train.sh` | (none) | Raw camera frames via ResNet-18 |
| **MediaPipe keypoint** | `USE_KEYPOINTS=true bash train.sh` | `--use-keypoints` | Normalized pose skeleton (no vision backbone) |
| **YOLO keypoint** | `USE_KEYPOINTS=true POSE_BACKEND=yolo bash train.sh` | `--use-keypoints --pose-backend yolo` | Segmented RGB + arm skeleton overlay via ResNet-18 |

**RGB mode** feeds raw webcam frames through a ResNet-18 vision backbone. Good for tasks requiring visual detail (grasping, object interaction).

**MediaPipe keypoint mode** extracts pose landmarks, then trains ACT on normalized skeleton + velocity features only (no images). Best for small datasets and gross arm gestures.

**YOLO keypoint mode** removes the background (YOLOv26n-seg), overlays 6 arm keypoints with an inverted-U skeleton (YOLOv26n-pose), and feeds the modified image through the ResNet-18 vision backbone. Combines the benefits of both: visual input with explicit skeletal structure, reduced background noise.

See [Keypoint Mode Details](#keypoint-mode-details) for the full preprocessing and configuration reference.

---

## Step 1: Train

All commands below assume you are inside the `gesture_mimic/` directory with the `act` conda environment activated:

```bash
cd gesture_mimic
conda activate act
```

### RGB mode (default)

```bash
# Train on the multigesture-mimic dataset
DATASET=AmolSapale181284/multigesture-mimic bash train.sh

# Train on the extended dataset (used if DATASET is not set)
bash train.sh

# Train on AMD-PAVS-AI datasets
DATASET=AMD-PAVS-AI/Action-per-video-multigesture-mimic bash train.sh
DATASET=AMD-PAVS-AI/multigesture_mimic_test bash train.sh

# Train on a merged dataset (see "Merging datasets" section above)
DATASET=local/gesture_mimic_merged DATASET_ROOT=data/local_gesture_mimic_merged bash train.sh

# AMD ROCm GPU — set MIOpen flags for stable kernel tuning
export MIOPEN_FIND_ENFORCE=5 MIOPEN_FIND_MODE=2
DATASET=AmolSapale181284/multigesture-mimic bash train.sh

# SLURM submission (edit #SBATCH lines in train.sh for your cluster)
sbatch train.sh
```

### MediaPipe keypoint mode

```bash
# Step A: Preprocess dataset — extract pose keypoints (run once)
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_keypoints

# Step B: Train on the keypoint dataset
USE_KEYPOINTS=true bash train.sh
```

### YOLO keypoint mode

```bash
# Step A: Preprocess dataset — segment person + overlay arm skeleton (run once)
python preprocess_dataset_yolo.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_yolo

# Step B: Train on the YOLO-preprocessed dataset
USE_KEYPOINTS=true POSE_BACKEND=yolo bash train.sh
```

### Common overrides

```bash
# Override hyperparameters
STEPS=30000 BATCH_SIZE=16 LR=5e-5 bash train.sh

# Fine-tune from a previous checkpoint
PRETRAINED=outputs/train/act_gesture/checkpoints/last/pretrained_model \
    STEPS=10000 LR=1e-5 bash train.sh

# Push trained model to HuggingFace Hub
PUSH_TO_HUB=true HUB_REPO=myuser/act_gesture_v5 bash train.sh
```

### Where checkpoints are saved

| Mode | Checkpoint directory |
|------|---------------------|
| RGB | `outputs/train/act_gesture/checkpoints/` |
| MediaPipe keypoint | `outputs/train/act_gesture_kp/checkpoints/` |
| YOLO keypoint | `outputs/train/act_gesture_yolo/checkpoints/` |

Checkpoints are saved at every `SAVE_FREQ` steps (default: 5000). A `last/` checkpoint is always saved at the end of training. Each checkpoint contains a `pretrained_model/` directory that can be loaded directly.

### Training configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET` | `BlankHead/extended_gesture_mimic` | HuggingFace dataset repo_id (or local merged dataset) |
| `DATASET_ROOT` | (auto) | Local path for cached/local datasets |
| `STEPS` | 20000 | Total training steps |
| `BATCH_SIZE` | 8 | Per-GPU batch size |
| `CHUNK_SIZE` | 100 | Action chunk length (~3.3s at 30fps) |
| `DIM_MODEL` | 512 | Transformer hidden dimension |
| `LR` | 1e-5 | Peak learning rate |
| `KL_WEIGHT` | 10.0 | VAE KL divergence weight |
| `PRETRAINED` | (none) | Checkpoint path for fine-tuning |
| `USE_KEYPOINTS` | `false` | Set to `true` to train on preprocessed keypoint dataset |
| `POSE_BACKEND` | `mediapipe` | Keypoint backend: `mediapipe` (state-only) or `yolo` (segmented RGB) |
| `PUSH_TO_HUB` | `false` | Push model to HuggingFace Hub after training |
| `HUB_REPO` | (none) | HuggingFace repo ID (required if `PUSH_TO_HUB=true`) |
| `WANDB` | `false` | Enable Weights & Biases logging |
| `SAVE_FREQ` | 5000 | Save checkpoint every N steps |
| `SEED` | 1000 | Random seed |

All parameters can be overridden via environment variables. Extra CLI arguments are passed through to `lerobot-train`.

---

## Step 2: Evaluate

### RGB mode

```bash
# Evaluate all checkpoints in a training run
python evaluate.py --training-dir outputs/train/act_gesture

# Evaluate a single checkpoint
python evaluate.py --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model

# Evaluate a HuggingFace Hub model
python evaluate.py --checkpoint AmolSapale181284/act_gesture_mimic_v4

# Custom dataset and validation split
python evaluate.py --training-dir outputs/train/act_gesture \
    --dataset AmolSapale181284/multigesture-mimic \
    --val-episodes 45 46 47 48 49
```

### Keypoint mode

```bash
# Point to the keypoint training output and the preprocessed dataset
python evaluate.py --training-dir outputs/train/act_gesture_kp \
    --dataset local/gesture_mimic_keypoints \
    --dataset-root data/local_gesture_mimic_keypoints
```

### Evaluation outputs

Saved to `<training-dir>/eval/`:

| File | Description |
|------|-------------|
| `eval_summary.png` | Loss curves + per-joint error bars + predicted vs actual |
| `eval_actions.png` | Per-joint predicted vs actual trajectories (all 6 joints) |
| `eval_metrics.json` | All metrics in machine-readable format |

---

## Step 3: Deploy

### RGB mode

```bash
# Real-time: webcam + SO-101 robot
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model \
    --port /dev/ttyUSB0

# Webcam-only mode (no robot hardware needed)
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model \
    --no-robot

# Process a pre-recorded video
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model \
    --video gesture_demo.mp4 --no-robot
```

### MediaPipe keypoint mode (add `--use-keypoints`)

```bash
# Real-time with webcam (runs MediaPipe online, draws skeleton overlay)
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --no-robot

# With specific landmark preset
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --landmarks right_arm --no-robot

# Process video file with skeleton overlay
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --video gesture_demo.mp4 --no-robot
```

### YOLO keypoint mode (add `--use-keypoints --pose-backend yolo`)

```bash
# Real-time with webcam (runs YOLO seg + pose online, shows segmented + skeleton)
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_yolo/checkpoints/last/pretrained_model \
    --use-keypoints --pose-backend yolo --no-robot

# Process video file
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_yolo/checkpoints/last/pretrained_model \
    --use-keypoints --pose-backend yolo --video gesture_demo.mp4 --no-robot
```

### Deploy controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset policy state (clear action chunk buffer) |

The deployment window shows a live overlay with current joint states and predicted actions for each joint. In keypoint mode, the detected skeleton is also drawn on the video feed.

### Deploy CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | (required) | Path to pretrained model or HuggingFace model ID |
| `--no-robot` | false | Run without robot hardware (visualization only) |
| `--port` | `/dev/ttyUSB0` | Robot serial port |
| `--camera-id` | 0 | Webcam device ID |
| `--video` | (none) | Path to input video file (instead of live webcam) |
| `--output` | (auto) | Path to save annotated output video |
| `--fps` | 30 | Target FPS for real-time loop |
| `--use-keypoints` | false | Enable keypoint mode (online pose estimation) |
| `--pose-backend` | `mediapipe` | Pose backend: `mediapipe` (state-only) or `yolo` (segmented RGB) |
| `--landmarks` | `upper_body` | Landmark preset for MediaPipe: `upper_body`, `right_arm`, `left_arm` |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

---

## Diffusion Policy

Diffusion Policy generates smooth future trajectories via iterative denoising instead of predicting action chunks directly. This produces more natural, anticipatory robot motions — especially important for continuous gesture tracking where ACT can produce jerky transitions at chunk boundaries.

**The same datasets used for ACT work for Diffusion Policy — no data recollection needed.**

### How it works

```
Training:
  Clean action trajectory → Add noise (100 steps) → Train UNet to predict noise

Inference:
  Random noise → Denoise (10 steps) → Smooth robot trajectory
```

The policy conditions on `n_obs_steps` recent observations (default: 2) to predict a `horizon`-length trajectory (default: 32 steps = ~1s at 30fps), of which `n_action_steps` (default: 16 steps = ~0.5s) are executed before re-planning.

### Installation

Diffusion Policy requires the `diffusers` package:

```bash
conda activate act
pip install diffusers
```

### Train

```bash
cd gesture_mimic
conda activate act

# RGB mode (default)
bash train_diffusion.sh

# With specific dataset
DATASET=AmolSapale181284/multigesture-mimic bash train_diffusion.sh

# MediaPipe keypoint mode (preprocess first)
python preprocess_dataset.py --source BlankHead/extended_gesture_mimic --target local/gesture_kp
USE_KEYPOINTS=true bash train_diffusion.sh

# YOLO keypoint mode (preprocess first)
python preprocess_dataset_yolo.py --source AmolSapale181284/multigesture-mimic --target local/gesture_mimic_yolo
USE_KEYPOINTS=true POSE_BACKEND=yolo bash train_diffusion.sh

# Override hyperparameters
STEPS=30000 BATCH_SIZE=16 HORIZON=64 N_ACTION_STEPS=32 bash train_diffusion.sh

# Fine-tune from a previous checkpoint
PRETRAINED=outputs/train/diffusion_gesture/checkpoints/last/pretrained_model bash train_diffusion.sh
```

### Diffusion training configuration reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N_OBS_STEPS` | 2 | Number of past observations for temporal context |
| `HORIZON` | 32 | Trajectory prediction length (~1s at 30fps) |
| `N_ACTION_STEPS` | 16 | Actions to execute before re-planning (~0.5s) |
| `DOWN_DIMS` | `[256,512,1024]` | UNet feature dimensions (3 stages) |
| `NOISE_SCHEDULER_TYPE` | `DDPM` | Noise scheduler for training |
| `NUM_TRAIN_TIMESTEPS` | 100 | Forward diffusion steps |
| `NUM_INFERENCE_STEPS` | 10 | Reverse denoising steps (DDIM, faster) |
| `PREDICTION_TYPE` | `epsilon` | Predict noise (`epsilon`) or sample (`sample`) |
| `LR` | 1e-4 | Learning rate |
| `STEPS` | 20000 | Total training steps |
| `BATCH_SIZE` | 8 | Per-GPU batch size |

All ACT environment variables (`DATASET`, `DATASET_ROOT`, `USE_KEYPOINTS`, `POSE_BACKEND`, `WANDB`, etc.) also apply.

### Evaluate

```bash
# Evaluate all checkpoints
python evaluate_diffusion.py --training-dir outputs/train/diffusion_gesture

# Single checkpoint
python evaluate_diffusion.py --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model

# Keypoint mode
python evaluate_diffusion.py --training-dir outputs/train/diffusion_gesture_kp \
    --dataset local/gesture_mimic_keypoints \
    --dataset-root data/local_gesture_mimic_keypoints
```

### Deploy

```bash
# RGB mode with robot
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model \
    --port /dev/ttyUSB0

# No-robot mode (visualization only)
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model \
    --no-robot

# MediaPipe keypoint mode
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --no-robot

# YOLO keypoint mode
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture_yolo/checkpoints/last/pretrained_model \
    --use-keypoints --pose-backend yolo --no-robot

# Process video file
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model \
    --video gesture_demo.mp4 --no-robot
```

The deploy overlay shows denoising inference time in addition to FPS and joint angles.

---

## Keypoint Mode Details

Instead of feeding raw RGB frames to ACT (which wastes capacity learning background, lighting, and clothing), keypoint mode extracts human pose information. Two backends are available:

### Pipeline comparison

```
RGB mode (default):
  webcam frame -> ResNet-18 -> ACT transformer -> robot action

MediaPipe keypoint mode (--use-keypoints):
  webcam frame -> MediaPipe Pose -> normalized keypoints -> ACT (state-only) -> robot action

YOLO keypoint mode (--use-keypoints --pose-backend yolo):
  webcam frame -> YOLOv26n-seg (remove background) -> YOLOv26n-pose (overlay 6 arm keypoints)
    -> modified RGB -> ResNet-18 -> ACT transformer -> robot action
```

### Preprocessing the dataset

Convert an existing RGB dataset to a keypoint dataset (run once):

```bash
# Upper body keypoints (11 landmarks, default)
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_keypoints

# Right arm only (7 landmarks, more focused)
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_kp_rarm \
    --landmarks right_arm

# Keep images alongside keypoints (hybrid dataset)
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_hybrid \
    --keep-images

# Disable velocity/acceleration features (position only)
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_kp_pos_only \
    --no-velocity --no-acceleration

# Push preprocessed dataset to HuggingFace Hub
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target myuser/gesture_mimic_keypoints \
    --push-to-hub
```

### What the preprocessor does

1. Extracts upper-body landmarks from each video frame using MediaPipe Pose
2. Normalizes keypoints relative to shoulder center, scaled by torso length (camera/scale invariant)
3. Computes velocity and acceleration for temporal dynamics
4. Applies exponential moving average smoothing to reduce MediaPipe jitter
5. Concatenates `[robot_state (6), keypoint_features (K)]` into extended `observation.state`
6. Creates a new LeRobot dataset without image features
7. Saves `meta/keypoint_config.json` for reproducibility

### Keypoint feature dimensions

| Landmark Preset | Landmarks | Positions | + Velocity | + Acceleration | Total Features |
|----------------|-----------|-----------|------------|----------------|----------------|
| `upper_body`   | 11        | 33        | 33         | 33             | 99             |
| `right_arm`    | 7         | 21        | 21         | 21             | 63             |
| `left_arm`     | 7         | 21        | 21         | 21             | 63             |

The new `observation.state` dimension = 6 (robot joints) + total keypoint features.

### Preprocessor CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | (required) | Source dataset repo_id |
| `--target` | (required) | Target dataset repo_id |
| `--target-root` | (auto) | Local path for target dataset |
| `--source-root` | (auto) | Local path for source dataset |
| `--landmarks` | `upper_body` | Preset: `upper_body`, `right_arm`, `left_arm` |
| `--no-velocity` | false | Disable velocity features |
| `--no-acceleration` | false | Disable acceleration features |
| `--smoothing` | 0.3 | EMA smoothing alpha (0 = off) |
| `--use-2d` | false | Use 2D coordinates instead of 3D |
| `--keep-images` | false | Keep image features alongside keypoints |
| `--push-to-hub` | false | Push target dataset to HuggingFace Hub |

### YOLO keypoint mode details

Unlike MediaPipe mode (which removes images entirely), YOLO mode keeps images but preprocesses them:

1. **Background removal**: YOLOv26n-seg segments the person and blacks out everything else
2. **Arm keypoint detection**: YOLOv26n-pose detects 17 COCO keypoints, but only 6 arm keypoints are used:

| COCO Index | Joint | Overlay Color |
|------------|-------|---------------|
| 5 | left_shoulder | Green |
| 6 | right_shoulder | Dark green |
| 7 | left_elbow | Blue |
| 8 | right_elbow | Light blue |
| 9 | left_wrist | Red |
| 10 | right_wrist | Dark red |

3. **Skeleton overlay**: Keypoints are connected in order 9→7→5→6→8→10 (left wrist → left elbow → left shoulder → right shoulder → right elbow → right wrist), forming an inverted-U shape
4. **Same observation space**: The modified image has the same shape as the original — the ACT policy, training config, and vision backbone are identical to RGB mode

#### YOLO preprocessing

```bash
python preprocess_dataset_yolo.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_yolo

# On CPU (slower but no GPU needed)
python preprocess_dataset_yolo.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_mimic_yolo \
    --device cpu
```

#### YOLO preprocessor CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | (required) | Source dataset repo_id |
| `--target` | (required) | Target dataset repo_id |
| `--target-root` | (auto) | Local path for target dataset |
| `--source-root` | (auto) | Local path for source dataset |
| `--seg-model` | `yolo26n-seg.pt` | YOLOv26n segmentation model |
| `--pose-model` | `yolo26n-pose.pt` | YOLOv26n pose model |
| `--device` | `cuda` | Device for YOLO inference |
| `--confidence` | 0.5 | Minimum keypoint confidence to overlay |
| `--push-to-hub` | false | Push preprocessed dataset to HuggingFace Hub |

### Why keypoints work better for gesture mimic

| Factor | RGB Mode | MediaPipe Keypoint | YOLO Keypoint |
|--------|----------|--------------------|---------------|
| What ACT learns | Perception + control | Control only | Guided perception + control |
| Data efficiency | Poor (50 episodes) | Excellent | Good |
| Training speed | Slow (ResNet-18) | Fast (state-only) | Slow (ResNet-18) |
| Camera invariance | None | Built-in (normalization) | Partial (background removed) |
| Lighting robustness | Poor | Immune | Good (background removed) |
| Background sensitivity | High | None | None (segmented) |
| Fine manipulation | Good | Limited | Good (image preserved) |
| Explicit skeleton info | None | Full (state features) | Visual (overlay on image) |

**Recommendations**:
- **MediaPipe keypoint**: Best for small datasets and gross arm gestures (waving, pointing, reaching)
- **YOLO keypoint**: Best balance — removes background noise while keeping visual detail + explicit skeleton structure
- **RGB**: Use if the full scene context matters (object interaction, environment-dependent tasks)

---

## Policy Architecture Comparison

### ACT (Action Chunking with Transformers)

```
Observation (single frame)
    ↓
ResNet-18 (images) + Linear (state)
    ↓
Transformer Encoder-Decoder + VAE
    ↓
Action Chunk: (100, 6) — 100 future joint positions
    ↓
Execute all 100 sequentially (~3.3s), then predict new chunk
```

- **Pros**: Simple, fast inference (single forward pass)
- **Cons**: Jerky at chunk boundaries, no temporal observation context, reacts late to direction changes

### Diffusion Policy

```
Observations (2 recent frames)
    ↓
ResNet-18 + SpatialSoftmax (images) + Linear (state)
    ↓
Conditioning Vector
    ↓
1D Conditional UNet (iterative denoising: noise → trajectory)
    ↓
Trajectory: (32, 6) — 32-step smooth trajectory (~1s)
    ↓
Execute 16 steps (~0.5s), then re-plan with overlap
```

- **Pros**: Smooth trajectories, temporal context from observation history, better multimodal behavior, natural motion anticipation
- **Cons**: Slower per-chunk inference (10 denoising steps), requires `diffusers` package

### Why Diffusion Policy works better for gesture mimic

| Factor | ACT | Diffusion Policy |
|--------|-----|-----------------|
| Trajectory smoothness | Chunk boundaries can be jerky | Naturally smooth (denoising) |
| Temporal context | Single observation | `n_obs_steps` recent observations |
| Motion anticipation | None (reactive) | Learns temporal correlations |
| Re-planning frequency | Every ~3.3s (100 steps) | Every ~0.5s (16 steps) |
| Direction change response | Delayed | Faster (shorter action horizon + overlap) |
| Inference speed | ~5ms (single pass) | ~50ms (10 denoising steps) |
| Training compute | Lower | Higher |

For gesture mimic, the Diffusion Policy's ability to generate smooth future trajectories conditioned on recent motion history makes it fundamentally better suited than ACT, which must constantly predict fresh action chunks and stitch them together.

---

## Training Tips

- **Overfitting** typically appears after 6K-10K steps on small datasets. Use checkpoint evaluation to find the best trade-off.
- **l1_loss** should steadily decrease. Plateau early = try increasing `DIM_MODEL` or `CHUNK_SIZE`.
- **kld_loss** should stabilize around 0.01-0.05. Collapse to 0 = VAE not learning useful latent structure.
- **50+ episodes** recommended for robust generalization.
- **Image augmentation** is enabled by default (`--dataset.image_transforms.enable=true`) in RGB mode.
- **Keypoint augmentation**: scale noise and temporal jitter can be applied during preprocessing for better generalization.
- **First ~50 steps on AMD GPUs** are slow due to MIOpen kernel auto-tuning. This is normal and only happens once per kernel configuration. Subsequent runs use cached kernels.

---

## Troubleshooting

### `torchcodec ABI mismatch` warning

```
WARNING: 'torchcodec' is installed but failed to load (ABI mismatch?), falling back to 'pyav'
```

Harmless. LeRobot falls back to PyAV for video decoding, which works correctly. Suppress by uninstalling torchcodec: `pip uninstall torchcodec`.

### `ValueError: 'repo_id' argument missing`

This happens if `push_to_hub` is enabled without a repo ID. The `train.sh` script disables push by default (`--policy.push_to_hub=false`). If you want to push, set both:

```bash
PUSH_TO_HUB=true HUB_REPO=myuser/model_name bash train.sh
```

### `FileNotFoundError: tasks.parquet`

The dataset cache is incomplete. Delete the cached snapshot and re-download:

```bash
rm -rf ~/.cache/huggingface/lerobot/hub/datasets--<org>--<dataset_name>
# The dataset will re-download on next training run
```

Or use `--dataset-root` to point to a fully downloaded local copy.

### Training loss not decreasing

- **Check dataset size.** At least 30-50 episodes recommended.
- **Reduce learning rate.** Try `LR=5e-6`.
- **Increase chunk_size.** Longer action chunks help temporal coherence.
- **Try keypoint mode.** Small datasets often perform much better without the vision backbone.

### MIOpen kernel tuning makes first steps very slow

Normal on AMD GPUs. The first ~50 steps take 5-10 seconds each while MIOpen auto-tunes convolution kernels. After that, step time drops to ~1-2s. Kernels are cached for subsequent runs. Set `MIOPEN_FIND_ENFORCE=5 MIOPEN_FIND_MODE=2` for deterministic tuning.

### `Cannot open camera 0`

The deploy script requires a webcam. If running on a headless server, use `--video` to process a pre-recorded video file instead:

```bash
python deploy_policy.py --checkpoint <path> --video input.mp4 --no-robot
```

### OpenGL / display errors on headless servers

Evaluation runs headlessly (uses `matplotlib.use("Agg")`). Deployment with `--video` also works headlessly. Only live webcam mode requires a display.
