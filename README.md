# Gesture Mimic — ACT Policy Training & Deployment

Train and deploy an ACT (Action Chunking with Transformers) policy that watches a human via camera and drives an SO-101 robot arm to mimic their gestures in real-time.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick Start: RGB vs Keypoint Mode](#quick-start-rgb-vs-keypoint-mode)
- [Step 1: Train](#step-1-train)
- [Step 2: Evaluate](#step-2-evaluate)
- [Step 3: Deploy](#step-3-deploy)
- [Keypoint Mode Details](#keypoint-mode-details)
- [ACT Policy Architecture](#act-policy-architecture)
- [Training Tips](#training-tips)
- [Troubleshooting](#troubleshooting)

---

## Project Structure

```
gesture_mimic/
├── train.sh                  # SLURM-compatible training script
├── evaluate.py               # Offline evaluation (loss, per-joint error, plots)
├── deploy_policy.py          # Real-time deployment (webcam + robot or video replay)
├── pose_estimator.py         # MediaPipe pose estimation + keypoint normalization
├── yolo_preprocessor.py      # YOLO segmentation + arm skeleton overlay
├── preprocess_dataset.py     # Convert RGB dataset to MediaPipe keypoint dataset
├── preprocess_dataset_yolo.py # Preprocess RGB dataset with YOLO seg + pose overlay
├── requirements.txt          # Python dependencies
├── logs/                     # (created at runtime) SLURM/training logs
├── data/                     # (created at runtime) Preprocessed datasets
└── outputs/                  # (created at runtime) Checkpoints and eval results
    ├── train/act_gesture/         # RGB mode outputs
    ├── train/act_gesture_kp/      # MediaPipe keypoint mode outputs
    └── train/act_gesture_yolo/    # YOLO keypoint mode outputs
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

Two LeRobot v3.0 datasets are available on HuggingFace Hub (auto-downloaded on first use):

| Dataset | Episodes | Frames | Description |
|---------|----------|--------|-------------|
| [AmolSapale181284/multigesture-mimic](https://huggingface.co/datasets/AmolSapale181284/multigesture-mimic) | 50 | 19,329 | Original multi-gesture dataset |
| [BlankHead/extended_gesture_mimic](https://huggingface.co/datasets/BlankHead/extended_gesture_mimic) | — | — | Extended dataset with more demonstrations |

### Dataset features

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `observation.state` | (6,) | float32 | Current joint angles |
| `observation.images.human` | (480, 640, 3) | video | Camera view of human demonstrator |
| `action` | (6,) | float32 | Target joint positions |

Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Robot type: `so_follower` (single SO-101 arm, 6-DOF)

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
| `DATASET` | `BlankHead/extended_gesture_mimic` | HuggingFace dataset repo_id |
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

## ACT Policy Architecture

### RGB mode

```
Input:
  Human camera:  (3, 480, 640) -> ResNet-18 -> (512,) features
  Robot state:   (6,) -> Linear(6, 512) -> (512,) features
  Combined:      Image + state tokens -> Transformer encoder

Architecture:
  Vision backbone:  ResNet-18 (pretrained on ImageNet)
  Transformer:      dim=512, heads=8, enc_layers=4, dec_layers=1
  VAE:              latent_dim=32, kl_weight=10.0
  Feedforward:      3200

Output:
  Action chunk:     (100, 6) — 100 future joint positions
  At inference:     execute all 100 actions sequentially (~3.3s),
                    then predict a new chunk
```

### MediaPipe keypoint mode

```
Input:
  Extended state:  (6 + K,) -> Linear(6+K, 512) -> (512,) features
    where:
      robot state:  (6,) joint angles
      keypoints:    (K,) normalized positions + velocities + accelerations

Architecture:
  Vision backbone:  NONE (no images)
  Transformer:      dim=512, heads=8, enc_layers=4, dec_layers=1
  VAE:              latent_dim=32, kl_weight=10.0
  Feedforward:      3200

Output:
  Action chunk:     (100, 6) — 100 future joint positions
```

### YOLO keypoint mode

```
Input:
  Modified camera:  (3, 480, 640) -> ResNet-18 -> (512,) features
    where image = segmented person + 6 colored arm keypoints + skeleton lines
  Robot state:      (6,) -> Linear(6, 512) -> (512,) features

Architecture:
  Same as RGB mode — vision backbone processes the YOLO-preprocessed image
  Vision backbone:  ResNet-18 (pretrained on ImageNet)
  Transformer:      dim=512, heads=8, enc_layers=4, dec_layers=1
  VAE:              latent_dim=32, kl_weight=10.0
  Feedforward:      3200

Output:
  Action chunk:     (100, 6) — 100 future joint positions
```

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
