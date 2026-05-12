# Gesture Mimic — ACT Policy Training & Deployment

Train and deploy an ACT (Action Chunking with Transformers) policy that watches a human via camera and drives an SO-101 robot arm to mimic their gestures in real-time.

---

## Project Structure

```
gesture_mimic/
├── train.sh              # SLURM-compatible training script
├── evaluate.py           # Offline evaluation (loss, per-joint error, plots)
├── deploy_policy.py      # Real-time deployment (webcam + robot or video replay)
├── pose_estimator.py     # MediaPipe pose estimation + keypoint normalization
├── preprocess_dataset.py # Convert RGB dataset to keypoint dataset
├── requirements.txt      # Python dependencies
├── logs/                 # (created at runtime) SLURM/training logs
├── data/                 # (created at runtime) Preprocessed keypoint datasets
└── outputs/              # (created at runtime) Checkpoints and eval results
    ├── train/act_gesture/      # RGB mode outputs
    └── train/act_gesture_kp/   # Keypoint mode outputs
```

No files inside the `lerobot/` repository are modified. Everything is standalone.

---

## Datasets

Two LeRobot v3.0 datasets are available on HuggingFace Hub:

| Dataset | Episodes | Frames | Description |
|---------|----------|--------|-------------|
| [AmolSapale181284/multigesture-mimic](https://huggingface.co/datasets/AmolSapale181284/multigesture-mimic) | 50 | 19,329 | Original multi-gesture dataset |
| [BlankHead/extended_gesture_mimic](https://huggingface.co/datasets/BlankHead/extended_gesture_mimic) | — | — | Extended dataset with more demonstrations |

### Dataset Features

| Feature | Shape | Type | Description |
|---------|-------|------|-------------|
| `observation.state` | (6,) | float32 | Current joint angles |
| `observation.images.human` | (480, 640, 3) | video | Camera view of human demonstrator |
| `action` | (6,) | float32 | Target joint positions |

Joint order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`

Robot type: `so_follower` (single SO-101 arm, 6-DOF)

---

## Usage

### Step 1: Train

```bash
# Default: train on extended dataset
bash train.sh

# SLURM submission
sbatch train.sh

# Use original dataset
DATASET=AmolSapale181284/multigesture-mimic bash train.sh

# Override hyperparameters
STEPS=30000 BATCH_SIZE=16 LR=5e-5 bash train.sh

# Fine-tune from a previous checkpoint
PRETRAINED=outputs/train/act_gesture/checkpoints/last/pretrained_model \
    STEPS=10000 LR=1e-5 bash train.sh

# Push trained model to HuggingFace Hub
PUSH_TO_HUB=true HUB_REPO=myuser/act_gesture_v5 bash train.sh
```

#### Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATASET` | `BlankHead/extended_gesture_mimic` | HuggingFace dataset repo_id |
| `STEPS` | 20000 | Total training steps |
| `BATCH_SIZE` | 8 | Per-GPU batch size |
| `CHUNK_SIZE` | 100 | Action chunk length (~3.3s at 30fps) |
| `DIM_MODEL` | 512 | Transformer hidden dimension |
| `LR` | 1e-5 | Peak learning rate |
| `KL_WEIGHT` | 10.0 | VAE KL divergence weight |
| `PRETRAINED` | (none) | Checkpoint path for fine-tuning |

All parameters can be overridden via environment variables. Extra CLI arguments are passed through to `lerobot-train`.

### Step 2: Evaluate

```bash
# Evaluate all checkpoints in a training run
python evaluate.py --training-dir outputs/train/act_gesture

# Evaluate a single checkpoint
python evaluate.py --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model

# Evaluate a HuggingFace Hub model
python evaluate.py --checkpoint AmolSapale181284/act_gesture_mimic_v4

# Custom validation split
python evaluate.py --training-dir outputs/train/act_gesture \
    --dataset AmolSapale181284/multigesture-mimic \
    --val-episodes 45 46 47 48 49
```

Outputs saved to `<training-dir>/eval/`:
- `eval_summary.png` — loss curves + per-joint error bars + predicted vs actual
- `eval_actions.png` — per-joint predicted vs actual trajectories
- `eval_metrics.json` — all metrics in machine-readable format

### Step 3: Deploy

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

# HuggingFace Hub model
python deploy_policy.py \
    --checkpoint AmolSapale181284/act_gesture_mimic_v4 \
    --no-robot
```

#### Deploy Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset policy state (clear action chunk buffer) |

The deployment window shows a live overlay with current joint states and predicted actions for each joint.

---

## Keypoint Mode (Pose Estimation)

Instead of feeding raw RGB frames to ACT (which wastes capacity learning background, lighting, and clothing), keypoint mode extracts human pose landmarks and feeds normalized skeleton data directly. This is dramatically more data-efficient for small datasets.

### Pipeline Comparison

```
RGB mode (default):
  webcam frame -> ResNet-18 -> ACT transformer -> robot action

Keypoint mode (--use-keypoints):
  webcam frame -> MediaPipe Pose -> normalized keypoints -> ACT (state-only) -> robot action
```

### Step 1: Preprocess the Dataset

Convert an existing RGB dataset to a keypoint dataset:

```bash
# Upper body keypoints (11 landmarks, default)
python preprocess_dataset.py \
    --source BlankHead/extended_gesture_mimic \
    --target local/gesture_mimic_keypoints

# Right arm only (7 landmarks, more focused)
python preprocess_dataset.py \
    --source BlankHead/extended_gesture_mimic \
    --target local/gesture_mimic_kp_rarm \
    --landmarks right_arm

# Keep images alongside keypoints (hybrid dataset)
python preprocess_dataset.py \
    --source BlankHead/extended_gesture_mimic \
    --target local/gesture_mimic_hybrid \
    --keep-images

# Disable velocity/acceleration features
python preprocess_dataset.py \
    --source AmolSapale181284/multigesture-mimic \
    --target local/gesture_kp_pos_only \
    --no-velocity --no-acceleration

# Push preprocessed dataset to HuggingFace Hub
python preprocess_dataset.py \
    --source BlankHead/extended_gesture_mimic \
    --target myuser/gesture_mimic_keypoints \
    --push-to-hub
```

#### What the Preprocessor Does

1. Extracts upper-body landmarks from each video frame using MediaPipe Pose
2. Normalizes keypoints relative to shoulder center, scaled by torso length (camera/scale invariant)
3. Computes velocity and acceleration for temporal dynamics
4. Applies exponential moving average smoothing to reduce jitter
5. Concatenates `[robot_state (6), keypoint_features (K)]` into extended `observation.state`
6. Creates a new LeRobot dataset without image features

#### Keypoint Features

| Landmark Preset | Landmarks | Positions | + Velocity | + Acceleration | Total Features |
|----------------|-----------|-----------|------------|----------------|----------------|
| `upper_body`   | 11        | 33        | 33         | 33             | 99             |
| `right_arm`    | 7         | 21        | 21         | 21             | 63             |
| `left_arm`     | 7         | 21        | 21         | 21             | 63             |

The new `observation.state` dimension = 6 (robot joints) + total keypoint features.

### Step 2: Train on Keypoints

```bash
# Using the USE_KEYPOINTS flag (auto-configures dataset paths)
USE_KEYPOINTS=true bash train.sh

# Or point directly to the preprocessed dataset
DATASET=local/gesture_mimic_keypoints \
    DATASET_ROOT=data/local_gesture_mimic_keypoints \
    OUTPUT_DIR=outputs/train/act_gesture_kp \
    bash train.sh
```

Since there are no images, ACT runs without the vision backbone — training is much faster.

### Step 3: Deploy with Keypoints

```bash
# Real-time with webcam (runs MediaPipe online)
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --no-robot

# With specific landmark preset
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --landmarks right_arm --no-robot

# Process video file with keypoint overlay
python deploy_policy.py \
    --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
    --use-keypoints --video gesture_demo.mp4 --no-robot
```

In keypoint deploy mode, the skeleton overlay is drawn on the video feed for visual feedback.

### Why Keypoints Work Better for Gesture Mimic

| Factor | RGB Mode | Keypoint Mode |
|--------|----------|---------------|
| What ACT learns | Perception + control | Control only |
| Data efficiency | Poor (50 episodes) | Excellent |
| Training speed | Slow (ResNet-18 backbone) | Fast (state-only) |
| Camera invariance | None | Built-in (normalization) |
| Lighting robustness | Poor | Immune |
| Background sensitivity | High | None |
| Fine manipulation | Good | Limited |

**Recommendation**: Use keypoint mode for gross arm gestures (waving, pointing, reaching). Use RGB mode if finger-level detail or object interaction matters.

---

## ACT Policy Architecture

### RGB Mode

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

### Keypoint Mode

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

---

## Training Tips

- **Overfitting** typically appears after 6K-10K steps on small datasets. Use checkpoint evaluation to find the best trade-off.
- **l1_loss** should steadily decrease. Plateau early = try increasing `DIM_MODEL` or `CHUNK_SIZE`.
- **kld_loss** should stabilize around 0.01-0.05. Collapse to 0 = VAE not learning latent structure.
- **50+ episodes** recommended for robust generalization.
- **Image augmentation** is enabled by default (`--dataset.image_transforms.enable=true`).

---

## Prerequisites

- **Python 3.12+**
- **Conda environment** with LeRobot installed (`conda activate act`)
- **GPU** recommended (AMD MI300X, NVIDIA A100/H100)
- **Webcam** for real-time deployment
- **SO-101 robot** (optional — deploy works in `--no-robot` mode)

```bash
conda activate act
pip install -r requirements.txt
```
