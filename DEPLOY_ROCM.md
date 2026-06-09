# Deploy Diffusion Policy on ROCm Setup

Deploy the trained Diffusion Policy (YOLO keypoint backend) for real-time gesture mimic on an SO-101 robot with an AMD ROCm GPU.

## Target Setup


| Component  | Version / Detail                  |
| ---------- | --------------------------------- |
| OS         | Ubuntu 22.04                      |
| GPU Stack  | ROCm 7.2.2                        |
| PyTorch    | 2.9.1+rocm7.2                     |
| LeRobot    | 0.5.2                             |
| Robot      | SO-101 (Feetech servos, 6 joints) |
| Robot Port | `/dev/ttyACM0`                    |
| Camera     | `/dev/video0` (OpenCV index 0)    |


## 1. Files to Copy

Transfer the minimal file set (~310 MB) from the training machine:

```
gesture_mimic/
├── deploy_diffusion.py                     # main inference script
├── yolo_preprocessor.py                    # YOLO seg + pose preprocessing
├── pose_estimator.py                       # pose estimation utilities
├── yolo26n-seg.pt                          # YOLO segmentation weights (~6.4 MB)
├── yolo26n-pose.pt                         # YOLO pose weights (~7.5 MB)
└── outputs/train/diffusion_gesture_yolo/
    └── checkpoints/080000/
        └── pretrained_model/               # trained model (~300 MB, 7 files)
            ├── config.json
            ├── model.safetensors
            ├── train_config.json
            ├── policy_preprocessor.json
            ├── policy_preprocessor_step_3_normalizer_processor.safetensors
            ├── policy_postprocessor.json
            └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

You do **not** need `training_state/`, the `data/` directory, or any evaluation scripts.

```bash
# From the training machine, inside gesture_mimic/
rsync -avz --progress \
  deploy_diffusion.py yolo_preprocessor.py pose_estimator.py \
  yolo26n-seg.pt yolo26n-pose.pt \
  outputs/train/diffusion_gesture_yolo/checkpoints/080000/pretrained_model \
  user@target-machine:/path/to/gesture_mimic/
```

## 2. Environment Setup

```bash
# Create a virtual environment
python3 -m venv ~/diffusion-env
source ~/diffusion-env/bin/activate

# PyTorch for ROCm 7.2
pip3 install torch==2.9.1 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/rocm7.2

# LeRobot
pip install lerobot==0.5.2

# Other dependencies
pip install opencv-python>=4.8 ultralytics>=8.3 numpy>=1.24
```

## 3. Hardware Setup

```bash
# Find the robot serial port
lerobot-find-port

# Grant serial access (after connecting USB)
sudo chmod 666 /dev/ttyACM0

# Find attached cameras
lerobot-find-cameras opencv
# Typically /dev/video0 -> camera-id 0
```

## 4. Run Inference

```bash
cd /path/to/gesture_mimic

# ROCm MIOpen kernel tuning (recommended, especially on first run)
export MIOPEN_FIND_ENFORCE=5
export MIOPEN_FIND_MODE=2

python deploy_diffusion.py \
    --checkpoint ../diffusion_yolo/pretrained_model \
    --use-keypoints --pose-backend yolo \
    --port /dev/ttyACM0 \
    --camera-id 0 \
    --fps 30
```

### Test Without Robot (Optional)

Run with `--no-robot` to verify the pipeline visually before connecting hardware:

```bash
python deploy_diffusion.py \
    --checkpoint outputs/train/diffusion_gesture_yolo/checkpoints/080000/pretrained_model \
    --use-keypoints --pose-backend yolo \
    --no-robot \
    --camera-id 0
```

This opens a preview window showing the YOLO-processed frame and predicted actions overlay without sending commands to the robot.

## 5. Runtime Controls


| Key | Action                                  |
| --- | --------------------------------------- |
| `q` | Quit                                    |
| `r` | Reset policy state and action ensembler |


## 6. CLI Reference


| Argument           | Default        | Description                           |
| ------------------ | -------------- | ------------------------------------- |
| `--checkpoint`     | (required)     | Path to `pretrained_model/` directory |
| `--device`         | auto (`cuda`)  | PyTorch device (auto-detects ROCm)    |
| `--fps`            | 30             | Target loop frequency                 |
| `--port`           | `/dev/ttyUSB0` | Robot serial port                     |
| `--camera-id`      | 0              | Webcam device index                   |
| `--no-robot`       | off            | Skip hardware, visualize only         |
| `--use-keypoints`  | off            | Enable pose estimation preprocessing  |
| `--pose-backend`   | `mediapipe`    | `mediapipe` or `yolo`                 |
| `--video`          | none           | Use video file instead of live webcam |
| `--output`         | auto           | Path to save annotated output video   |
| `--ensemble-k`     | 4              | Rolling action average window size    |
| `--ensemble-alpha` | 0.5            | Exponential decay weight for ensemble |
| `--no-ensemble`    | off            | Disable temporal action ensembling    |


## Notes

- **No code changes required** -- `deploy_diffusion.py` handles SO-101 via `lerobot.robots.so_follower.SOFollower`.
- **ROCm GPU auto-detection** -- PyTorch's `torch.cuda.is_available()` returns `True` on ROCm, so `--device` does not need to be specified.
- **First run latency** -- MIOpen may take ~30 seconds to tune GPU kernels on the first run. Subsequent runs are fast.
- **YOLO on CPU fallback** -- If `ultralytics` has issues on ROCm, YOLO inference falls back to CPU. Only the diffusion policy needs the GPU.
- **Port difference** -- The target setup uses `/dev/ttyACM0` (not `/dev/ttyUSB0` which is the script default), so always pass `--port /dev/ttyACM0`.

