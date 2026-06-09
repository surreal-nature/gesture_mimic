"""Deploy a trained Diffusion Policy for gesture mimic on a real SO-101 robot.

The robot observes a human via webcam and reproduces their gestures
in real-time using a trained Diffusion Policy. Unlike ACT which predicts
action chunks directly, Diffusion Policy generates smooth trajectories
via iterative denoising, producing more natural robot motions.

Supports three modes (controlled via --use-keypoints and --pose-backend flags):
  RGB mode:              webcam frame -> Diffusion (with vision backbone) -> robot action
  MediaPipe keypoint:    webcam frame -> MediaPipe -> keypoints -> Diffusion (state-only) -> robot action
  YOLO keypoint:         webcam frame -> YOLO seg+pose -> modified RGB -> Diffusion (with vision backbone) -> robot action

Usage:
    # RGB mode: real-time with webcam + SO-101 robot
    python deploy_diffusion.py --checkpoint outputs/train/diffusion_gesture/checkpoints/last/pretrained_model

    # MediaPipe keypoint mode
    python deploy_diffusion.py --checkpoint outputs/train/diffusion_gesture_kp/checkpoints/last/pretrained_model \
        --use-keypoints

    # YOLO keypoint mode
    python deploy_diffusion.py --checkpoint outputs/train/diffusion_gesture_yolo/checkpoints/last/pretrained_model \
        --use-keypoints --pose-backend yolo

    # Webcam-only mode (no robot)
    python deploy_diffusion.py --checkpoint <path> --no-robot

    # Replay from video file
    python deploy_diffusion.py --checkpoint <path> --video path/to/gesture.mp4 --no-robot
"""

import argparse
import os
import time
from collections import deque

import cv2
import numpy as np
import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_device(device):
    global DEVICE
    DEVICE = device


def opencv_gui_available():
    try:
        cv2.namedWindow("__gui_test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__gui_test__")
        return True
    except cv2.error:
        return False


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


class ActionEnsembler:
    """Rolling weighted average of recent actions to smooth jitter.

    Keeps the last K actions from select_action() and outputs their
    weighted average with exponentially decaying weights (most recent = highest).
    """

    def __init__(self, k=4, alpha=0.5, action_dim=6):
        self.k = k
        self.alpha = alpha
        self.action_dim = action_dim
        self._buffer = deque(maxlen=k)
        weights = np.array([alpha ** i for i in range(k)])[::-1]
        self._weights = weights / weights.sum()

    def add(self, action):
        self._buffer.append(action.copy())

    def get(self):
        n = len(self._buffer)
        if n == 0:
            return np.zeros(self.action_dim, dtype=np.float32)
        actions = np.stack(list(self._buffer))
        w = self._weights[-n:]
        w = w / w.sum()
        return (actions * w[:, None]).sum(axis=0).astype(np.float32)

    def reset(self):
        self._buffer.clear()


def load_policy(checkpoint_path, device):
    policy = DiffusionPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )
    return policy, preprocessor, postprocessor


def frame_to_observation_rgb(frame, state, preprocessor):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    state_tensor = torch.from_numpy(state).unsqueeze(0).float()

    batch = {
        "observation.state": state_tensor.to(DEVICE),
        "observation.images.human": img_tensor.to(DEVICE),
    }
    return preprocessor(batch)


def frame_to_observation_keypoints(frame, robot_state, pose_estimator, preprocessor):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoint_features = pose_estimator.process_frame(rgb)
    extended_state = np.concatenate([robot_state, keypoint_features])
    state_tensor = torch.from_numpy(extended_state).unsqueeze(0).float()
    batch = {"observation.state": state_tensor.to(DEVICE)}
    return preprocessor(batch)


def frame_to_observation_yolo(frame, state, yolo_preprocessor, preprocessor):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    modified_rgb = yolo_preprocessor.process_frame(rgb)
    img = cv2.resize(modified_rgb, (IMAGE_WIDTH, IMAGE_HEIGHT))
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    state_tensor = torch.from_numpy(state).unsqueeze(0).float()

    batch = {
        "observation.state": state_tensor.to(DEVICE),
        "observation.images.human": img_tensor.to(DEVICE),
    }
    return preprocessor(batch), modified_rgb


def connect_robot(port, robot_id="my_awesome_follower_arm", baudrate=1000000):
    try:
        import scservo_sdk  # noqa: F401 — required for SO-101 Feetech servos
    except ImportError as e:
        raise ImportError(
            "SO-101 robot connection requires the Feetech SDK. "
            "Install it with: pip install feetech-servo-sdk"
        ) from e

    from lerobot.robots.so_follower import SO101FollowerConfig, SO101Follower

    config = SO101FollowerConfig(port=port, id=robot_id)
    robot = SO101Follower(config)
    robot.connect()
    return robot


def read_robot_state(robot):
    try:
        state = robot.get_observation()
        return np.array([state[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    except (AttributeError, TypeError):
        positions = robot.read("Present_Position")
        return np.array(list(positions.values()), dtype=np.float32)


def send_robot_action(robot, action):
    try:
        robot.send_action({f"{name}.pos": float(action[i]) for i, name in enumerate(JOINT_NAMES)})
    except (AttributeError, TypeError):
        goal = {name: float(action[i]) for i, name in enumerate(JOINT_NAMES)}
        robot.write("Goal_Position", list(goal.values()))


def draw_skeleton(frame, pose_estimator):
    try:
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.pose.process(rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
            )
    except Exception:
        pass
    return frame


def draw_overlay(frame, action, state, fps, step, denoise_ms, use_keypoints=False, pose_backend="mediapipe"):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    panel_h = 220
    cv2.rectangle(overlay, (w - 280, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if use_keypoints and pose_backend == "yolo":
        mode_label = "DIFF+YOLO"
    elif use_keypoints:
        mode_label = "DIFF+KP"
    else:
        mode_label = "DIFFUSION"
    cv2.putText(frame, f"FPS: {fps:.1f}  Step: {step}  [{mode_label}]", (w - 270, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    if denoise_ms > 0:
        cv2.putText(frame, f"Denoise: {denoise_ms:.0f}ms", (w - 270, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    for i, name in enumerate(JOINT_NAMES):
        y = 56 + i * 22
        s = state[i] if state is not None else 0.0
        a = action[i] if action is not None else 0.0
        cv2.putText(frame, f"{name[:12]:>12}: {s:+.3f} -> {a:+.3f}", (w - 270, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def run_webcam(policy, preprocessor, postprocessor, robot, camera_id, fps_target, device,
               use_keypoints=False, pose_estimator=None, yolo_preprocessor=None,
               pose_backend="mediapipe", ensemble_k=0, ensemble_alpha=0.5, no_display=False):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    n_obs_steps = policy.config.n_obs_steps
    n_action_steps = policy.config.n_action_steps

    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if use_keypoints and pose_backend == "yolo":
        mode_str = "YOLO (segmented RGB + skeleton -> Diffusion Policy)"
    elif use_keypoints:
        mode_str = "KEYPOINT/MediaPipe (pose estimation -> Diffusion Policy)"
    else:
        mode_str = "RGB (vision backbone Diffusion Policy)"
    print(f"Mode: {mode_str}")
    print(f"Obs steps: {n_obs_steps}, Horizon: {policy.config.horizon}, Action steps: {n_action_steps}")
    print(f"Inference denoising steps: {policy.diffusion.num_inference_steps}")
    ensembler = ActionEnsembler(k=ensemble_k, alpha=ensemble_alpha) if ensemble_k > 1 else None
    if ensembler:
        print(f"Temporal ensembling: k={ensemble_k}, alpha={ensemble_alpha}")
    if no_display:
        print("Running headless (no preview window). Press Ctrl+C to quit.")
    else:
        print("Press 'q' to quit, 'r' to reset policy state")

    policy.reset()
    postprocessor.reset()
    if pose_estimator:
        pose_estimator.reset()
    if ensembler:
        ensembler.reset()
    robot_state = read_robot_state(robot) if robot else np.zeros(6, dtype=np.float32)
    step = 0
    frame_time = 1.0 / fps_target
    denoise_ms = 0.0

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            yolo_display_rgb = None
            if use_keypoints and yolo_preprocessor:
                batch, yolo_display_rgb = frame_to_observation_yolo(
                    frame, robot_state, yolo_preprocessor, preprocessor)
            elif use_keypoints and pose_estimator:
                batch = frame_to_observation_keypoints(frame, robot_state, pose_estimator, preprocessor)
            else:
                batch = frame_to_observation_rgb(frame, robot_state, preprocessor)

            t_denoise_start = time.time()
            with torch.no_grad():
                action = policy.select_action(batch)
                action = postprocessor(action)
            denoise_ms = (time.time() - t_denoise_start) * 1000

            raw_action = action.squeeze(0).cpu().numpy()

            if ensembler:
                ensembler.add(raw_action)
                current_action = ensembler.get()
            else:
                current_action = raw_action

            if robot:
                send_robot_action(robot, current_action)
                robot_state = read_robot_state(robot)

            elapsed = time.time() - t_start
            fps_actual = 1.0 / max(elapsed, 1e-6)

            if yolo_display_rgb is not None:
                frame = cv2.cvtColor(yolo_display_rgb, cv2.COLOR_RGB2BGR)
            elif use_keypoints and pose_estimator:
                draw_skeleton(frame, pose_estimator)

            sleep_time = frame_time - elapsed
            step += 1

            if no_display:
                if step == 1 or step % 30 == 0:
                    print(f"Step {step}, FPS {fps_actual:.1f}, denoise {denoise_ms:.0f}ms")
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                display = draw_overlay(frame, current_action, robot_state, fps_actual, step,
                                       denoise_ms, use_keypoints, pose_backend)
                cv2.imshow("Gesture Mimic - Diffusion Policy", display)
                wait_ms = max(1, int(sleep_time * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    policy.reset()
                    postprocessor.reset()
                    if pose_estimator:
                        pose_estimator.reset()
                    if ensembler:
                        ensembler.reset()
                    denoise_ms = 0.0
                    print(f"Policy reset at step {step}")

    finally:
        cap.release()
        if not no_display:
            cv2.destroyAllWindows()
        if robot:
            try:
                robot.disconnect()
            except Exception:
                pass

    detector = yolo_preprocessor or pose_estimator
    if detector:
        failures = detector.detection_failure_count
        if failures > 0:
            print(f"Detection failures: {failures}/{step} ({failures / max(step, 1) * 100:.1f}%)")
    print(f"Finished after {step} steps")


def run_video(policy, preprocessor, postprocessor, video_path, output_path, device,
              use_keypoints=False, pose_estimator=None, yolo_preprocessor=None,
              pose_backend="mediapipe", ensemble_k=0, ensemble_alpha=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")
    print(f"Obs steps: {policy.config.n_obs_steps}, Horizon: {policy.config.horizon}, "
          f"Action steps: {policy.config.n_action_steps}")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    policy.reset()
    if pose_estimator:
        pose_estimator.reset()
    ensembler = ActionEnsembler(k=ensemble_k, alpha=ensemble_alpha) if ensemble_k > 1 else None
    robot_state = np.zeros(6, dtype=np.float32)
    all_actions = []
    denoise_ms = 0.0

    step = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_display_rgb = None
        if use_keypoints and yolo_preprocessor:
            batch, yolo_display_rgb = frame_to_observation_yolo(
                frame, robot_state, yolo_preprocessor, preprocessor)
        elif use_keypoints and pose_estimator:
            batch = frame_to_observation_keypoints(frame, robot_state, pose_estimator, preprocessor)
        else:
            batch = frame_to_observation_rgb(frame, robot_state, preprocessor)

        t_denoise = time.time()
        with torch.no_grad():
            action = policy.select_action(batch)
            action = postprocessor(action)
        denoise_ms = (time.time() - t_denoise) * 1000

        raw_action = action.squeeze(0).cpu().numpy()
        if ensembler:
            ensembler.add(raw_action)
            current_action = ensembler.get()
        else:
            current_action = raw_action
        all_actions.append(current_action.copy())

        if yolo_display_rgb is not None:
            frame = cv2.cvtColor(yolo_display_rgb, cv2.COLOR_RGB2BGR)
        elif use_keypoints and pose_estimator:
            draw_skeleton(frame, pose_estimator)
        display = draw_overlay(frame, current_action, robot_state, fps, step,
                               denoise_ms, use_keypoints, pose_backend)
        if writer:
            writer.write(display)
        step += 1

        if step % 100 == 0:
            print(f"  Processed {step}/{total} frames")

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video: {output_path}")

    detector = yolo_preprocessor or pose_estimator
    if detector:
        failures = detector.detection_failure_count
        if failures > 0:
            print(f"Detection failures: {failures}/{step} ({failures / max(step, 1) * 100:.1f}%)")

    actions_path = output_path.replace(".mp4", "_actions.npy") if output_path else "predicted_actions.npy"
    np.save(actions_path, np.array(all_actions))
    print(f"Saved predicted actions: {actions_path} ({len(all_actions)} frames x {len(JOINT_NAMES)} joints)")


def main():
    parser = argparse.ArgumentParser(description="Deploy Diffusion Policy for gesture mimic")
    parser.add_argument("--checkpoint", required=True, help="Pretrained model path or HuggingFace model ID")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for real-time loop")

    parser.add_argument("--no-robot", action="store_true", help="Run without robot (webcam-only visualization)")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Robot serial port")
    parser.add_argument(
        "--robot-id",
        default="my_awesome_follower_arm",
        help="Robot ID matching calibration file (~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json)",
    )

    parser.add_argument("--no-display", action="store_true",
                        help="Run without OpenCV preview window (robot still moves)")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (instead of webcam)")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")

    parser.add_argument("--use-keypoints", action="store_true",
                        help="Use pose estimation keypoints instead of raw RGB images")
    parser.add_argument("--pose-backend", default="mediapipe",
                        choices=["mediapipe", "yolo"],
                        help="Pose backend (mediapipe=state-only, yolo=segmented RGB)")
    parser.add_argument("--landmarks", default="upper_body",
                        choices=["upper_body", "right_arm", "left_arm"],
                        help="Landmark preset for MediaPipe keypoint mode")
    parser.add_argument("--no-velocity", action="store_true", help="Disable velocity features (MediaPipe mode)")
    parser.add_argument("--no-acceleration", action="store_true", help="Disable acceleration features (MediaPipe mode)")
    parser.add_argument("--smoothing", type=float, default=0.3, help="EMA smoothing alpha (MediaPipe mode)")

    parser.add_argument("--ensemble-k", type=int, default=4,
                        help="Number of recent actions to blend for temporal ensembling (default: 4, set 0 or 1 to disable)")
    parser.add_argument("--ensemble-alpha", type=float, default=0.5,
                        help="Exponential decay base for ensemble weights (default: 0.5)")
    parser.add_argument("--no-ensemble", action="store_true",
                        help="Disable temporal ensembling")

    args = parser.parse_args()

    _set_device(args.device)

    print(f"Loading Diffusion Policy from: {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device)
    print(f"Policy loaded. Action dim: {policy.config.action_feature.shape}")
    print(f"Image features: {policy.config.image_features}")
    print(f"Horizon: {policy.config.horizon}, Action steps: {policy.config.n_action_steps}, "
          f"Obs steps: {policy.config.n_obs_steps}")
    print(f"Noise scheduler: {policy.config.noise_scheduler_type}, "
          f"Inference steps: {policy.diffusion.num_inference_steps}")

    pose_estimator = None
    yolo_preprocessor = None
    if args.use_keypoints:
        if args.pose_backend == "yolo":
            from yolo_preprocessor import YoloPosePreprocessor
            yolo_preprocessor = YoloPosePreprocessor(device=args.device)
            print(f"YOLO mode: segmented RGB + arm skeleton overlay (vision backbone Diffusion)")
        else:
            from pose_estimator import PoseEstimator
            pose_estimator = PoseEstimator(
                landmarks=args.landmarks,
                include_velocity=not args.no_velocity,
                include_acceleration=not args.no_acceleration,
                smoothing_alpha=args.smoothing,
            )
            print(f"MediaPipe mode: {args.landmarks} preset, {pose_estimator.feature_dim} features")

    ens_k = 0 if args.no_ensemble else args.ensemble_k
    ens_alpha = args.ensemble_alpha

    if args.video:
        output = args.output or args.video.replace(".mp4", "_diffusion_annotated.mp4")
        run_video(policy, preprocessor, postprocessor, args.video, output, args.device,
                  use_keypoints=args.use_keypoints, pose_estimator=pose_estimator,
                  yolo_preprocessor=yolo_preprocessor, pose_backend=args.pose_backend,
                  ensemble_k=ens_k, ensemble_alpha=ens_alpha)
    else:
        robot = None
        if not args.no_robot:
            print(f"Connecting to robot on {args.port} (id={args.robot_id})...")
            robot = connect_robot(args.port, robot_id=args.robot_id)
            print("Robot connected.")
        else:
            print("Running in no-robot mode (visualization only)")

        no_display = args.no_display
        if not no_display and not opencv_gui_available():
            print(
                "OpenCV GUI unavailable (likely opencv-python-headless). "
                "Running headless; use Ctrl+C to quit.\n"
                "To restore preview: pip uninstall opencv-python-headless && pip install opencv-python"
            )
            no_display = True

        run_webcam(policy, preprocessor, postprocessor, robot, args.camera_id, args.fps, args.device,
                   use_keypoints=args.use_keypoints, pose_estimator=pose_estimator,
                   yolo_preprocessor=yolo_preprocessor, pose_backend=args.pose_backend,
                   ensemble_k=ens_k, ensemble_alpha=ens_alpha, no_display=no_display)

    if pose_estimator:
        pose_estimator.close()
    if yolo_preprocessor:
        yolo_preprocessor.close()


if __name__ == "__main__":
    main()
