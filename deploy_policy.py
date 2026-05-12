"""Deploy a trained ACT gesture mimic policy on a real SO-101 robot.

The robot observes a human via webcam and reproduces their gestures
in real-time using the trained ACT policy.

Supports three modes (controlled via --use-keypoints and --pose-backend flags):
  RGB mode:              webcam frame -> ACT (with vision backbone) -> robot action
  MediaPipe keypoint:    webcam frame -> MediaPipe -> keypoints -> ACT (state-only) -> robot action
  YOLO keypoint:         webcam frame -> YOLO seg+pose -> modified RGB -> ACT (with vision backbone) -> robot action

Usage:
    # RGB mode: real-time with webcam + SO-101 robot
    python deploy_policy.py --checkpoint outputs/train/act_gesture/checkpoints/last/pretrained_model

    # MediaPipe keypoint mode: pose estimation + state-only policy
    python deploy_policy.py --checkpoint outputs/train/act_gesture_kp/checkpoints/last/pretrained_model \
        --use-keypoints

    # YOLO keypoint mode: segmented RGB + arm skeleton overlay
    python deploy_policy.py --checkpoint outputs/train/act_gesture_yolo/checkpoints/last/pretrained_model \
        --use-keypoints --pose-backend yolo

    # From HuggingFace Hub
    python deploy_policy.py --checkpoint AmolSapale181284/act_gesture_mimic_v4

    # Webcam-only mode (no robot, just show predicted joint angles)
    python deploy_policy.py --checkpoint <path> --no-robot

    # Replay from video file instead of webcam
    python deploy_policy.py --checkpoint <path> --video path/to/gesture.mp4 --no-robot

    # MediaPipe keypoint mode with custom landmark set
    python deploy_policy.py --checkpoint <path> --use-keypoints --landmarks right_arm --no-robot
"""

import argparse
import os
import time

import cv2
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def load_policy(checkpoint_path, device):
    policy = ACTPolicy.from_pretrained(checkpoint_path)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint_path),
    )
    return policy, preprocessor, postprocessor


def frame_to_observation_rgb(frame, state, preprocessor):
    """Convert a camera frame + robot state to a policy-ready batch (RGB mode)."""
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
    """Convert a camera frame to keypoints, concatenate with robot state (keypoint mode)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoint_features = pose_estimator.process_frame(rgb)
    extended_state = np.concatenate([robot_state, keypoint_features])
    state_tensor = torch.from_numpy(extended_state).unsqueeze(0).float()
    batch = {"observation.state": state_tensor.to(DEVICE)}
    return preprocessor(batch)


def frame_to_observation_yolo(frame, state, yolo_preprocessor, preprocessor):
    """YOLO mode: segment person, overlay skeleton, feed modified RGB to ACT."""
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


def connect_robot(port, baudrate=1000000):
    """Connect to a real SO-101 follower arm via Feetech serial."""
    try:
        from lerobot.robots.so_follower import SOFollowerConfig, SOFollower
        config = SOFollowerConfig(port=port, baudrate=baudrate)
        robot = SOFollower(config)
        robot.connect()
        return robot
    except ImportError:
        from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
        motors = FeetechMotorsBus(
            port=port,
            motors={
                "shoulder_pan": 1,
                "shoulder_lift": 2,
                "elbow_flex": 3,
                "wrist_flex": 4,
                "wrist_roll": 5,
                "gripper": 6,
            },
        )
        motors.connect()
        return motors


def read_robot_state(robot):
    """Read current joint positions from the robot."""
    try:
        state = robot.get_observation()
        return np.array([state[f"{name}.pos"] for name in JOINT_NAMES], dtype=np.float32)
    except (AttributeError, TypeError):
        positions = robot.read("Present_Position")
        return np.array(list(positions.values()), dtype=np.float32)


def send_robot_action(robot, action):
    """Send joint position targets to the robot."""
    try:
        robot.send_action({f"{name}.pos": float(action[i]) for i, name in enumerate(JOINT_NAMES)})
    except (AttributeError, TypeError):
        goal = {name: float(action[i]) for i, name in enumerate(JOINT_NAMES)}
        robot.write("Goal_Position", list(goal.values()))


def draw_skeleton(frame, pose_estimator):
    """Draw detected skeleton on the frame for visual feedback."""
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


def draw_overlay(frame, action, state, fps, step, use_keypoints=False, pose_backend="mediapipe"):
    """Draw joint angle overlay on the video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    panel_h = 200 if use_keypoints else 180
    cv2.rectangle(overlay, (w - 280, 0), (w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    if use_keypoints and pose_backend == "yolo":
        mode_label = "YOLO"
    elif use_keypoints:
        mode_label = "KEYPOINT"
    else:
        mode_label = "RGB"
    cv2.putText(frame, f"FPS: {fps:.1f}  Step: {step}  [{mode_label}]", (w - 270, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    for i, name in enumerate(JOINT_NAMES):
        y = 40 + i * 22
        s = state[i] if state is not None else 0.0
        a = action[i] if action is not None else 0.0
        cv2.putText(frame, f"{name[:12]:>12}: {s:+.3f} -> {a:+.3f}", (w - 270, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return frame


def run_webcam(policy, preprocessor, postprocessor, robot, camera_id, fps_target, device,
               use_keypoints=False, pose_estimator=None, yolo_preprocessor=None,
               pose_backend="mediapipe"):
    """Run real-time inference loop with webcam input."""
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    if use_keypoints and pose_backend == "yolo":
        mode_str = "YOLO (segmented RGB + skeleton overlay -> vision backbone ACT)"
    elif use_keypoints:
        mode_str = "KEYPOINT/MediaPipe (pose estimation -> state-only ACT)"
    else:
        mode_str = "RGB (vision backbone ACT)"
    print(f"Mode: {mode_str}")
    print("Press 'q' to quit, 'r' to reset policy state")

    policy.reset()
    if pose_estimator:
        pose_estimator.reset()
    robot_state = read_robot_state(robot) if robot else np.zeros(6, dtype=np.float32)
    action_chunk = None
    chunk_idx = 0
    step = 0
    frame_time = 1.0 / fps_target

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            yolo_display_rgb = None
            if action_chunk is None or chunk_idx >= len(action_chunk):
                if use_keypoints and yolo_preprocessor:
                    batch, yolo_display_rgb = frame_to_observation_yolo(
                        frame, robot_state, yolo_preprocessor, preprocessor)
                elif use_keypoints and pose_estimator:
                    batch = frame_to_observation_keypoints(frame, robot_state, pose_estimator, preprocessor)
                else:
                    batch = frame_to_observation_rgb(frame, robot_state, preprocessor)
                with torch.no_grad():
                    action = policy.select_action(batch)
                action_chunk = action.squeeze(0).cpu().numpy()
                chunk_idx = 0

            current_action = action_chunk[chunk_idx]
            chunk_idx += 1

            if robot:
                send_robot_action(robot, current_action)
                robot_state = read_robot_state(robot)

            elapsed = time.time() - t_start
            fps_actual = 1.0 / max(elapsed, 1e-6)

            if yolo_display_rgb is not None:
                frame = cv2.cvtColor(yolo_display_rgb, cv2.COLOR_RGB2BGR)
            elif use_keypoints and pose_estimator:
                draw_skeleton(frame, pose_estimator)

            display = draw_overlay(frame, current_action, robot_state, fps_actual, step,
                                   use_keypoints, pose_backend)
            cv2.imshow("Gesture Mimic - ACT Policy", display)
            step += 1

            sleep_time = frame_time - elapsed
            wait_ms = max(1, int(sleep_time * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                policy.reset()
                if pose_estimator:
                    pose_estimator.reset()
                action_chunk = None
                print(f"Policy reset at step {step}")

    finally:
        cap.release()
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
              pose_backend="mediapipe"):
    """Run inference on a pre-recorded video and save annotated output."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    policy.reset()
    if pose_estimator:
        pose_estimator.reset()
    robot_state = np.zeros(6, dtype=np.float32)
    action_chunk = None
    chunk_idx = 0
    all_actions = []

    step = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yolo_display_rgb = None
        if action_chunk is None or chunk_idx >= len(action_chunk):
            if use_keypoints and yolo_preprocessor:
                batch, yolo_display_rgb = frame_to_observation_yolo(
                    frame, robot_state, yolo_preprocessor, preprocessor)
            elif use_keypoints and pose_estimator:
                batch = frame_to_observation_keypoints(frame, robot_state, pose_estimator, preprocessor)
            else:
                batch = frame_to_observation_rgb(frame, robot_state, preprocessor)
            with torch.no_grad():
                action = policy.select_action(batch)
            action_chunk = action.squeeze(0).cpu().numpy()
            chunk_idx = 0

        current_action = action_chunk[chunk_idx]
        chunk_idx += 1
        all_actions.append(current_action.copy())

        if yolo_display_rgb is not None:
            frame = cv2.cvtColor(yolo_display_rgb, cv2.COLOR_RGB2BGR)
        elif use_keypoints and pose_estimator:
            draw_skeleton(frame, pose_estimator)
        display = draw_overlay(frame, current_action, robot_state, fps, step,
                               use_keypoints, pose_backend)
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
    parser = argparse.ArgumentParser(description="Deploy ACT gesture mimic policy")
    parser.add_argument("--checkpoint", required=True, help="Pretrained model path or HuggingFace model ID")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for real-time loop")

    # Robot config
    parser.add_argument("--no-robot", action="store_true", help="Run without robot (webcam-only visualization)")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Robot serial port")

    # Input source
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--video", type=str, default=None, help="Path to input video (instead of webcam)")
    parser.add_argument("--output", type=str, default=None, help="Path to save annotated output video")

    # Keypoint mode
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

    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    print(f"Loading policy from: {args.checkpoint}")
    policy, preprocessor, postprocessor = load_policy(args.checkpoint, args.device)
    print(f"Policy loaded. Action dim: {policy.config.action_feature.shape}")
    print(f"Image features: {policy.config.image_features}")
    print(f"Chunk size: {policy.config.chunk_size}, Action steps: {policy.config.n_action_steps}")

    pose_estimator = None
    yolo_preprocessor = None
    if args.use_keypoints:
        if args.pose_backend == "yolo":
            from yolo_preprocessor import YoloPosePreprocessor
            yolo_preprocessor = YoloPosePreprocessor(device=args.device)
            print(f"YOLO mode: segmented RGB + arm skeleton overlay (vision backbone ACT)")
        else:
            from pose_estimator import PoseEstimator
            pose_estimator = PoseEstimator(
                landmarks=args.landmarks,
                include_velocity=not args.no_velocity,
                include_acceleration=not args.no_acceleration,
                smoothing_alpha=args.smoothing,
            )
            print(f"MediaPipe mode: {args.landmarks} preset, {pose_estimator.feature_dim} features")

    if args.video:
        output = args.output or args.video.replace(".mp4", "_annotated.mp4")
        run_video(policy, preprocessor, postprocessor, args.video, output, args.device,
                  use_keypoints=args.use_keypoints, pose_estimator=pose_estimator,
                  yolo_preprocessor=yolo_preprocessor, pose_backend=args.pose_backend)
    else:
        robot = None
        if not args.no_robot:
            print(f"Connecting to robot on {args.port}...")
            robot = connect_robot(args.port)
            print("Robot connected.")
        else:
            print("Running in no-robot mode (visualization only)")

        run_webcam(policy, preprocessor, postprocessor, robot, args.camera_id, args.fps, args.device,
                   use_keypoints=args.use_keypoints, pose_estimator=pose_estimator,
                   yolo_preprocessor=yolo_preprocessor, pose_backend=args.pose_backend)

    if pose_estimator:
        pose_estimator.close()
    if yolo_preprocessor:
        yolo_preprocessor.close()


if __name__ == "__main__":
    main()
