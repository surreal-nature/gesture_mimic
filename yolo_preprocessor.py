"""YOLO-based image preprocessing for gesture mimic.

Uses YOLOv26n-seg to remove background and YOLOv26n-pose to detect
6 upper-body arm keypoints, then overlays colored keypoint circles
and an inverted-U skeleton on the segmented image.

The output is a modified RGB image fed to the standard ACT RGB pipeline
(ResNet-18 vision backbone). Unlike the MediaPipe keypoint mode which
replaces images with numeric features, this approach keeps images but
makes them cleaner and more informative for the policy.

Usage:
    preprocessor = YoloPosePreprocessor()
    modified = preprocessor.process_frame(rgb_frame)  # (H, W, 3) uint8
    preprocessor.close()
"""

import cv2
import numpy as np


ARM_KEYPOINT_INDICES = [5, 6, 7, 8, 9, 10]
ARM_KEYPOINT_NAMES = [
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]

SKELETON_CONNECTIONS = [(9, 7), (7, 5), (5, 6), (6, 8), (8, 10)]

KEYPOINT_COLORS_BGR = {
    5: (0, 255, 0),
    6: (0, 180, 0),
    7: (255, 0, 0),
    8: (255, 140, 0),
    9: (0, 0, 255),
    10: (0, 0, 200),
}

SKELETON_LINE_COLOR_BGR = (255, 255, 255)


class YoloPosePreprocessor:
    """Segments person and overlays arm skeleton on RGB frames.

    Args:
        seg_model: YOLOv26n-seg model name or path (auto-downloads).
        pose_model: YOLOv26n-pose model name or path (auto-downloads).
        device: Device for inference ('cuda' or 'cpu').
        confidence_threshold: Minimum keypoint confidence to draw.
        keypoint_radius: Circle radius for keypoint markers (pixels).
        line_thickness: Skeleton line thickness (pixels).
    """

    def __init__(
        self,
        seg_model="yolo26n-seg.pt",
        pose_model="yolo26n-pose.pt",
        device="cuda",
        confidence_threshold=0.5,
        keypoint_radius=6,
        line_thickness=3,
    ):
        from ultralytics import YOLO

        self.seg = YOLO(seg_model)
        self.pose = YOLO(pose_model)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.keypoint_radius = keypoint_radius
        self.line_thickness = line_thickness
        self._detection_failures = 0

    def process_frame(self, rgb_frame):
        """Full pipeline: segment person, overlay arm skeleton.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB numpy array.

        Returns:
            Modified (H, W, 3) uint8 RGB numpy array with background removed
            and arm skeleton overlaid. Returns all-black if no person detected.
        """
        h, w = rgb_frame.shape[:2]

        mask = self._segment_person(rgb_frame)
        if mask is None:
            self._detection_failures += 1
            return np.zeros_like(rgb_frame)

        masked = self._apply_mask(rgb_frame, mask)

        keypoints = self._detect_arm_keypoints(rgb_frame)
        if keypoints is not None:
            xy, conf = keypoints
            masked = self._draw_skeleton(masked, xy, conf)
        else:
            self._detection_failures += 1

        return masked

    def _segment_person(self, rgb_frame):
        """Run segmentation, return binary mask for highest-confidence person."""
        h, w = rgb_frame.shape[:2]
        results = self.seg(rgb_frame, device=self.device, verbose=False, classes=[0])

        if not results or results[0].masks is None or len(results[0].masks.data) == 0:
            return None

        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes

        if len(boxes) > 1:
            areas = boxes.xyxy.cpu().numpy()
            box_areas = (areas[:, 2] - areas[:, 0]) * (areas[:, 3] - areas[:, 1])
            best_idx = int(np.argmax(box_areas))
        else:
            best_idx = 0

        mask = masks[best_idx]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask > 0.5).astype(np.uint8)

    def _detect_arm_keypoints(self, rgb_frame):
        """Run pose estimation, return (6, 2) xy coords and (6,) confidences."""
        results = self.pose(rgb_frame, device=self.device, verbose=False)

        if not results or results[0].keypoints is None:
            return None

        kp = results[0].keypoints
        if kp.xy.shape[0] == 0:
            return None

        if kp.xy.shape[0] > 1:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(box_areas))
        else:
            best_idx = 0

        xy = kp.xy[best_idx].cpu().numpy()
        conf = kp.conf[best_idx].cpu().numpy()

        arm_xy = xy[ARM_KEYPOINT_INDICES]
        arm_conf = conf[ARM_KEYPOINT_INDICES]

        return arm_xy, arm_conf

    def _apply_mask(self, rgb_frame, mask):
        """Zero out background pixels."""
        return rgb_frame * mask[:, :, np.newaxis]

    def _draw_skeleton(self, image, keypoints_xy, keypoints_conf):
        """Draw colored keypoint circles and connecting lines."""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for src_idx, dst_idx in SKELETON_CONNECTIONS:
            src_pos = ARM_KEYPOINT_INDICES.index(src_idx)
            dst_pos = ARM_KEYPOINT_INDICES.index(dst_idx)

            if (keypoints_conf[src_pos] < self.confidence_threshold or
                    keypoints_conf[dst_pos] < self.confidence_threshold):
                continue

            pt1 = tuple(keypoints_xy[src_pos].astype(int))
            pt2 = tuple(keypoints_xy[dst_pos].astype(int))
            cv2.line(image_bgr, pt1, pt2, SKELETON_LINE_COLOR_BGR, self.line_thickness)

        for i, coco_idx in enumerate(ARM_KEYPOINT_INDICES):
            if keypoints_conf[i] < self.confidence_threshold:
                continue
            pt = tuple(keypoints_xy[i].astype(int))
            color = KEYPOINT_COLORS_BGR[coco_idx]
            cv2.circle(image_bgr, pt, self.keypoint_radius, color, -1)
            cv2.circle(image_bgr, pt, self.keypoint_radius, (255, 255, 255), 1)

        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @property
    def detection_failure_count(self):
        return self._detection_failures

    def close(self):
        """Release model resources."""
        del self.seg
        del self.pose
