"""Auto-zoom and framing using face detection with MediaPipe."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from opengling.core.models import ProcessingConfig, ZoomKeyframe

logger = logging.getLogger(__name__)


class AutoZoomProcessor:
    """Applies automatic zoom and framing based on face detection."""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._face_detector = None

    def _load_face_detector(self):
        """Lazy load MediaPipe face detection."""
        if self._face_detector is not None:
            return

        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required for auto-zoom. "
                "Install with: pip install opengling[zoom]"
            )

        self._mp_face = mp.solutions.face_detection
        self._face_detector = self._mp_face.FaceDetection(
            model_selection=1,  # Full range model (better for video)
            min_detection_confidence=0.5,
        )

    def generate_zoom_keyframes(
        self,
        video_path: Path | str,
        sample_rate: float = 1.0,  # Analyze every N seconds
    ) -> list[ZoomKeyframe]:
        """
        Generate zoom keyframes based on face detection.

        Args:
            video_path: Path to video file
            sample_rate: How often to sample frames (seconds)

        Returns:
            List of zoom keyframes for the video
        """
        if not self.config.auto_zoom:
            return []

        video_path = Path(video_path)
        logger.info(f"Generating zoom keyframes for {video_path.name}")

        self._load_face_detector()

        try:
            import cv2
        except ImportError:
            raise ImportError("opencv-python is required. Install with: pip install opencv-python")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        sample_interval = int(fps * sample_rate)

        keyframes = []
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process at sample interval
            if frame_num % sample_interval == 0:
                time = frame_num / fps
                keyframe = self._process_frame(
                    frame, time, frame_width, frame_height
                )
                if keyframe:
                    keyframes.append(keyframe)

            frame_num += 1

        cap.release()

        # Apply smoothing
        keyframes = self._smooth_keyframes(keyframes)

        logger.info(f"Generated {len(keyframes)} zoom keyframes")
        return keyframes

    def _process_frame(
        self,
        frame: np.ndarray,
        time: float,
        frame_width: int,
        frame_height: int,
    ) -> Optional[ZoomKeyframe]:
        """Process a single frame and generate zoom keyframe."""
        import cv2

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self._face_detector.process(rgb_frame)

        if not results.detections:
            # No face detected, return neutral keyframe
            return ZoomKeyframe(
                time=time,
                zoom_level=1.0,
                center_x=0.5,
                center_y=0.5,
            )

        # Get the largest/most prominent face
        best_detection = max(
            results.detections,
            key=lambda d: d.location_data.relative_bounding_box.width *
                         d.location_data.relative_bounding_box.height
        )

        bbox = best_detection.location_data.relative_bounding_box

        # Calculate center of face (relative coordinates 0-1)
        center_x = bbox.xmin + bbox.width / 2
        center_y = bbox.ymin + bbox.height / 2

        # Calculate appropriate zoom level
        # Larger face = less zoom needed
        face_area = bbox.width * bbox.height

        # Target: face should take up ~20-30% of frame when zoomed
        target_face_ratio = 0.25
        current_face_ratio = face_area

        if current_face_ratio > 0:
            zoom_level = min(
                self.config.max_zoom,
                max(1.0, (target_face_ratio / current_face_ratio) ** 0.5)
            )
        else:
            zoom_level = 1.0

        # Constrain center to keep frame within bounds at zoom level
        margin = (1.0 - 1.0 / zoom_level) / 2
        center_x = max(margin, min(1.0 - margin, center_x))
        center_y = max(margin, min(1.0 - margin, center_y))

        return ZoomKeyframe(
            time=time,
            zoom_level=zoom_level,
            center_x=center_x,
            center_y=center_y,
        )

    def _smooth_keyframes(
        self,
        keyframes: list[ZoomKeyframe],
    ) -> list[ZoomKeyframe]:
        """Apply smoothing to keyframes to avoid jittery motion."""
        if len(keyframes) < 3:
            return keyframes

        smoothing = self.config.zoom_smoothing

        # Simple exponential moving average
        smoothed = [keyframes[0]]

        for i in range(1, len(keyframes)):
            prev = smoothed[-1]
            curr = keyframes[i]

            # Interpolate
            alpha = 1.0 - smoothing

            smoothed.append(ZoomKeyframe(
                time=curr.time,
                zoom_level=prev.zoom_level * (1 - alpha) + curr.zoom_level * alpha,
                center_x=prev.center_x * (1 - alpha) + curr.center_x * alpha,
                center_y=prev.center_y * (1 - alpha) + curr.center_y * alpha,
            ))

        return smoothed


def apply_zoom_to_frame(
    frame: np.ndarray,
    keyframe: ZoomKeyframe,
) -> np.ndarray:
    """
    Apply zoom transformation to a frame.

    Args:
        frame: Input frame (numpy array)
        keyframe: Zoom keyframe to apply

    Returns:
        Zoomed and cropped frame
    """
    import cv2

    h, w = frame.shape[:2]

    zoom = keyframe.zoom_level
    cx, cy = keyframe.center_x, keyframe.center_y

    # Calculate crop region
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    # Calculate top-left corner of crop
    x1 = int(cx * w - crop_w / 2)
    y1 = int(cy * h - crop_h / 2)

    # Clamp to frame bounds
    x1 = max(0, min(w - crop_w, x1))
    y1 = max(0, min(h - crop_h, y1))

    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Crop and resize back to original dimensions
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return zoomed


def interpolate_keyframe(
    keyframes: list[ZoomKeyframe],
    time: float,
) -> ZoomKeyframe:
    """
    Interpolate zoom keyframe for a specific time.

    Args:
        keyframes: List of zoom keyframes
        time: Time to interpolate for

    Returns:
        Interpolated keyframe
    """
    if not keyframes:
        return ZoomKeyframe(time=time, zoom_level=1.0, center_x=0.5, center_y=0.5)

    if time <= keyframes[0].time:
        return keyframes[0]

    if time >= keyframes[-1].time:
        return keyframes[-1]

    # Find surrounding keyframes
    for i in range(len(keyframes) - 1):
        if keyframes[i].time <= time <= keyframes[i + 1].time:
            k1, k2 = keyframes[i], keyframes[i + 1]

            # Linear interpolation factor
            t = (time - k1.time) / (k2.time - k1.time) if k2.time != k1.time else 0

            return ZoomKeyframe(
                time=time,
                zoom_level=k1.zoom_level + t * (k2.zoom_level - k1.zoom_level),
                center_x=k1.center_x + t * (k2.center_x - k1.center_x),
                center_y=k1.center_y + t * (k2.center_y - k1.center_y),
            )

    return keyframes[-1]

