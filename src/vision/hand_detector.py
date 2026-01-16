"""
Hand detection module using MediaPipe.
Detects two open palms and maps distance to BPM control.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from logger import get_logger

logger = get_logger(__name__)

# Graceful degradation for MediaPipe
MEDIAPIPE_AVAILABLE = False
mp = None

try:
    import mediapipe as mp
    # Test if solutions module is accessible
    _ = mp.solutions.hands
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe loaded successfully for hand detection")
except (ImportError, AttributeError, OSError) as e:
    logger.warning(f"MediaPipe not available: {e}")
    logger.warning("Hand detection will be disabled")
    mp = None


class HandDetector:
    """
    Detects hands using MediaPipe.
    Distance between two open palms controls BPM.
    """
    
    def __init__(self, min_bpm: int = 20, max_bpm: int = 220,
                 min_distance: int = 80, max_distance: int = 880):
        """
        Args:
            min_bpm: BPM when hands are closest
            max_bpm: BPM when hands are furthest apart
            min_distance: Minimum hand distance in pixels
            max_distance: Maximum hand distance in pixels
        """
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        # MediaPipe hands - with graceful fallback
        if not MEDIAPIPE_AVAILABLE or mp is None:
            logger.warning("HandDetector: MediaPipe not available, hand detection disabled")
            self.enabled = False
            self.hands = None
            self.mp_draw = None
            self.mp_hands = None
        else:
            try:
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
                self.mp_draw = mp.solutions.drawing_utils
                self.enabled = True
                logger.info("HandDetector initialized successfully")
            except (AttributeError, ImportError, OSError, RuntimeError) as e:
                logger.error(f"HandDetector initialization failed: {e}", exc_info=True)
                self.enabled = False
                self.hands = None
                self.mp_draw = None
                self.mp_hands = None
        
        # State
        self.last_bpm: Optional[int] = None
        self.hands_detected = False
        self.bpm_active = False
        self._active_streak = 0
        self._inactive_streak = 0
        self._active_required_frames = 5
        self._release_required_frames = 8
        self.last_distance: Optional[float] = None
        self.status = "idle"

    def reset_state(self):
        """Reset detection state."""
        self.last_bpm = None
        self.hands_detected = False
        self.bpm_active = False
        self._active_streak = 0
        self._inactive_streak = 0
        self.last_distance = None
        self.status = "idle"

    def _hand_open(self, landmarks) -> bool:
        """Check if hand is open based on finger extension."""
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z], dtype=np.float32)
        # Check 4 fingers (index, middle, ring, pinky)
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
        extended = 0
        for tip_idx, mcp_idx in pairs:
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z], dtype=np.float32)
            mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z], dtype=np.float32)
            if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.1:
                extended += 1
        return extended >= 3

    def _palm_center(self, landmarks, frame_shape) -> tuple:
        """Calculate palm center from key landmarks."""
        h, w = frame_shape[:2]
        # Average of wrist and base of each finger
        ids = [0, 5, 9, 13, 17]
        xs = [landmarks[i].x for i in ids]
        ys = [landmarks[i].y for i in ids]
        return (int(np.mean(xs) * w), int(np.mean(ys) * h))
    
    def process(self, frame: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
        """
        Process frame for hand detection.
        
        Args:
            frame: BGR frame from camera
            
        Returns:
            (bpm, annotated_frame) - bpm is None if no BPM change
        """
        if not self.enabled or self.hands is None:
            return None, frame
        
        try:
            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)
        except Exception as e:
            logger.error(f"MediaPipe hand processing error: {e}", exc_info=True)
            return None, frame
        
        bpm = None
        self.hands_detected = False
        
        # Find candidate hand pair
        candidate = None
        if results.multi_hand_landmarks:
            hand_entries = []
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = None
                if results.multi_handedness and idx < len(results.multi_handedness):
                    label = results.multi_handedness[idx].classification[0].label
                hand_entries.append((label, hand_landmarks))

            # Find open hands
            open_hands = []
            for label, hand_landmarks in hand_entries:
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self._hand_open(hand_landmarks.landmark):
                    center = self._palm_center(hand_landmarks.landmark, frame.shape)
                    open_hands.append((label, center))

            # Need exactly 2 open hands
            left_hand = next((h for h in open_hands if h[0] == "Left"), None)
            right_hand = next((h for h in open_hands if h[0] == "Right"), None)
            if left_hand and right_hand:
                candidate = (left_hand[1], right_hand[1])
            elif len(open_hands) >= 2 and all(h[0] is None for h in open_hands[:2]):
                # Fallback if handedness not detected
                candidate = (open_hands[0][1], open_hands[1][1])

        # Update activation streaks
        if candidate:
            self._active_streak += 1
            self._inactive_streak = 0
        else:
            self._inactive_streak += 1
            self._active_streak = 0

        # State machine: activate after N frames of consistent detection
        if not self.bpm_active and self._active_streak >= self._active_required_frames:
            self.bpm_active = True
            logger.debug("Hand BPM control activated")

        # Deactivate after N frames of no detection
        if self.bpm_active and self._inactive_streak >= self._release_required_frames:
            self.bpm_active = False
            logger.debug("Hand BPM control deactivated")

        # Calculate BPM if active
        if candidate and self.bpm_active:
            centers = candidate
            self.hands_detected = True
            
            # Draw visual feedback
            cv2.circle(frame, centers[0], 10, (0, 255, 0), -1)
            cv2.circle(frame, centers[1], 10, (0, 255, 0), -1)
            cv2.line(frame, centers[0], centers[1], (0, 255, 255), 3)

            # Calculate distance
            distance = np.sqrt(
                (centers[0][0] - centers[1][0])**2 +
                (centers[0][1] - centers[1][1])**2
            )
            self.last_distance = float(distance)

            # Map distance to BPM
            min_dist = self.min_distance
            max_dist = self.max_distance
            if max_dist <= min_dist:
                max_dist = min_dist + 1.0

            t = (distance - min_dist) / (max_dist - min_dist)
            t = max(0, min(1, t))
            bpm = int(self.min_bpm + t * (self.max_bpm - self.min_bpm))
            self.last_bpm = bpm
            self.status = "active"
            
        elif candidate and not self.bpm_active:
            # Hands detected but not active yet
            self.hands_detected = False
            self.status = "hold"
        else:
            # No hands or not active
            self.hands_detected = False
            self.status = "idle"
        
        return bpm, frame
    
    def close(self):
        """Clean up MediaPipe resources."""
        if self.enabled and self.hands:
            try:
                self.hands.close()
                logger.debug("HandDetector closed")
            except Exception as e:
                logger.warning(f"Error closing HandDetector: {e}")
