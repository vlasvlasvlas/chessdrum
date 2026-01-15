"""
Camera module for ChessDrum.
Detects chessboard (pieces + rotation) and hands (BPM control).
"""
import cv2
import numpy as np
try:
    import mediapipe as mp
except ImportError:
    mp = None
from typing import Optional, Tuple, List
import threading
import time

from grid import BLACK


def _put_text_right(frame, text, y, color, scale=0.7, thickness=2, margin=10):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(margin, frame.shape[1] - text_w - margin)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness)


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
        
        # MediaPipe hands
        if mp is None:
            print("⚠ Hand detection disabled: mediapipe not installed")
            self.enabled = False
            self.hands = None
            self.mp_draw = None
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
            except (AttributeError, ImportError) as e:
                print(f"⚠ Hand detection disabled: {e}")
                self.enabled = False
                self.hands = None
                self.mp_draw = None
        
        # State
        self.last_bpm: Optional[int] = None
        self.hands_detected = False

    def _hand_open(self, landmarks) -> bool:
        wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z], dtype=np.float32)
        pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
        extended = 0
        for tip_idx, mcp_idx in pairs:
            tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z], dtype=np.float32)
            mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z], dtype=np.float32)
            if np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.1:
                extended += 1
        return extended >= 3

    def _palm_center(self, landmarks, frame_shape) -> tuple:
        h, w = frame_shape[:2]
        ids = [0, 5, 9, 13, 17]
        xs = [landmarks[i].x for i in ids]
        ys = [landmarks[i].y for i in ids]
        return (int(np.mean(xs) * w), int(np.mean(ys) * h))
    
    def process(self, frame: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
        """
        Process frame for hand detection.
        
        Returns:
            (bpm, annotated_frame) - bpm is None if no hand detected
        """
        if not self.enabled or self.hands is None:
            return None, frame
            
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        bpm = None
        self.hands_detected = False
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            centers = []
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                if self._hand_open(hand_landmarks.landmark):
                    centers.append(self._palm_center(hand_landmarks.landmark, frame.shape))

            if len(centers) == 2:
                self.hands_detected = True
                cv2.circle(frame, centers[0], 10, (0, 255, 0), -1)
                cv2.circle(frame, centers[1], 10, (0, 255, 0), -1)
                cv2.line(frame, centers[0], centers[1], (0, 255, 255), 3)

                distance = np.sqrt(
                    (centers[0][0] - centers[1][0])**2 +
                    (centers[0][1] - centers[1][1])**2
                )

                min_dist = self.min_distance
                max_dist = self.max_distance
                if max_dist <= min_dist:
                    max_dist = min_dist + 1.0

                t = (distance - min_dist) / (max_dist - min_dist)
                t = max(0, min(1, t))
                bpm = int(self.min_bpm + t * (self.max_bpm - self.min_bpm))
                self.last_bpm = bpm

                _put_text_right(frame, f"BPM: {bpm}", 60, (0, 255, 255), scale=1.5, thickness=3)
                _put_text_right(frame, f"Hands: {int(distance)}px", 100, (200, 200, 200), scale=0.8, thickness=2)
                _put_text_right(frame, "BPM ACTIVE", 130, (0, 255, 0), scale=0.7, thickness=2)
            else:
                _put_text_right(frame, "Show 2 open hands for BPM", 60, (100, 100, 255), scale=0.7, thickness=2)
        else:
            _put_text_right(frame, "Show 2 hands for BPM", 60, (100, 100, 255), scale=0.7, thickness=2)
        
        return bpm, frame
    
    def close(self):
        if self.enabled and self.hands:
            self.hands.close()


class BoardDetector:
    """
    Detects chessboard using contour detection.
    Finds the largest rectangular shape and analyzes cells for BLACK pieces only.
    """
    
    def __init__(self, debug_mode=False, sensitivity=0.5, dark_threshold=50):
        self.last_corners = None
        self.rotation_angle = 0.0
        self.board_rect = None
        self.corners = None
        self.piece_grid = np.zeros((8, 8), dtype=np.int8)
        self.manual_corners_norm = None  # Normalized 4-point corners (TL, TR, BR, BL)
        
        # Detection parameters (adjustable in real-time)
        self.sensitivity = sensitivity  # 0.0 to 1.0 (higher = more sensitive)
        self.dark_threshold = dark_threshold  # Base threshold for dark pixels
        
        # Temporal filtering: track detection history (STRICT)
        self.detection_history = []  # List of last N detection grids
        self.history_size = 7  # Require detection in 5/7 frames
        self.stable_grid = np.zeros((8, 8), dtype=np.int8)  # Stable output
        
        # Calibration data for adaptive thresholds
        self.empty_cells_baseline = None  # Store baseline brightness of empty squares
        self.calibrated = False
        self.debug_mode = debug_mode
        self.warped_debug = None  # For debug visualization

    def set_manual_corners(self, corners, frame_shape) -> bool:
        """Set manual board corners using absolute frame coordinates."""
        if corners is None or len(corners) != 4:
            return False
        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            return False
        norm = [(float(x) / w, float(y) / h) for x, y in corners]
        self.manual_corners_norm = np.array(norm, dtype=np.float32)
        self.calibrated = False
        self.detection_history.clear()
        return True

    def set_manual_corners_norm(self, corners_norm) -> bool:
        """Set manual board corners using normalized coordinates."""
        if corners_norm is None or len(corners_norm) != 4:
            return False
        pts = np.array(corners_norm, dtype=np.float32)
        pts = np.clip(pts, 0.0, 1.0)
        self.manual_corners_norm = pts
        self.calibrated = False
        self.detection_history.clear()
        return True

    def clear_manual_corners(self):
        """Disable manual board corners and return to auto detection."""
        self.manual_corners_norm = None
        self.board_rect = None
        self.corners = None
        self.calibrated = False
        self.detection_history.clear()

    def _get_manual_corners_abs(self, frame_shape):
        """Return manual corners in absolute pixel coordinates."""
        if self.manual_corners_norm is None:
            return None
        h, w = frame_shape[:2]
        if h <= 0 or w <= 0:
            return None
        pts = np.array([(p[0] * w, p[1] * h) for p in self.manual_corners_norm], dtype=np.float32)
        if pts.shape != (4, 2):
            return None
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts
        
    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect
    
    def _calibrate_empty_board(self, warped: np.ndarray):
        """Calibrate baseline brightness for empty squares."""
        h, w = warped.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        light_cells = []
        dark_cells = []
        
        for row in range(8):
            for col in range(8):
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                cell = gray[y1:y2, x1:x2]
                margin = cell_h // 4
                center = cell[margin:-margin, margin:-margin] if margin > 0 else cell
                
                if center.size == 0:
                    continue
                    
                is_dark_square = (row + col) % 2 == 1
                cell_mean = np.mean(center)
                
                if is_dark_square:
                    dark_cells.append(cell_mean)
                else:
                    light_cells.append(cell_mean)
        
        self.empty_cells_baseline = {
            'light': np.median(light_cells) if light_cells else 180,
            'dark': np.median(dark_cells) if dark_cells else 100
        }
        self.calibrated = True
    
    def _detect_pieces(self, warped: np.ndarray) -> np.ndarray:
        """Analyze warped board to detect BLACK pieces only (optimized for contrast)."""
        h, w = warped.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        
        grid = np.zeros((8, 8), dtype=np.int8)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Auto-calibrate on first few frames
        if not self.calibrated and len(self.detection_history) < 3:
            self._calibrate_empty_board(warped)
        
        # Calculate overall board brightness
        board_mean = float(np.mean(gray))
        board_std = float(np.std(gray))
        
        # Adaptive threshold based on sensitivity (adjustable in real-time)
        # sensitivity: 0.0 = very strict, 1.0 = very permissive
        base_threshold = self.dark_threshold
        adaptive_factor = 1.0 + (self.sensitivity - 0.5) * 0.8  # 0.6 to 1.4 range
        threshold = int(max(10, min(200, base_threshold * adaptive_factor)))
        
        # Ratio threshold also depends on sensitivity
        min_dark_ratio = 0.25 - (self.sensitivity - 0.5) * 0.15  # 0.175 to 0.325
        min_dark_ratio = max(0.1, min(0.4, min_dark_ratio))

        # Baseline expectations (used for adaptive detection)
        if self.empty_cells_baseline:
            baseline_light = float(self.empty_cells_baseline.get('light', 180))
            baseline_dark = float(self.empty_cells_baseline.get('dark', 100))
            baseline_mean = (baseline_light + baseline_dark) / 2.0
            brightness_shift = board_mean - baseline_mean
        else:
            baseline_light = board_mean
            baseline_dark = board_mean
            brightness_shift = 0.0
        
        # Store for debug
        self.warped_debug = warped.copy()
        
        for row in range(8):
            for col in range(8):
                # Extract cell
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                cell = gray[y1:y2, x1:x2]
                
                # Check center region (where piece would be) - OPTIMIZED MARGIN
                margin = cell_h // 3
                center = cell[margin:-margin, margin:-margin] if margin > 0 else cell
                
                if center.size == 0:
                    continue
                
                # Get cell statistics
                cell_mean = float(np.mean(center))

                # Expected brightness for this square (adjusted for lighting shift)
                is_dark_square = (row + col) % 2 == 1
                expected = (baseline_dark if is_dark_square else baseline_light) + brightness_shift
                expected = max(0.0, min(255.0, expected))
                
                # BLACK piece detection: significantly darker than expected
                # Count very dark pixels
                dark_count = np.sum(center < threshold)
                dark_ratio = dark_count / center.size
                
                # Detection logic: Must meet BOTH conditions
                # 1. High percentage of dark pixels
                # 2. Significantly darker than expected for that square
                deviation = max(8.0, board_std * 0.4)
                deviation *= 1.0 + (0.5 - self.sensitivity) * 0.7
                mean_cutoff = expected - deviation
                
                if dark_ratio > min_dark_ratio and cell_mean < mean_cutoff:
                    grid[row, col] = BLACK  # BLACK piece detected
                
                # Debug visualization
                if self.debug_mode:
                    # Draw cell grid
                    cv2.rectangle(self.warped_debug, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    
                    # Draw detected pieces
                    if grid[row, col] == BLACK:
                        cv2.circle(self.warped_debug, (x1 + cell_w//2, y1 + cell_h//2), 
                                 cell_w//4, (0, 0, 0), -1)
                        cv2.circle(self.warped_debug, (x1 + cell_w//2, y1 + cell_h//2), 
                                 cell_w//4, (255, 255, 0), 2)  # Yellow border
                        cv2.putText(self.warped_debug, "B", (x1+5, y1+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    
                    # Show cell brightness value and dark ratio
                    cv2.putText(self.warped_debug, f"{int(cell_mean)}", (x1+5, y2-15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 0), 1)
                    cv2.putText(self.warped_debug, f"{dark_ratio:.2f}", (x1+5, y2-5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 200, 255), 1)
        
        return grid
        
    def process(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], np.ndarray]:
        """
        Process frame for chessboard detection using contours.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        manual_corners = self._get_manual_corners_abs(frame.shape)
        best_board = None
        
        if manual_corners is None:
            # Strategy 1: Adaptive Threshold (good for uneven lighting)
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(thresh, 30, 100)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Helper to check if contour is board
            def is_board_contour(c):
                # Area check
                if cv2.contourArea(c) < (w * h) * 0.10:
                    return False, None
                
                # Shape check
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                
                if 4 <= len(approx) <= 6:
                    rect = cv2.minAreaRect(c)
                    width = rect[1][0]
                    height = rect[1][1]
                    if width > 0 and height > 0:
                        aspect = max(width, height) / min(width, height)
                        if aspect < 1.8:
                            return True, (c, rect, approx)
                return False, None

            # Try finding board with Strategy 1
            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                valid, data = is_board_contour(contour)
                if valid:
                    best_board = data
                    break
            
            # Strategy 2: Global Threshold (Otsu) - Fallback if Strategy 1 failed
            if not best_board:
                _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                edges_otsu = cv2.Canny(thresh_otsu, 30, 100)
                edges_otsu = cv2.dilate(edges_otsu, kernel, iterations=2)
                contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in sorted(contours_otsu, key=cv2.contourArea, reverse=True):
                    valid, data = is_board_contour(contour)
                    if valid:
                        best_board = data
                        break

        rotation = None
        piece_grid = None
        board_found = False
        manual_used = False
        ordered = None
        angle = None
        
        if manual_corners is not None:
            ordered = manual_corners.astype(np.float32)
            board_found = True
            manual_used = True
            self.board_rect = None
            self.corners = ordered
            top_vec = ordered[1] - ordered[0]
            angle = float(np.degrees(np.arctan2(top_vec[1], top_vec[0])))
        elif best_board:
            contour, rect, approx = best_board
            
            # Get box points
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            board_found = True
            self.board_rect = rect
            
            # Order the points
            ordered = self._order_points(box.astype(np.float32))
            self.corners = ordered
            
            width = rect[1][0]
            height = rect[1][1]
            
            # Calculate rotation angle
            angle = rect[2]
            if width < height:
                angle = angle + 90
        
        if ordered is not None:
            # Draw board outline with colored edges
            pts = ordered.astype(int)
            # TOP edge - RED (steps 1-8)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 4)
            # RIGHT edge - GREEN  
            cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 3)
            # BOTTOM edge - BLUE (steps 9-16)
            cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 4)
            # LEFT edge - GREEN
            cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 3)
            
            if angle is None:
                angle = 0.0
            rotation = max(-1, min(1, angle / 45))
            self.rotation_angle = rotation
            
            # Draw center
            center = np.mean(ordered, axis=0)
            cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 255), -1)
            
            # Perspective transform
            try:
                dst_size = 400
                dst = np.array([
                    [0, 0],
                    [dst_size - 1, 0],
                    [dst_size - 1, dst_size - 1],
                    [0, dst_size - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(ordered, dst)
                warped = cv2.warpPerspective(frame, M, (dst_size, dst_size))
                
                # Detect pieces (raw detection)
                raw_grid = self._detect_pieces(warped)
                
                # Temporal filtering: add to history
                self.detection_history.append(raw_grid.copy())
                if len(self.detection_history) > self.history_size:
                    self.detection_history.pop(0)
                
                # Calculate stable grid (detected in 5+ of last 7 frames)
                min_detections = 5  # Increased from 3
                if len(self.detection_history) >= min_detections:
                    # For each cell, use the most common value across history
                    self.stable_grid = np.zeros((8, 8), dtype=np.int8)
                    for r in range(8):
                        for c in range(8):
                            # Get history of this cell
                            cell_history = [h[r, c] for h in self.detection_history]
                            # Count occurrences: 0=empty, 1=black, 2=white
                            counts = [cell_history.count(v) for v in [0, 1, 2]]
                            # Use value that appears most AND meets threshold
                            max_count = max(counts)
                            if max_count >= min_detections:
                                self.stable_grid[r, c] = counts.index(max_count)
                            # Otherwise keep as empty (0)
                else:
                    self.stable_grid = raw_grid
                
                piece_grid = self.stable_grid
                self.piece_grid = piece_grid
                
                black_count = np.sum(piece_grid == 1)
                _put_text_right(frame, f"Pieces: {black_count}", 200, (255, 255, 255), scale=0.7, thickness=2)
            except:
                pass
            
            label = "Board detected (manual)" if manual_used else "Board detected"
            _put_text_right(frame, label, 140, (0, 255, 0), scale=0.8, thickness=2)
            _put_text_right(frame, f"Rotation: {angle:.1f}deg", 170, (255, 200, 100), scale=0.8, thickness=2)
        
        if not board_found:
            _put_text_right(frame, "Board not detected", 140, (100, 100, 255), scale=0.8, thickness=2)
        
        return rotation, piece_grid, frame


class CameraController:
    """
    Main camera controller that combines hand and board detection.
    """
    
    def __init__(self, device_id: int = 0, config: dict = None):
        """
        Args:
            device_id: Camera device ID
            config: Camera configuration dict
        """
        self.device_id = device_id
        self.config = config or {}
        
        # Camera settings
        self.show_feed = self.config.get('show_feed', True)
        self.board_detection = self.config.get('board_detection', True)
        self.hand_detection = self.config.get('hand_detection', True)
        self.debug_mode = self.config.get('debug_mode', False)
        
        # Image adjustments (can be changed at runtime)
        self.brightness = int(self.config.get('brightness', 0))  # -100 to +100
        self.contrast = float(self.config.get('contrast', 1.0))  # 0.5 to 2.0
        self.brightness = max(-100, min(100, self.brightness))
        self.contrast = max(0.1, min(3.0, self.contrast))
        
        # BPM settings
        bpm_min_distance = self.config.get('bpm_min_distance', 50)
        bpm_max_distance = self.config.get('bpm_max_distance', 400)
        
        # Initialize detectors
        if self.hand_detection:
            self.hand_detector = HandDetector(
                min_distance=bpm_min_distance,
                max_distance=bpm_max_distance
            )
        else:
            self.hand_detector = None
            
        if self.board_detection:
            # Get detection parameters from config
            sensitivity = self.config.get('detection_sensitivity', 0.5)
            dark_threshold = self.config.get('dark_threshold', 50)
            self.board_detector = BoardDetector(
                debug_mode=self.debug_mode,
                sensitivity=sensitivity,
                dark_threshold=dark_threshold
            )
            manual_corners = self.config.get('manual_corners')
            if manual_corners:
                self.board_detector.set_manual_corners_norm(manual_corners)
        else:
            self.board_detector = None
        
        # Camera capture
        self.cap = None
        self.running = False
        self.thread = None
        
        # Callbacks
        self.on_bpm_change = None  # Callback when BPM changes
        self.on_rotation_change = None  # Callback when rotation changes
        self.on_pieces_change = None  # Callback when pieces change
        
        # State
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
    def start(self):
        """Start camera capture in a separate thread."""
        if self.running:
            return
        
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            print(f"✗ Could not open camera {self.device_id}")
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"✓ Camera started (device {self.device_id})")
        return True
    
    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        if self.hand_detector:
            self.hand_detector.close()
        print("✓ Camera stopped")
    
    def _adjust_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
    
    def _capture_loop(self):
        """Main capture loop running in thread."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply brightness/contrast adjustments
            frame = self._adjust_image(frame)
            
            # Process hands
            if self.hand_detector:
                bpm, frame = self.hand_detector.process(frame)
                if bpm is not None and self.on_bpm_change:
                    try:
                        self.on_bpm_change(bpm)
                    except Exception as e:
                        print(f"BPM callback error: {e}")
            
            # Process board
            if self.board_detector:
                rotation, pieces, frame = self.board_detector.process(frame)
                if rotation is not None and self.on_rotation_change:
                    self.on_rotation_change(rotation)
                if pieces is not None and self.on_pieces_change:
                    self.on_pieces_change(pieces)
            
            # Store frame for display
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Note: cv2.imshow doesn't work well from threads on macOS
            # The main thread should call get_frame() and display it
            
            time.sleep(0.016)  # ~60 FPS
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame (thread-safe)."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_debug_warped(self) -> Optional[np.ndarray]:
        """Get the debug warped board view (thread-safe)."""
        if self.board_detector and self.board_detector.warped_debug is not None:
            return self.board_detector.warped_debug.copy()
        return None
