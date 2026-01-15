"""
Camera module for ChessDrum.
Detects chessboard (pieces + rotation) and hands (BPM control).
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import threading
import time


class HandDetector:
    """
    Detects hands using MediaPipe.
    Distance between two hands controls BPM.
    """
    
    def __init__(self, min_bpm: int = 60, max_bpm: int = 200,
                 min_distance: int = 50, max_distance: int = 400):
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
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # State
        self.last_bpm: Optional[int] = None
        self.hands_detected = False
    
    def process(self, frame: np.ndarray) -> Tuple[Optional[int], np.ndarray]:
        """
        Process frame for hand detection.
        
        Returns:
            (bpm, annotated_frame) - bpm is None if not enough hands
        """
        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        
        bpm = None
        self.hands_detected = False
        
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
            self.hands_detected = True
            
            # Get center of each hand
            centers = []
            for hand_landmarks in results.multi_hand_landmarks[:2]:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Calculate center (average of all landmarks)
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
                centers.append(center)
                
                # Draw center point
                cv2.circle(frame, center, 10, (0, 255, 0), -1)
            
            # Calculate distance between hand centers
            distance = np.sqrt(
                (centers[0][0] - centers[1][0])**2 + 
                (centers[0][1] - centers[1][1])**2
            )
            
            # Draw line between hands
            cv2.line(frame, centers[0], centers[1], (0, 255, 255), 3)
            
            # Map distance to BPM
            t = (distance - self.min_distance) / (self.max_distance - self.min_distance)
            t = max(0, min(1, t))
            bpm = int(self.min_bpm + t * (self.max_bpm - self.min_bpm))
            self.last_bpm = bpm
            
            # Display BPM on frame
            cv2.putText(frame, f"BPM: {bpm}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f"Distance: {int(distance)}px", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        elif results.multi_hand_landmarks:
            # Only one hand - draw it but don't change BPM
            self.mp_draw.draw_landmarks(
                frame, results.multi_hand_landmarks[0], 
                self.mp_hands.HAND_CONNECTIONS)
            cv2.putText(frame, "Show 2 hands for BPM", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        return bpm, frame
    
    def close(self):
        self.hands.close()


class BoardDetector:
    """
    Detects chessboard using contour detection.
    Finds the largest rectangular shape and analyzes cells for pieces.
    """
    
    def __init__(self):
        self.last_corners = None
        self.rotation_angle = 0.0
        self.board_rect = None
        self.corners = None
        self.piece_grid = np.zeros((8, 8), dtype=np.int8)
        
        # Temporal filtering: track detection history
        self.detection_history = []  # List of last N detection grids
        self.history_size = 5  # Require detection in 3/5 frames
        self.stable_grid = np.zeros((8, 8), dtype=np.int8)  # Stable output
        
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
    
    def _detect_pieces(self, warped: np.ndarray) -> np.ndarray:
        """Analyze warped board to detect black pieces only."""
        h, w = warped.shape[:2]
        cell_h = h // 8
        cell_w = w // 8
        
        grid = np.zeros((8, 8), dtype=np.int8)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Calculate overall board brightness to adapt threshold
        board_mean = np.mean(gray)
        
        for row in range(8):
            for col in range(8):
                # Extract cell
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                cell = gray[y1:y2, x1:x2]
                
                # Check center region (where piece would be)
                margin = cell_h // 4
                center = cell[margin:-margin, margin:-margin] if margin > 0 else cell
                
                if center.size == 0:
                    continue
                
                # Dark square or light square? (chessboard pattern)
                is_dark_square = (row + col) % 2 == 1
                
                # Get cell statistics  
                cell_mean = np.mean(center)
                
                # Very strict threshold for black pieces
                # A black piece should be MUCH darker than even dark squares
                dark_threshold = 40  # Very dark
                dark_count = np.sum(center < dark_threshold)
                total_pixels = center.size
                dark_ratio = dark_count / total_pixels if total_pixels > 0 else 0
                
                # Also check if center is significantly darker than expected
                # Dark squares ~80-120, Light squares ~160-220, Black pieces <50
                if dark_ratio > 0.25 and cell_mean < 55:
                    grid[row, col] = 1  # Black piece detected
        
        return grid
        
    def process(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], np.ndarray]:
        """
        Process frame for chessboard detection using contours.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
        best_board = None
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            valid, data = is_board_contour(contour)
            if valid:
                best_board = data
                break
        
        # Strategy 2: Global Threshold (Otsu) - Fallback if Strategy 1 failed
        if not best_board:
            _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Invert because board border might be dark
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
        
        if best_board:
            contour, rect, approx = best_board
            
            # Get box points
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            board_found = True
            self.board_rect = rect
            
            # Order the points
            ordered = self._order_points(box.astype(np.float32))
            self.corners = ordered
            
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
            
            width = rect[1][0]
            height = rect[1][1]
            
            # Calculate rotation angle
            angle = rect[2]
            if width < height:
                angle = angle + 90
            
            rotation = max(-1, min(1, angle / 45))
            self.rotation_angle = rotation
            
            # Draw center
            center = (int(rect[0][0]), int(rect[0][1]))
            cv2.circle(frame, center, 10, (0, 255, 255), -1)
            
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
                
                # Calculate stable grid (detected in 3+ of last 5 frames)
                if len(self.detection_history) >= 3:
                    stacked = np.stack(self.detection_history, axis=0)
                    counts = np.sum(stacked, axis=0)
                    self.stable_grid = (counts >= 3).astype(np.int8)
                else:
                    self.stable_grid = raw_grid
                
                piece_grid = self.stable_grid
                self.piece_grid = piece_grid
                
                piece_count = np.sum(piece_grid == 1)
                cv2.putText(frame, f"Pieces: {piece_count}", (10, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except:
                pass
            
            cv2.putText(frame, f"Board detected", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Rotation: {angle:.1f}deg", (10, 170),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)
        
        if not board_found:
            cv2.putText(frame, "Board not detected", (10, 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
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
        
        # Image adjustments (can be changed at runtime)
        self.brightness = 0  # -100 to +100
        self.contrast = 1.0  # 0.5 to 2.0
        
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
            self.board_detector = BoardDetector()
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
        # Convert to float
        adjusted = frame.astype(np.float32)
        
        # Apply contrast (multiply)
        adjusted = adjusted * self.contrast
        
        # Apply brightness (add)
        adjusted = adjusted + self.brightness
        
        # Clip to valid range and convert back
        return np.clip(adjusted, 0, 255).astype(np.uint8)
    
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
