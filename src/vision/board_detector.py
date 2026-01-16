"""
Board detection module for ChessDrum.
Detects chessboard using contour detection and analyzes cells for BLACK pieces.
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from logger import get_logger

# Import from parent directory
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grid import BLACK

logger = get_logger(__name__)


class BoardDetector:
    """
    Detects chessboard using contour detection.
    Finds the largest rectangular shape and analyzes cells for BLACK pieces only.
    
    Features:
    - Adaptive thresholding for varying lighting
    - Manual corner override
    - Real-time calibration
    - Temporal filtering (5/7 frames consensus)
    - Dynamic board sizing (Phase 1)
    """
    
    def __init__(self, debug_mode=False, sensitivity=0.5, dark_threshold=50):
        """
        Args:
            debug_mode: Enable debug visualization
            sensitivity: Detection sensitivity 0.0-1.0 (higher = more sensitive)
            dark_threshold: Base threshold for dark pixel detection (20-100)
        """
        self.last_corners = None
        self.rotation_angle = 0.0
        self.board_rect = None
        self.corners = None
        self.piece_grid = np.zeros((8, 8), dtype=np.int8)
        self.manual_corners_norm = None  # Normalized 4-point corners (TL, TR, BR, BL)
        self.last_board_found = False
        self.last_manual_used = False
        self.last_rotation_deg = None
        self.last_piece_count = None
        self.last_status_text = None
        
        # Detection parameters (adjustable in real-time)
        self.sensitivity = sensitivity
        self.dark_threshold = dark_threshold
        
        # PHASE 1: Dynamic board sizing
        self.warp_size_mode = "auto"  # "auto" or fixed number
        self.min_warp_size = 200
        self.max_warp_size = 800
        self.last_warp_size = 400  # Default
        
        # Temporal filtering: track detection history
        self.detection_history = []
        self.history_size = 7
        self.stable_grid = np.zeros((8, 8), dtype=np.int8)
        
        # PHASE 1: Adaptive temporal filtering
        self.board_stability_score = 0.0  # 0.0 to 1.0
        self.stability_history = []
        self.adaptive_threshold = 5  # Will adjust dynamically
        
        # Calibration data
        self.empty_cells_baseline = None
        self.calibrated = False
        self.calibration_stable_frames = 0  # Count of stable frames for saving
        
        # Debug
        self.debug_mode = debug_mode
        self.warped_debug = None
        
        logger.info(f"BoardDetector initialized (sensitivity={sensitivity}, threshold={dark_threshold})")

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
        logger.info("Manual corners set")
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
        logger.info("Manual corners set (normalized)")
        return True

    def clear_manual_corners(self):
        """Disable manual board corners and return to auto detection."""
        self.manual_corners_norm = None
        self.board_rect = None
        self.corners = None
        self.calibrated = False
        self.detection_history.clear()
        logger.info("Manual corners cleared")

    def get_calibration_data(self) -> Optional[dict]:
        """
        PHASE 1: Get calibration data for persistence.
        Returns None if not calibrated or not stable enough.
        """
        if not self.calibrated or self.empty_cells_baseline is None:
            return None
        
        # Only return if stable for enough frames
        if self.calibration_stable_frames < 30:
            return None
        
        # Calculate board_id from corners
        board_id = None
        if self.corners is not None:
            # Simple checksum from corner coordinates
            corners_str = "".join([f"{int(p[0])}{int(p[1])}" for p in self.corners])
            board_id = hash(corners_str) % 100000
        
        return {
            "baseline_light": float(self.empty_cells_baseline['light']),
            "baseline_dark": float(self.empty_cells_baseline['dark']),
            "board_id": board_id,
            "sensitivity": self.sensitivity,
            "dark_threshold": self.dark_threshold
        }

    def load_calibration_data(self, data: dict) -> bool:
        """
        PHASE 1: Load calibration data from persistence.
        Returns True if loaded successfully.
        """
        try:
            self.empty_cells_baseline = {
                'light': data.get('baseline_light', 180),
                'dark': data.get('baseline_dark', 100)
            }
            self.calibrated = True
            self.calibration_stable_frames = 30  # Mark as stable
            logger.info(f"Calibration data loaded: light={self.empty_cells_baseline['light']:.1f}, "
                       f"dark={self.empty_cells_baseline['dark']:.1f}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration data: {e}")
            return False

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
    
    def _calculate_dynamic_warp_size(self, contour_area: float, frame_area: float) -> int:
        """
        PHASE 1: Calculate optimal warp size based on detected board area.
        Larger boards get higher resolution warping.
        """
        if self.warp_size_mode != "auto":
            # Fixed size mode
            if isinstance(self.warp_size_mode, int):
                return np.clip(self.warp_size_mode, self.min_warp_size, self.max_warp_size)
            return 400  # Default fallback
        
        # Auto mode: scale based on board area percentage
        area_ratio = contour_area / frame_area
        # Boards that fill 10-50% of frame get 200-800 warp size
        area_ratio = np.clip(area_ratio, 0.10, 0.50)
        # Linear interpolation
        warp_size = self.min_warp_size + (area_ratio - 0.10) / 0.40 * (self.max_warp_size - self.min_warp_size)
        warp_size = int(warp_size)
        # Round to nearest 50 for cache efficiency
        warp_size = round(warp_size / 50) * 50
        return np.clip(warp_size, self.min_warp_size, self.max_warp_size)
    
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
        logger.debug(f"Board calibrated: light={self.empty_cells_baseline['light']:.1f}, "
                    f"dark={self.empty_cells_baseline['dark']:.1f}")
    
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
        
        # Adaptive threshold based on sensitivity
        base_threshold = self.dark_threshold
        adaptive_factor = 1.0 + (self.sensitivity - 0.5) * 0.8
        threshold = int(max(10, min(200, base_threshold * adaptive_factor)))
        
        # Ratio threshold
        min_dark_ratio = 0.25 - (self.sensitivity - 0.5) * 0.15
        min_dark_ratio = max(0.1, min(0.4, min_dark_ratio))

        # Baseline expectations
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
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                cell = gray[y1:y2, x1:x2]
                margin = cell_h // 3
                center = cell[margin:-margin, margin:-margin] if margin > 0 else cell
                
                if center.size == 0:
                    continue
                
                cell_mean = float(np.mean(center))
                is_dark_square = (row + col) % 2 == 1
                expected = (baseline_dark if is_dark_square else baseline_light) + brightness_shift
                expected = max(0.0, min(255.0, expected))
                
                # BLACK piece detection
                dark_count = np.sum(center < threshold)
                dark_ratio = dark_count / center.size
                
                deviation = max(8.0, board_std * 0.4)
                deviation *= 1.0 + (0.5 - self.sensitivity) * 0.7
                mean_cutoff = expected - deviation
                
                if dark_ratio > min_dark_ratio and cell_mean < mean_cutoff:
                    grid[row, col] = BLACK
                
                # Debug visualization
                if self.debug_mode:
                    cv2.rectangle(self.warped_debug, (x1, y1), (x2, y2), (128, 128, 128), 1)
                    if grid[row, col] == BLACK:
                        cv2.circle(self.warped_debug, (x1 + cell_w//2, y1 + cell_h//2), 
                                 cell_w//4, (0, 0, 0), -1)
                        cv2.circle(self.warped_debug, (x1 + cell_w//2, y1 + cell_h//2), 
                                 cell_w//4, (255, 255, 0), 2)
                    cv2.putText(self.warped_debug, f"{int(cell_mean)}", (x1+5, y2-15),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 0), 1)
        
        return grid
    
    def _update_board_stability(self):
        """
        PHASE 1: Calculate board stability score.
        Used for adaptive temporal filtering.
        """
        if len(self.detection_history) < 3:
            self.board_stability_score = 0.0
            return
        
        # Compare last 3 frames
        recent = self.detection_history[-3:]
        
        # Count differences between consecutive frames
        diff_count = 0
        total_cells = 8 * 8
        
        for i in range(len(recent) - 1):
            diff_count += np.sum(recent[i] != recent[i + 1])
        
        # Stability score: 1.0 = no changes, 0.0 = all cells changing
        avg_diff = diff_count / (len(recent) - 1)
        self.board_stability_score = 1.0 - (avg_diff / total_cells)
        
        # Track stability over time
        self.stability_history.append(self.board_stability_score)
        if len(self.stability_history) > 30:
            self.stability_history.pop(0)
        
        # Update adaptive threshold
        avg_stability = np.mean(self.stability_history) if self.stability_history else 0.5
        
        if avg_stability > 0.80:
            # Very stable: relax to 3/5 frames
            self.adaptive_threshold = 3
        elif avg_stability > 0.50:
            # Moderately stable: 4/7
            self.adaptive_threshold = 4
        else:
            # Unstable: strict 5/7
            self.adaptive_threshold = 5
        
        logger.debug(f"Board stability: {self.board_stability_score:.2f}, "
                    f"adaptive threshold: {self.adaptive_threshold}/{self.history_size}")

    def process(self, frame: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray], np.ndarray]:
        """
        Process frame for chessboard detection.
        
        Returns:
            (rotation, pieces_grid, annotated_frame)
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        manual_corners = self._get_manual_corners_abs(frame.shape)
        best_board = None
        contour_area = 0
        
        if manual_corners is None:
            # Auto detection with adaptive thresholding
            blurred = cv2.GaussianBlur(gray, (7, 7), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            edges = cv2.Canny(thresh, 30, 100)
            kernel = np.ones((5, 5), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            def is_board_contour(c):
                area = cv2.contourArea(c)
                if area < frame_area * 0.10:
                    return False, None, 0
                
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                
                if 4 <= len(approx) <= 6:
                    rect = cv2.minAreaRect(c)
                    width, height = rect[1]
                    if width > 0 and height > 0:
                        aspect = max(width, height) / min(width, height)
                        if aspect < 1.8:
                            return True, (c, rect, approx), area
                return False, None, 0

            for contour in sorted(contours, key=cv2.contourArea, reverse=True):
                valid, data, area = is_board_contour(contour)
                if valid:
                    best_board = data
                    contour_area = area
                    break
            
            # Fallback: Otsu threshold
            if not best_board:
                _, thresh_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                edges_otsu = cv2.Canny(thresh_otsu, 30, 100)
                edges_otsu = cv2.dilate(edges_otsu, kernel, iterations=2)
                contours_otsu, _ = cv2.findContours(edges_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in sorted(contours_otsu, key=cv2.contourArea, reverse=True):
                    valid, data, area = is_board_contour(contour)
                    if valid:
                        best_board = data
                        contour_area = area
                        break

        rotation = None
        piece_grid = None
        board_found = False
        manual_used = False
        ordered = None
        angle = None
        piece_count = None
        
        if manual_corners is not None:
            ordered = manual_corners.astype(np.float32)
            board_found = True
            manual_used = True
            self.board_rect = None
            self.corners = ordered
            top_vec = ordered[1] - ordered[0]
            angle = float(np.degrees(np.arctan2(top_vec[1], top_vec[0])))
            contour_area = frame_area * 0.25  # Assume 25% for manual
        elif best_board:
            contour, rect, approx = best_board
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            
            board_found = True
            self.board_rect = rect
            ordered = self._order_points(box.astype(np.float32))
            self.corners = ordered
            
            angle = rect[2]
            width, height = rect[1]
            if width < height:
                angle = angle + 90
        
        if ordered is not None:
            # Draw board outline
            pts = ordered.astype(int)
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), (0, 0, 255), 4)
            cv2.line(frame, tuple(pts[1]), tuple(pts[2]), (0, 255, 0), 3)
            cv2.line(frame, tuple(pts[2]), tuple(pts[3]), (255, 0, 0), 4)
            cv2.line(frame, tuple(pts[3]), tuple(pts[0]), (0, 255, 0), 3)
            
            if angle is None:
                angle = 0.0
            rotation = max(-1, min(1, angle / 45))
            self.rotation_angle = rotation
            
            center = np.mean(ordered, axis=0)
            cv2.circle(frame, (int(center[0]), int(center[1])), 10, (0, 255, 255), -1)
            
            # PHASE 1: Dynamic warp sizing
            dst_size = self._calculate_dynamic_warp_size(contour_area, frame_area)
            self.last_warp_size = dst_size
            logger.debug(f"Warp size: {dst_size}x{dst_size} (area ratio: {contour_area/frame_area:.2%})")
            
            try:
                dst = np.array([
                    [0, 0],
                    [dst_size - 1, 0],
                    [dst_size - 1, dst_size - 1],
                    [0, dst_size - 1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(ordered, dst)
                warped = cv2.warpPerspective(frame, M, (dst_size, dst_size))
                
                raw_grid = self._detect_pieces(warped)
                
                # Temporal filtering
                self.detection_history.append(raw_grid.copy())
                if len(self.detection_history) > self.history_size:
                    self.detection_history.pop(0)
                
                # PHASE 1: Adaptive temporal filtering
                self._update_board_stability()
                min_detections = self.adaptive_threshold
                
                if len(self.detection_history) >= min_detections:
                    self.stable_grid = np.zeros((8, 8), dtype=np.int8)
                    for r in range(8):
                        for c in range(8):
                            cell_history = [h[r, c] for h in self.detection_history]
                            counts = [cell_history.count(v) for v in [0, 1, 2]]
                            max_count = max(counts)
                            if max_count >= min_detections:
                                self.stable_grid[r, c] = counts.index(max_count)
                else:
                    self.stable_grid = raw_grid
                
                piece_grid = self.stable_grid
                self.piece_grid = piece_grid
                piece_count = int(np.sum(piece_grid == BLACK))
                
                # Track calibration stability
                if self.calibrated:
                    self.calibration_stable_frames += 1
                else:
                    self.calibration_stable_frames = 0
                    
            except Exception as e:
                logger.error(f"Board processing error: {e}", exc_info=True)
                piece_grid = None

        if board_found:
            self.last_board_found = True
            self.last_manual_used = manual_used
            self.last_rotation_deg = float(angle) if angle is not None else None
            self.last_piece_count = piece_count
            label = "Board detected (manual)" if manual_used else "Board detected"
            self.last_status_text = label
        else:
            self.last_board_found = False
            self.last_manual_used = False
            self.last_rotation_deg = None
            self.last_piece_count = None
            self.last_status_text = "Board not detected"
            self.calibration_stable_frames = 0
        
        return rotation, piece_grid, frame
