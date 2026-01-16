"""
Camera controller with producer-consumer pattern for ChessDrum.
Separates frame capture from detection processing for optimal performance.
"""
import cv2
import numpy as np
from typing import Optional
import threading
import time
import queue
from logger import get_logger

from .hand_detector import HandDetector
from .board_detector import BoardDetector

logger = get_logger(__name__)


class CameraController:
    """
    Main camera controller with producer-consumer pattern.
    
    Architecture (Phase 1):
    - Producer thread: Captures frames at max FPS (60+ FPS)
    - Consumer thread: Processes detection (may be slower, ~15-30 FPS)
    - Main thread: Displays results without blocking
    
    Benefits:
    - Frame capture never blocks on slow detection
    - Detection can take its time without dropping frames
    - Clean separation of concerns
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
        self.hand_bpm_enabled = self.config.get('hand_bpm_enabled', True)
        self.debug_mode = self.config.get('debug_mode', False)
        
        # Image adjustments (can be changed at runtime)
        self.brightness = int(self.config.get('brightness', 0))
        self.contrast = float(self.config.get('contrast', 1.0))
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
            sensitivity = self.config.get('detection_sensitivity', 0.5)
            dark_threshold = self.config.get('dark_threshold', 50)
            self.board_detector = BoardDetector(
                debug_mode=self.debug_mode,
                sensitivity=sensitivity,
                dark_threshold=dark_threshold
            )
            
            # PHASE 1: Load calibration if available
            calibration = self.config.get('calibration')
            if calibration:
                self.board_detector.load_calibration_data(calibration)
            
            manual_corners = self.config.get('manual_corners')
            if manual_corners:
                self.board_detector.set_manual_corners_norm(manual_corners)
        else:
            self.board_detector = None
        
        # PHASE 1: Producer-Consumer queues
        self.frame_queue = queue.Queue(maxsize=2)  # Raw frames from camera
        self.result_queue = queue.Queue(maxsize=1)  # Processed results
        
        # Camera capture
        self.cap = None
        self.running = False
        self.capture_thread = None
        self.detection_thread = None
        
        # Callbacks
        self.on_bpm_change = None
        self.on_rotation_change = None
        self.on_pieces_change = None
        
        # State (shared)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Performance metrics
        self.capture_fps = 0.0
        self.detection_fps = 0.0
        self._last_capture_time = time.time()
        self._last_detection_time = time.time()
        self._capture_frame_count = 0
        self._detection_frame_count = 0
        
        logger.info(f"CameraController initialized (producer-consumer mode)")
        
    def start(self):
        """Start camera capture and detection threads."""
        if self.running:
            logger.warning("Camera already running")
            return True
        
        logger.info(f"Opening camera device {self.device_id}...")
        self.cap = cv2.VideoCapture(self.device_id)
        if not self.cap.isOpened():
            logger.error(f"Could not open camera device {self.device_id}")
            return False
        
        self.running = True
        
        # Start producer thread (capture)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="CaptureThread")
        self.capture_thread.start()
        
        # Start consumer thread (detection)
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True, name="DetectionThread")
        self.detection_thread.start()
        
        logger.info(f"Camera threads started (device {self.device_id})")
        return True
    
    def stop(self):
        """Stop camera capture and detection."""
        if not self.running:
            return
            
        logger.info("Stopping camera...")
        self.running = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
            if self.capture_thread.is_alive():
                logger.warning("Capture thread did not stop cleanly")
                
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)
            if self.detection_thread.is_alive():
                logger.warning("Detection thread did not stop cleanly")
        
        # Release resources
        if self.cap:
            self.cap.release()
        if self.hand_detector:
            self.hand_detector.close()
            
        # PHASE 1: Save calibration if stable
        if self.board_detector:
            cal_data = self.board_detector.get_calibration_data()
            if cal_data:
                logger.info(f"Calibration data available for saving: {cal_data}")
                # This would be saved to config.json by the caller
        
        logger.info(f"Camera stopped (capture: {self.capture_fps:.1f} FPS, detection: {self.detection_fps:.1f} FPS)")
    
    def _adjust_image(self, frame: np.ndarray) -> np.ndarray:
        """Apply brightness and contrast adjustments."""
        return cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
    
    def _capture_loop(self):
        """
        Producer thread: Capture frames at maximum FPS.
        This runs fast and never blocks on detection.
        """
        logger.info("Capture loop started")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.001)
                continue
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Apply brightness/contrast adjustments
            frame = self._adjust_image(frame)
            
            # Put frame in queue (non-blocking, drop old frames if full)
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Queue full, skip this frame (detection is slower than capture)
                try:
                    # Remove oldest frame and add new one
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
            
            # Update FPS metrics
            self._capture_frame_count += 1
            if self._capture_frame_count % 30 == 0:
                now = time.time()
                elapsed = now - self._last_capture_time
                self.capture_fps = 30.0 / elapsed if elapsed > 0 else 0
                self._last_capture_time = now
                logger.debug(f"Capture FPS: {self.capture_fps:.1f}")
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.001)
        
        logger.info("Capture loop stopped")
    
    def _detection_loop(self):
        """
        Consumer thread: Process frames with hand and board detection.
        This can be slower without affecting capture FPS.
        """
        logger.info("Detection loop started")
        
        while self.running:
            try:
                # Get frame from queue (blocking with timeout)
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            # Process hands
            if self.hand_detector and self.hand_bpm_enabled:
                try:
                    bpm, frame = self.hand_detector.process(frame)
                    if bpm is not None and self.on_bpm_change:
                        self.on_bpm_change(bpm)
                except Exception as e:
                    logger.error(f"Hand detection error: {e}", exc_info=True)
            elif self.hand_detector:
                self.hand_detector.reset_state()
            
            # Process board
            if self.board_detector:
                try:
                    rotation, pieces, frame = self.board_detector.process(frame)
                    
                    if rotation is not None and self.on_rotation_change:
                        self.on_rotation_change(rotation)
                    
                    if pieces is not None and self.on_pieces_change:
                        self.on_pieces_change(pieces)
                except Exception as e:
                    logger.error(f"Board detection error: {e}", exc_info=True)
            
            # Store processed frame for display
            with self.frame_lock:
                self.current_frame = frame.copy()
            
            # Update detection FPS metrics
            self._detection_frame_count += 1
            if self._detection_frame_count % 15 == 0:
                now = time.time()
                elapsed = now - self._last_detection_time
                self.detection_fps = 15.0 / elapsed if elapsed > 0 else 0
                self._last_detection_time = now
                logger.debug(f"Detection FPS: {self.detection_fps:.1f}")
        
        logger.info("Detection loop stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the current processed frame (thread-safe)."""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_debug_warped(self) -> Optional[np.ndarray]:
        """Get the debug warped board view (thread-safe)."""
        if self.board_detector and self.board_detector.warped_debug is not None:
            return self.board_detector.warped_debug.copy()
        return None

    def set_hand_bpm_enabled(self, enabled: bool):
        """Enable/disable hand BPM detection."""
        self.hand_bpm_enabled = bool(enabled)
        if self.hand_detector and not self.hand_bpm_enabled:
            self.hand_detector.reset_state()
    
    def get_performance_stats(self) -> dict:
        """Get performance metrics (Phase 1)."""
        return {
            'capture_fps': self.capture_fps,
            'detection_fps': self.detection_fps,
            'frame_queue_size': self.frame_queue.qsize(),
            'frame_queue_maxsize': self.frame_queue.maxsize
        }
