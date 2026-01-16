"""
Pygame-based UI for ChessDrum.
Camera-first layout with small grid overlay.
"""
import pygame
try:
    import cv2
except ImportError:
    cv2 = None
from typing import Optional

from grid import Grid, EMPTY, WHITE, BLACK, INSTRUMENTS
from libraries import load_pattern_libraries
from sequencer import Sequencer

# Colors
COLOR_BG = (30, 30, 35)
COLOR_BOARD_LIGHT = (139, 195, 74)   # Green like a game board
COLOR_BOARD_DARK = (104, 159, 56)
COLOR_PIECE_WHITE = (255, 255, 255)
COLOR_PIECE_BLACK = (40, 40, 40)
COLOR_PIECE_BORDER = (80, 80, 80)
COLOR_PLAYHEAD = (255, 193, 7)       # Amber
COLOR_TEXT = (255, 255, 255)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_ACTIVE = (65, 105, 225)
COLOR_SLIDER_BG = (60, 60, 65)
COLOR_SLIDER_FILL = (70, 130, 180)

# Layout constants - camera-first design
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700
CONTROLS_HEIGHT = 120
CAMERA_HEIGHT = WINDOW_HEIGHT - CONTROLS_HEIGHT

# FASE 4: Mixer panel (lateral derecho)
MIXER_WIDTH = 200
MIXER_ENABLED_WIDTH = WINDOW_WIDTH + MIXER_WIDTH  # 1000px total when mixer open

# Mini grid settings (bottom right corner)
MINI_CELL = 18
MINI_MARGIN = 10
MINI_SIZE = MINI_CELL * 8
MINI_LABEL_WIDTH = 30

# FASE 2: Large grid settings (for virtual/no-camera mode)
LARGE_CELL = 60
LARGE_MARGIN = 40
LARGE_SIZE = LARGE_CELL * 8
LARGE_LABEL_WIDTH = 50

# Instrument labels
INSTRUMENT_LABELS = ['HH', 'CP', 'SD', 'KK']

# FASE 4: Channel labels con números
CHANNEL_NAMES = ["1.HH", "2.CP", "3.SD", "4.KK"]


class UI:
    """Pygame-based UI for the drum sequencer - camera-first layout."""
    
    def __init__(self, grid: Grid, sequencer: Sequencer, audio_output=None, config=None, camera=None):
        self.grid = grid
        self.sequencer = sequencer
        self.audio_output = audio_output
        self.config = config
        self.camera = camera
        
        # Get title from config
        if config:
            window_title = config.get('ui', 'window_title', default='ChessDrum 🎵')
        else:
            window_title = 'ChessDrum 🎵'
        
        # Pygame setup
        pygame.init()
        pygame.display.set_caption(window_title)
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 20)
        
        # UI state
        self.running = True
        self._highlighted_step = 0
        self._dragging_bpm_slider = False
        self._dragging_cutoff_slider = False
        self._dragging_resonance_slider = False
        self._manual_board_mode = False
        self._manual_points = []
        self._last_frame_shape = None
        self._library_browser_open = False
        self._library_mode = "sound"
        self._library_index = 0
        self._pattern_index = 0
        self._large_grid_bounds = None  # FASE 2: For virtual mode click detection
        
        # FASE 2: Notification system for errors
        self._notifications = []  # List of (message, color, timestamp)
        self._notification_duration = 3.0  # seconds
        
        # FASE 4: Mixer panel state
        self._mixer_open = False
        self._dragging_channel_fader = None  # (channel_idx, start_y)
        self._dragging_delay_knob = None     # channel_idx
        self._dragging_reverb_knob = None    # channel_idx
        self._pattern_index = 0
        self._large_grid_bounds = None  # FASE 2: For virtual mode click detection
        
        # FASE 2: Notification system for errors
        self._notifications = []  # List of (message, color, timestamp)
        self._notification_duration = 3.0  # seconds

        # Load pattern libraries
        pattern_path = None
        if config:
            pattern_path = config.get('libraries', 'pattern_file', default=None)
        self.pattern_libraries = load_pattern_libraries(pattern_path)
        self.active_pattern_libraries = [lib for lib in self.pattern_libraries if lib.get("active", True)]
        
        # Register callback for step changes
        self.sequencer.on_step_change = self._on_step_change
    
    def _on_step_change(self, step: int):
        """Called by sequencer when step changes."""
        self._highlighted_step = step
    
    def show_notification(self, message: str, color=(255, 100, 100)):
        """
        FASE 2: Show a temporary notification message.
        Args:
            message: Text to display
            color: RGB color (default red for errors)
        """
        import time
        self._notifications.append((message, color, time.time()))
        # Keep only last 3 notifications
        if len(self._notifications) > 3:
            self._notifications.pop(0)

    def _draw_board_corners_indicator(self):
        """
        FASE 2: Draw visual indicator of board corners detection.
        Shows where the board is detected in the camera feed.
        """
        if not self.camera or not hasattr(self.camera, 'board_detector'):
            return
        
        board_det = self.camera.board_detector
        
        # Check if corners are detected
        if not hasattr(board_det, 'corners_norm') or board_det.corners_norm is None:
            return
        
        try:
            # Get current frame to determine size
            frame = self.camera.get_frame()
            if frame is None:
                return
            
            frame_h, frame_w = frame.shape[:2]
            
            # Convert normalized corners to screen coordinates
            corners_norm = board_det.corners_norm
            corners_screen = []
            
            for corner_norm in corners_norm:
                # Denormalize from [0,1] range
                x_norm, y_norm = corner_norm
                
                # Scale to frame size
                x_frame = x_norm * frame_w
                y_frame = y_norm * frame_h
                
                # Scale to screen size (might be different aspect ratio)
                if self._last_frame_shape:
                    last_h, last_w = self._last_frame_shape[:2]
                    scale_x = WINDOW_WIDTH / last_w
                    scale_y = CAMERA_HEIGHT / last_h
                    x_screen = int(x_frame * scale_x)
                    y_screen = int(y_frame * scale_y)
                else:
                    x_screen = int(x_frame)
                    y_screen = int(y_frame)
                
                corners_screen.append((x_screen, y_screen))
            
            if len(corners_screen) == 4:
                # Draw filled polygon with transparency
                poly_color = (50, 255, 50, 60)
                poly_surf = pygame.Surface((WINDOW_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(poly_surf, poly_color, corners_screen)
                self.screen.blit(poly_surf, (0, 0))
                
                # Draw corner points
                for i, corner in enumerate(corners_screen):
                    pygame.draw.circle(self.screen, (0, 255, 0), corner, 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), corner, 6, 2)
                    
                    # Label corners: TL, TR, BR, BL
                    labels = ["TL", "TR", "BR", "BL"]
                    label_text = self.font_small.render(labels[i], True, (255, 255, 255))
                    self.screen.blit(label_text, (corner[0] + 10, corner[1] + 10))
                
                # Draw border lines
                for i in range(4):
                    p1 = corners_screen[i]
                    p2 = corners_screen[(i + 1) % 4]
                    pygame.draw.line(self.screen, (100, 255, 100), p1, p2, 2)
        except:
            pass

    def _draw_camera_overlay(self):
        """Draw camera settings and detection status."""
        if not self.camera:
            return
        settings_font = self.font_small
        line_height = settings_font.get_height() + 4
        gap_height = 6
        padding = 10
        margin = 10

        sections = []

        # Camera image controls
        sections.append([
            (f"Brightness (Q/A): {self.camera.brightness}", (255, 255, 255)),
            (f"Contrast (W/S): {self.camera.contrast:.1f}", (255, 255, 255)),
        ])

        board_detector = getattr(self.camera, 'board_detector', None)
        if board_detector:
            detect_lines = []
            sens = board_detector.sensitivity
            sens_color = (100, 255, 100) if 0.4 <= sens <= 0.6 else (255, 200, 100)
            detect_lines.append((f"Sensitivity (1-9): {sens:.1f}", sens_color))
            detect_lines.append((f"Dark Thresh (T/G): {board_detector.dark_threshold}", (255, 255, 100)))
            sections.append(detect_lines)

            status_text = board_detector.last_status_text
            if status_text is None:
                status_text = "Board detected" if board_detector.corners is not None else "Board not detected"
            status_color = (0, 255, 0) if (board_detector.last_board_found or board_detector.corners is not None) else (100, 100, 255)

            board_lines = [(status_text, status_color)]
            if board_detector.last_piece_count is not None:
                board_lines.append((f"Pieces: {board_detector.last_piece_count}", (255, 255, 255)))

            if board_detector.calibrated:
                board_lines.append(("Calibrated (R to reset)", (0, 255, 0)))
            else:
                board_lines.append(("Calibrating... (R to reset)", (255, 200, 0)))

            if board_detector.debug_mode:
                board_lines.append(("Debug: ON (D to toggle)", (255, 255, 0)))

            manual_corners = getattr(board_detector, 'manual_corners_norm', None)
            if self._manual_board_mode:
                step = min(len(self._manual_points) + 1, 4)
                board_lines.append((f"Manual board: click {step}/4 (TL,TR,BR,BL)", (255, 255, 0)))
            elif manual_corners is not None:
                board_lines.append(("Manual board: ON (Backspace to clear)", (255, 255, 0)))

            sections.append(board_lines)

        hand_detector = getattr(self.camera, 'hand_detector', None)
        if hand_detector:
            status = hand_detector.status
            if status == "active":
                status_label = "Active"
                status_color = (0, 255, 0)
            elif status == "hold":
                status_label = "Hold"
                status_color = (255, 200, 0)
            else:
                status_label = "Idle"
                status_color = (160, 160, 160)

            hand_enabled = getattr(self.camera, 'hand_bpm_enabled', True)
            if not hand_enabled:
                status_label = "Off"
                status_color = (140, 140, 140)

            bpm_lines = [
                (f"BPM: {self.sequencer.bpm}", status_color),
                (f"Hands: {status_label}", status_color),
                (f"Hand BPM: {'ON' if hand_enabled else 'OFF'} (H)", status_color),
            ]
            if hand_detector.last_distance is not None and status == "active":
                bpm_lines.append((f"Hands dist: {int(hand_detector.last_distance)}px", status_color))
            sections.append(bpm_lines)

        if self.audio_output and hasattr(self.audio_output, 'kit_name'):
            volume_pct = None
            if hasattr(self.audio_output, 'master_volume'):
                volume_pct = int(self.audio_output.master_volume * 100)
            kit_label = getattr(self.audio_output, 'kit_display_name', self.audio_output.kit_name)
            kit_lines = [
                (f"Kit (Left/Right): {kit_label}", (200, 220, 255)),
                ("Libraries (L): browse", (200, 220, 255)),
            ]
            if volume_pct is not None:
                kit_lines.append((f"Volume (Up/Down): {volume_pct}%", (200, 220, 255)))
            sections.append(kit_lines)

        sections.append([
            ("RED = TOP (Steps 1-8)", (255, 50, 50)),
            ("BLUE = BOTTOM (Steps 9-16)", (50, 50, 255)),
        ])

        layout = []
        for section in sections:
            if not section:
                continue
            if layout:
                layout.append(None)
            layout.extend(section)

        if not layout:
            return

        widths = []
        for item in layout:
            if item is None:
                continue
            text, _ = item
            widths.append(settings_font.size(text)[0])
        max_width = max(widths) if widths else 0
        gaps = sum(1 for item in layout if item is None)
        panel_height = (line_height * (len(layout) - gaps)) + (gap_height * gaps) + padding * 2
        panel_width = max_width + padding * 2
        panel_x = WINDOW_WIDTH - panel_width - margin
        panel_y = margin

        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        self.screen.blit(panel, (panel_x, panel_y))

        right_edge = panel_x + panel_width - padding
        cursor_y = panel_y + padding
        for item in layout:
            if item is None:
                cursor_y += gap_height
                continue
            text, color = item
            txt = settings_font.render(text, True, color)
            self.screen.blit(txt, (right_edge - txt.get_width(), cursor_y))
            cursor_y += line_height

    def _screen_to_frame(self, pos):
        if not self._last_frame_shape:
            return None
        x, y = pos
        if y < 0 or y >= CAMERA_HEIGHT:
            return None
        frame_h, frame_w = self._last_frame_shape
        scale_x = frame_w / WINDOW_WIDTH
        scale_y = frame_h / CAMERA_HEIGHT
        return (x * scale_x, y * scale_y)

    def _frame_to_screen(self, point):
        if not self._last_frame_shape:
            return None
        frame_h, frame_w = self._last_frame_shape
        scale_x = WINDOW_WIDTH / frame_w
        scale_y = CAMERA_HEIGHT / frame_h
        return (point[0] * scale_x, point[1] * scale_y)

    def _get_manual_points_for_draw(self):
        if self._manual_points:
            return self._manual_points
        if self.camera and hasattr(self.camera, 'board_detector'):
            corners_norm = getattr(self.camera.board_detector, 'manual_corners_norm', None)
            if corners_norm is not None and self._last_frame_shape:
                frame_h, frame_w = self._last_frame_shape
                return [(p[0] * frame_w, p[1] * frame_h) for p in corners_norm]
        return []

    def _draw_manual_board_overlay(self):
        points = self._get_manual_points_for_draw()
        if not points:
            return
        screen_pts = []
        for point in points:
            screen_point = self._frame_to_screen(point)
            if screen_point is not None:
                screen_pts.append(screen_point)
        if not screen_pts:
            return
        screen_pts_int = [(int(p[0]), int(p[1])) for p in screen_pts]
        for pt in screen_pts_int:
            pygame.draw.circle(self.screen, (255, 255, 0), pt, 6)
        if len(screen_pts_int) >= 2:
            closed = len(screen_pts) == 4
            pygame.draw.lines(self.screen, (255, 255, 0), closed, screen_pts_int, 3)
    
    def _draw_camera_feed(self):
        """Draw camera feed as main view."""
        if not self.camera:
            # No camera - show placeholder
            pygame.draw.rect(self.screen, (50, 50, 55), 
                           (0, 0, WINDOW_WIDTH, CAMERA_HEIGHT))
            text = self.font_large.render("No Camera", True, (100, 100, 100))
            x = (WINDOW_WIDTH - text.get_width()) // 2
            y = (CAMERA_HEIGHT - text.get_height()) // 2
            self.screen.blit(text, (x, y))
            
            hint = self.font.render("Run with --camera flag", True, (80, 80, 80))
            self.screen.blit(hint, ((WINDOW_WIDTH - hint.get_width()) // 2, y + 50))
            return
        
        frame = self.camera.get_frame()
        if frame is None:
            pygame.draw.rect(self.screen, (50, 50, 55), 
                           (0, 0, WINDOW_WIDTH, CAMERA_HEIGHT))
            text = self.font.render("Waiting for camera...", True, (100, 100, 100))
            self.screen.blit(text, (20, 20))
            self._draw_camera_overlay()
            return
        if cv2 is None:
            pygame.draw.rect(self.screen, (50, 50, 55), 
                           (0, 0, WINDOW_WIDTH, CAMERA_HEIGHT))
            text = self.font.render("OpenCV not available", True, (100, 100, 100))
            self.screen.blit(text, (20, 20))
            self._draw_camera_overlay()
            return

        self._last_frame_shape = frame.shape[:2]

        # Create surface from frame (RGB)
        # Flip mainly handled in camera capture loop, but Pygame needs it right way up
        # CV2 is BGR, Pygame needs RGB. and Pygame surface is (W,H)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Scale to fit window width? Or keep aspect ratio?
        # Current UI assumes WINDOW_WIDTH x CAMERA_HEIGHT
        frame_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, CAMERA_HEIGHT))
        self.screen.blit(frame_surface, (0, 0))
        self._draw_manual_board_overlay()
        
        # --- DRAW OVERLAY ---
        if self.camera and hasattr(self.camera, 'board_detector') and self.camera.board_detector.corners is not None:
            corners = self.camera.board_detector.corners
            
            # Scale corners to UI space
            # Camera frame size vs UI size
            h, w = frame.shape[:2]
            scale_x = WINDOW_WIDTH / w
            scale_y = CAMERA_HEIGHT / h
            
            # --- DRAW PLAYHEAD OVERLAY (like reference image) ---
            step = self._highlighted_step
            col_idx = step % 8
            # Is it top half (0-7) or bottom half (8-15)?
            # Mapping: Upper half (rows 0-3) = Steps 0-7
            # Mapping: Lower half (rows 4-7) = Steps 8-15
            is_bottom = step >= 8
            
            # Interpolate points for the column
            # Top edge: corners[0] -> corners[1]
            # Bottom edge: corners[3] -> corners[2]
            
            def lerp_point(p1, p2, t):
                return p1 + (p2 - p1) * t
            
            t1 = col_idx / 8.0
            t2 = (col_idx + 1) / 8.0
            
            # Full column strip
            top_start = lerp_point(corners[0], corners[1], t1)
            top_end = lerp_point(corners[0], corners[1], t2)
            bot_start = lerp_point(corners[3], corners[2], t1)
            bot_end = lerp_point(corners[3], corners[2], t2)
            
            # Refine for Half-Height
            # If Top Half: strip goes from Top Edge to Middle
            # If Bottom Half: strip goes from Middle to Bottom Edge
            
            mid_start = (top_start + bot_start) / 2
            mid_end = (top_end + bot_end) / 2
            
            if not is_bottom:
                # Top Half
                poly_pts = [top_start, top_end, mid_end, mid_start]
            else:
                # Bottom Half
                poly_pts = [mid_start, mid_end, bot_end, bot_start]
            
            # Scale to UI
            ui_poly = []
            for p in poly_pts:
                ui_poly.append((p[0] * scale_x, p[1] * scale_y))
            
            # Draw ENHANCED Transparent Overlay (like reference image)
            overlay = pygame.Surface((WINDOW_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
            # Semi-transparent yellow fill
            pygame.draw.polygon(overlay, (255, 255, 0, 60), ui_poly)  # More transparent fill
            # Bright yellow border (thicker and more visible)
            pygame.draw.polygon(overlay, (255, 255, 0, 255), ui_poly, 5)  # Opaque thick border
            self.screen.blit(overlay, (0, 0))
            
            # Optional: Draw step number in center of overlay
            center_x = sum(p[0] for p in ui_poly) / 4
            center_y = sum(p[1] for p in ui_poly) / 4
            step_text = self.font_large.render(str(step + 1), True, (255, 255, 0))
            text_rect = step_text.get_rect(center=(int(center_x), int(center_y)))
            # Shadow for visibility
            shadow = self.font_large.render(str(step + 1), True, (0, 0, 0))
            shadow_rect = shadow.get_rect(center=(int(center_x) + 2, int(center_y) + 2))
            self.screen.blit(shadow, shadow_rect)
            self.screen.blit(step_text, text_rect)

        self._draw_camera_overlay()
        self._draw_board_corners_indicator()  # FASE 2: Show board corners
    
    def _draw_large_grid(self):
        """
        FASE 2: Draw large interactive grid for virtual/no-camera mode.
        Centered in the screen, easy to click and edit.
        """
        # Center the grid
        base_x = (WINDOW_WIDTH - LARGE_SIZE - LARGE_LABEL_WIDTH - 20) // 2
        base_y = (CAMERA_HEIGHT - LARGE_SIZE) // 2
        label_x = base_x - LARGE_LABEL_WIDTH - 10
        
        # Title
        title = self.font_large.render("Virtual Mode - Click to Toggle Pieces", True, (255, 255, 100))
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, base_y - 40))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Click cells: Empty → Black → Empty",
            "Space: Play/Stop  |  C: Clear  |  L: Library",
            "↑/↓: Volume  |  ←/→: Kit  |  ESC: Quit"
        ]
        for i, text in enumerate(instructions):
            instr_surf = self.font_small.render(text, True, (200, 200, 200))
            instr_rect = instr_surf.get_rect(center=(WINDOW_WIDTH // 2, CAMERA_HEIGHT - 80 + i * 22))
            self.screen.blit(instr_surf, instr_rect)
        
        # Store grid bounds for click detection
        self._large_grid_bounds = {
            'base_x': base_x,
            'base_y': base_y,
            'cell_size': LARGE_CELL,
            'label_x': label_x
        }
        
        # Draw grid cells
        for row in range(8):
            for col in range(8):
                x = base_x + col * LARGE_CELL
                y = base_y + row * LARGE_CELL
                rect = pygame.Rect(x, y, LARGE_CELL, LARGE_CELL)
                
                # Alternate colors
                if (row + col) % 2 == 0:
                    color = COLOR_BOARD_LIGHT
                else:
                    color = COLOR_BOARD_DARK
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (60, 60, 60), rect, 2)
                
                # Draw pieces
                state = self.grid.get_cell(row, col)
                if state != EMPTY:
                    center = (x + LARGE_CELL // 2, y + LARGE_CELL // 2)
                    radius = LARGE_CELL // 2 - 6
                    piece_color = COLOR_PIECE_WHITE if state == WHITE else COLOR_PIECE_BLACK
                    pygame.draw.circle(self.screen, piece_color, center, radius)
                    pygame.draw.circle(self.screen, COLOR_PIECE_BORDER, center, radius, 2)
            
            # Instrument labels per row
            label = INSTRUMENT_LABELS[row % 4]
            label_surface = self.font.render(label, True, COLOR_TEXT)
            label_y = base_y + row * LARGE_CELL + (LARGE_CELL - label_surface.get_height()) // 2
            self.screen.blit(label_surface, (label_x, label_y))
        
        # Column numbers (steps)
        for col in range(8):
            x = base_x + col * LARGE_CELL
            # Upper half label
            label_surf = self.font_small.render(str(col + 1), True, (255, 100, 100))
            label_rect = label_surf.get_rect(center=(x + LARGE_CELL // 2, base_y - 15))
            self.screen.blit(label_surf, label_rect)
            # Lower half label
            label_surf = self.font_small.render(str(col + 9), True, (100, 100, 255))
            label_rect = label_surf.get_rect(center=(x + LARGE_CELL // 2, base_y + LARGE_SIZE + 15))
            self.screen.blit(label_surf, label_rect)
        
        # Draw playhead indicator
        step = self._highlighted_step
        is_upper = step < 8
        col = step % 8
        
        if is_upper:
            start_row = 0
            end_row = 4
            color = (255, 100, 100)  # Red for upper
        else:
            start_row = 4
            end_row = 8
            color = (100, 100, 255)  # Blue for lower
        
        x = base_x + col * LARGE_CELL
        y = base_y + start_row * LARGE_CELL
        height = (end_row - start_row) * LARGE_CELL
        
        # Thick playhead
        pygame.draw.rect(self.screen, color, (x, y, LARGE_CELL, height), 4)
        
        # Step number in playhead
        step_text = self.font_large.render(str(step + 1), True, color)
        step_rect = step_text.get_rect(center=(x + LARGE_CELL // 2, y + height // 2))
        # Shadow
        shadow = self.font_large.render(str(step + 1), True, (0, 0, 0))
        shadow_rect = shadow.get_rect(center=(x + LARGE_CELL // 2 + 2, y + height // 2 + 2))
        self.screen.blit(shadow, shadow_rect)
        self.screen.blit(step_text, step_rect)

    def _draw_mini_grid(self):
        """Draw small grid overlay in bottom right corner of camera view."""
        # Position in bottom right of camera area
        base_x = WINDOW_WIDTH - MINI_SIZE - MINI_MARGIN
        base_y = CAMERA_HEIGHT - MINI_SIZE - MINI_MARGIN
        label_x = base_x - MINI_LABEL_WIDTH - 6
        
        # Semi-transparent background
        bg_width = MINI_SIZE + MINI_LABEL_WIDTH + 12
        bg_surface = pygame.Surface((bg_width, MINI_SIZE + 4), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, (label_x - 2, base_y - 2))
        
        # Draw grid cells
        for row in range(8):
            for col in range(8):
                x = base_x + col * MINI_CELL
                y = base_y + row * MINI_CELL
                rect = pygame.Rect(x, y, MINI_CELL, MINI_CELL)
                
                # Alternate colors
                if (row + col) % 2 == 0:
                    color = COLOR_BOARD_LIGHT
                else:
                    color = COLOR_BOARD_DARK
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
                
                # Draw pieces
                state = self.grid.get_cell(row, col)
                if state != EMPTY:
                    center = (x + MINI_CELL // 2, y + MINI_CELL // 2)
                    radius = MINI_CELL // 2 - 2
                    piece_color = COLOR_PIECE_WHITE if state == WHITE else COLOR_PIECE_BLACK
                    pygame.draw.circle(self.screen, piece_color, center, radius)
                    pygame.draw.circle(self.screen, COLOR_PIECE_BORDER, center, radius, 1)
            
            # Instrument labels per row
            label = INSTRUMENT_LABELS[row % 4]
            label_surface = self.font_small.render(label, True, COLOR_TEXT)
            label_y = base_y + row * MINI_CELL + (MINI_CELL - label_surface.get_height()) // 2
            self.screen.blit(label_surface, (label_x, label_y))
        
        # Draw playhead indicator
        step = self._highlighted_step
        is_upper = step < 8
        col = step % 8
        
        if is_upper:
            start_row = 0
            end_row = 4
        else:
            start_row = 4
            end_row = 8
        
        x = base_x + col * MINI_CELL
        y = base_y + start_row * MINI_CELL
        height = (end_row - start_row) * MINI_CELL
        
        pygame.draw.rect(self.screen, COLOR_PLAYHEAD, (x, y, MINI_CELL, height), 2)
        
        # DEBUG: Show warped board view if debug mode is on
        if self.camera and hasattr(self.camera, 'board_detector'):
            if self.camera.board_detector.debug_mode:
                warped = self.camera.get_debug_warped()
                if warped is not None:
                    if cv2 is None:
                        return
                    # Position in bottom LEFT corner
                    debug_size = 200
                    debug_x = MINI_MARGIN
                    debug_y = CAMERA_HEIGHT - debug_size - MINI_MARGIN
                    
                    # Convert BGR to RGB and create surface
                    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
                    warped_surface = pygame.surfarray.make_surface(warped_rgb.swapaxes(0, 1))
                    warped_surface = pygame.transform.scale(warped_surface, (debug_size, debug_size))
                    
                    # Background
                    bg = pygame.Surface((debug_size + 4, debug_size + 4), pygame.SRCALPHA)
                    bg.fill((0, 0, 0, 200))
                    self.screen.blit(bg, (debug_x - 2, debug_y - 2))
                    
                    # Warped board
                    self.screen.blit(warped_surface, (debug_x, debug_y))
                    
                    # Label
                    label = self.font_small.render("DEBUG WARPED", True, (255, 255, 0))
                    self.screen.blit(label, (debug_x, debug_y - 20))

    def _draw_library_browser(self):
        if not self._library_browser_open:
            return

        panel_width = int(WINDOW_WIDTH * 0.55)
        panel_height = int(CAMERA_HEIGHT * 0.6)
        panel_x = (WINDOW_WIDTH - panel_width) // 2
        panel_y = (CAMERA_HEIGHT - panel_height) // 2
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 200))
        self.screen.blit(panel, (panel_x, panel_y))

        title = "Sound Libraries" if self._library_mode == "sound" else "Pattern Libraries"
        title_surf = self.font.render(title, True, COLOR_TEXT)
        self.screen.blit(title_surf, (panel_x + 20, panel_y + 15))

        libs = []
        if self._library_mode == "sound":
            if self.audio_output and hasattr(self.audio_output, 'sound_libraries'):
                libs = self.audio_output.sound_libraries
        else:
            libs = self.pattern_libraries

        if not libs:
            empty = self.font_small.render("No libraries loaded.", True, (200, 200, 200))
            self.screen.blit(empty, (panel_x + 20, panel_y + 60))
            return

        visible_rows = 8
        start = max(0, min(self._library_index, len(libs) - 1) - visible_rows + 1)
        end = min(len(libs), start + visible_rows)

        y = panel_y + 60
        for idx in range(start, end):
            lib = libs[idx]
            active = lib.get("active", True)
            name = lib.get("name", lib.get("id", ""))
            author = lib.get("author", "")
            date = lib.get("date", "")
            marker = "*" if active else "o"
            line = f"{marker} {name}"
            if author or date:
                line += f" - {author} {date}".strip()
            color = (255, 215, 120) if idx == self._library_index else (220, 220, 220)
            if not active:
                color = (120, 120, 120)
            line_surf = self.font_small.render(line, True, color)
            self.screen.blit(line_surf, (panel_x + 20, y))

            desc = lib.get("description", "")
            if desc:
                desc_surf = self.font_small.render(desc, True, (160, 160, 160))
                self.screen.blit(desc_surf, (panel_x + 40, y + 18))
                y += 36
            else:
                y += 26

        if self._library_mode == "pattern":
            lib = libs[self._library_index]
            patterns = lib.get("patterns", []) or []
            if patterns:
                pattern = patterns[self._pattern_index % len(patterns)]
                p_name = pattern.get("name", f"Pattern {self._pattern_index + 1}")
                p_desc = pattern.get("description", "")
                p_text = f"Selected: {p_name}"
                p_surf = self.font_small.render(p_text, True, (180, 220, 255))
                self.screen.blit(p_surf, (panel_x + 20, panel_y + panel_height - 80))
                if p_desc:
                    d_surf = self.font_small.render(p_desc, True, (140, 140, 140))
                    self.screen.blit(d_surf, (panel_x + 20, panel_y + panel_height - 60))

        if self._library_mode == "pattern":
            hint = "L: close  Tab: switch  Enter: apply  Up/Down: select  Left/Right: pattern"
        else:
            hint = "L: close  Tab: switch  Enter: apply  Up/Down: select"
        hint_surf = self.font_small.render(hint, True, (180, 180, 180))
        self.screen.blit(hint_surf, (panel_x + 20, panel_y + panel_height - 30))
    
    def _draw_controls(self):
        """Draw controls at bottom of window."""
        controls_y = CAMERA_HEIGHT + 10
        
        # BPM label and slider
        bpm_label = self.font.render("BPM:", True, COLOR_TEXT)
        self.screen.blit(bpm_label, (20, controls_y + 20))
        
        # BPM slider
        slider_x = 90
        slider_width = 300
        slider_rect = pygame.Rect(slider_x, controls_y + 25, slider_width, 10)
        pygame.draw.rect(self.screen, COLOR_SLIDER_BG, slider_rect, border_radius=5)
        
        # Fill based on BPM
        bpm_ratio = (self.sequencer.bpm - 30) / (300 - 30)
        fill_width = int(slider_width * bpm_ratio)
        fill_rect = pygame.Rect(slider_x, controls_y + 25, fill_width, 10)
        pygame.draw.rect(self.screen, COLOR_SLIDER_FILL, fill_rect, border_radius=5)
        
        # Slider handle
        handle_x = slider_x + fill_width
        pygame.draw.circle(self.screen, COLOR_TEXT, (handle_x, controls_y + 30), 8)
        
        # BPM value
        bpm_value = self.font_large.render(str(self.sequencer.bpm), True, COLOR_TEXT)
        self.screen.blit(bpm_value, (slider_x + slider_width + 20, controls_y + 10))
        
        # Stop/Play button
        btn_x = slider_x + slider_width + 120
        btn_rect = pygame.Rect(btn_x, controls_y + 15, 70, 40)
        btn_color = COLOR_BUTTON_ACTIVE if self.sequencer.is_playing else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=5)
        btn_text = "Stop" if self.sequencer.is_playing else "Play"
        text = self.font.render(btn_text, True, COLOR_TEXT)
        self.screen.blit(text, text.get_rect(center=btn_rect.center))

        # Kit label
        kit_y = controls_y + 90
        if self.audio_output and hasattr(self.audio_output, 'kit_name'):
            kit_label = getattr(self.audio_output, 'kit_display_name', self.audio_output.kit_name)
            kit_label = f"Kit (Left/Right): {kit_label}"
            kit_text = self.font_small.render(kit_label, True, COLOR_TEXT)
            self.screen.blit(kit_text, (20, kit_y))
            if hasattr(self.audio_output, 'master_volume'):
                volume_pct = int(self.audio_output.master_volume * 100)
                vol_text = self.font_small.render(f"Vol (Up/Down): {volume_pct}%", True, COLOR_TEXT)
                vol_x = 20 + kit_text.get_width() + 20
                self.screen.blit(vol_text, (vol_x, kit_y))
        
        # BPM indicator
        hand_toggle_rect = pygame.Rect(0, 0, 0, 0)
        if self.camera and hasattr(self.camera, 'hand_detector') and self.camera.hand_detector:
            hand_enabled = getattr(self.camera, 'hand_bpm_enabled', True)
            hand_label = "Hands BPM: ON" if hand_enabled else "Hands BPM: OFF"
            hand_color = COLOR_BUTTON_ACTIVE if hand_enabled else COLOR_BUTTON
            hand_toggle_rect = pygame.Rect(WINDOW_WIDTH - 180, controls_y + 15, 160, 36)
            pygame.draw.rect(self.screen, hand_color, hand_toggle_rect, border_radius=5)
            hand_text = self.font_small.render(hand_label, True, COLOR_TEXT)
            self.screen.blit(hand_text, hand_text.get_rect(center=hand_toggle_rect.center))
        
        cutoff_rect = pygame.Rect(0, 0, 0, 0)
        resonance_rect = pygame.Rect(0, 0, 0, 0)
        if self.audio_output and hasattr(self.audio_output, 'filter_cutoff') and getattr(self.audio_output, 'filter_enabled', True):
            filter_row_y = controls_y + 70
            row_padding = 20
            gap = 40
            column_width = int((WINDOW_WIDTH - (row_padding * 2) - gap) / 2)
            slider_height = 8
            slider_radius = 6

            cutoff_value = float(self.audio_output.filter_cutoff)
            cutoff_min = float(self.audio_output.filter_cutoff_min)
            cutoff_max = float(self.audio_output.filter_cutoff_max)
            cutoff_ratio = 0.0 if cutoff_max <= cutoff_min else (cutoff_value - cutoff_min) / (cutoff_max - cutoff_min)
            cutoff_ratio = max(0.0, min(1.0, cutoff_ratio))
            if cutoff_value >= 1000:
                cutoff_label = f"Cutoff {cutoff_value / 1000:.1f}kHz"
            else:
                cutoff_label = f"Cutoff {int(cutoff_value)}Hz"
            cutoff_label_surf = self.font_small.render(cutoff_label, True, COLOR_TEXT)
            cutoff_x = int(row_padding)
            self.screen.blit(cutoff_label_surf, (cutoff_x, filter_row_y - cutoff_label_surf.get_height() // 2))
            cutoff_slider_x = int(cutoff_x + cutoff_label_surf.get_width() + 8)
            cutoff_slider_w = max(60, int(column_width - cutoff_label_surf.get_width() - 8))
            cutoff_rect = pygame.Rect(cutoff_slider_x, filter_row_y - slider_height // 2, cutoff_slider_w, slider_height)
            pygame.draw.rect(self.screen, COLOR_SLIDER_BG, cutoff_rect, border_radius=4)
            cutoff_fill_w = int(cutoff_slider_w * cutoff_ratio)
            cutoff_fill = pygame.Rect(cutoff_slider_x, cutoff_rect.y, cutoff_fill_w, slider_height)
            pygame.draw.rect(self.screen, COLOR_SLIDER_FILL, cutoff_fill, border_radius=4)
            cutoff_handle_x = cutoff_slider_x + cutoff_fill_w
            pygame.draw.circle(self.screen, COLOR_TEXT, (cutoff_handle_x, filter_row_y), slider_radius)

            res_value = float(self.audio_output.filter_resonance)
            res_min = float(self.audio_output.filter_resonance_min)
            res_max = float(self.audio_output.filter_resonance_max)
            res_ratio = 0.0 if res_max <= res_min else (res_value - res_min) / (res_max - res_min)
            res_ratio = max(0.0, min(1.0, res_ratio))
            res_label = f"Resonance {res_value:.2f}"
            res_label_surf = self.font_small.render(res_label, True, COLOR_TEXT)
            res_x = int(row_padding + column_width + gap)
            self.screen.blit(res_label_surf, (res_x, filter_row_y - res_label_surf.get_height() // 2))
            res_slider_x = int(res_x + res_label_surf.get_width() + 8)
            res_slider_w = max(60, int(column_width - res_label_surf.get_width() - 8))
            resonance_rect = pygame.Rect(res_slider_x, filter_row_y - slider_height // 2, res_slider_w, slider_height)
            pygame.draw.rect(self.screen, COLOR_SLIDER_BG, resonance_rect, border_radius=4)
            res_fill_w = int(res_slider_w * res_ratio)
            res_fill = pygame.Rect(res_slider_x, resonance_rect.y, res_fill_w, slider_height)
            pygame.draw.rect(self.screen, COLOR_SLIDER_FILL, res_fill, border_radius=4)
            res_handle_x = res_slider_x + res_fill_w
            pygame.draw.circle(self.screen, COLOR_TEXT, (res_handle_x, filter_row_y), slider_radius)

        return (
            btn_rect,
            pygame.Rect(slider_x, controls_y + 15, slider_width, 30),
            cutoff_rect,
            resonance_rect,
            hand_toggle_rect,
        )
    
    def _get_mini_cell_from_pos(self, pos: tuple) -> Optional[tuple]:
        """Get (row, col) from mouse position on mini grid."""
        x, y = pos
        base_x = WINDOW_WIDTH - MINI_SIZE - MINI_MARGIN
        base_y = CAMERA_HEIGHT - MINI_SIZE - MINI_MARGIN
        
        col = (x - base_x) // MINI_CELL
        row = (y - base_y) // MINI_CELL
        
        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)
        return None
    
    def _get_large_cell_from_pos(self, pos: tuple) -> Optional[tuple]:
        """
        FASE 2: Get (row, col) from mouse position on large grid.
        Used in virtual/no-camera mode.
        """
        if not self._large_grid_bounds:
            return None
        
        x, y = pos
        base_x = self._large_grid_bounds['base_x']
        base_y = self._large_grid_bounds['base_y']
        cell_size = self._large_grid_bounds['cell_size']
        
        col = (x - base_x) // cell_size
        row = (y - base_y) // cell_size
        
        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)
        return None
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
        btn_rect, slider_rect, cutoff_rect, resonance_rect, hand_toggle_rect = self._control_rects
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                
                # FASE 4: Check mixer interactions first (if mixer is open)
                if self._mixer_open and self.audio_output:
                    handled = self._handle_mixer_mousedown(pos)
                    if handled:
                        continue
                
                if self._manual_board_mode:
                    if event.button == 1:
                        frame_point = self._screen_to_frame(pos)
                        if frame_point:
                            self._manual_points.append(frame_point)
                            if len(self._manual_points) == 4:
                                if self.camera and hasattr(self.camera, 'board_detector'):
                                    self.camera.board_detector.set_manual_corners(
                                        self._manual_points, self._last_frame_shape)
                                self._manual_points = []
                                self._manual_board_mode = False
                    elif event.button == 3 and self._manual_points:
                        self._manual_points.pop()
                    continue
                
                if btn_rect.collidepoint(pos):
                    self.sequencer.toggle()
                elif hand_toggle_rect.collidepoint(pos):
                    if self.camera and hasattr(self.camera, 'set_hand_bpm_enabled'):
                        self.camera.set_hand_bpm_enabled(not self.camera.hand_bpm_enabled)
                elif slider_rect.collidepoint(pos):
                    self._dragging_bpm_slider = True
                    self._update_bpm_from_pos(pos[0], slider_rect)
                elif cutoff_rect.collidepoint(pos):
                    self._dragging_cutoff_slider = True
                    self._update_cutoff_from_pos(pos[0], cutoff_rect)
                elif resonance_rect.collidepoint(pos):
                    self._dragging_resonance_slider = True
                    self._update_resonance_from_pos(pos[0], resonance_rect)
                else:
                    # FASE 2: Check appropriate grid based on mode
                    cell = None
                    if self.camera:
                        # Camera mode: use mini grid
                        cell = self._get_mini_cell_from_pos(pos)
                    else:
                        # Virtual mode: use large grid
                        cell = self._get_large_cell_from_pos(pos)
                    
                    if cell:
                        self.grid.toggle(cell[0], cell[1])
            
            elif event.type == pygame.MOUSEBUTTONUP:
                # FASE 4: Release mixer dragging
                self._dragging_channel_fader = None
                self._dragging_delay_knob = None
                self._dragging_reverb_knob = None
                
                self._dragging_bpm_slider = False
                self._dragging_cutoff_slider = False
                self._dragging_resonance_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                # FASE 4: Handle mixer dragging
                if self._mixer_open and self.audio_output:
                    self._handle_mixer_drag(event.pos)
                
                if self._dragging_bpm_slider:
                    self._update_bpm_from_pos(event.pos[0], slider_rect)
                elif self._dragging_cutoff_slider:
                    self._update_cutoff_from_pos(event.pos[0], cutoff_rect)
                elif self._dragging_resonance_slider:
                    self._update_resonance_from_pos(event.pos[0], resonance_rect)
            
            elif event.type == pygame.KEYDOWN:
                if self._library_browser_open:
                    if event.key == pygame.K_l or event.key == pygame.K_ESCAPE:
                        self._library_browser_open = False
                        continue
                    if event.key == pygame.K_TAB:
                        self._library_mode = "pattern" if self._library_mode == "sound" else "sound"
                        self._library_index = 0
                        self._pattern_index = 0
                        continue
                    if event.key == pygame.K_UP:
                        self._library_index = max(0, self._library_index - 1)
                        continue
                    if event.key == pygame.K_DOWN:
                        max_idx = 0
                        if self._library_mode == "sound":
                            libs = getattr(self.audio_output, 'sound_libraries', []) if self.audio_output else []
                            max_idx = max(0, len(libs) - 1)
                        else:
                            max_idx = max(0, len(self.pattern_libraries) - 1)
                        self._library_index = min(max_idx, self._library_index + 1)
                        continue
                    if self._library_mode == "pattern":
                        if event.key == pygame.K_LEFT:
                            self._pattern_index = max(0, self._pattern_index - 1)
                            continue
                        if event.key == pygame.K_RIGHT:
                            self._pattern_index = self._pattern_index + 1
                            continue
                    if event.key == pygame.K_RETURN:
                        self._apply_library_selection()
                        continue
                    continue

                if event.key == pygame.K_SPACE:
                    self.sequencer.toggle()
                elif event.key == pygame.K_c:
                    self.grid.clear()
                elif event.key == pygame.K_r:
                    # RECALIBRATE board baseline
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.calibrated = False
                        self.camera.board_detector.empty_cells_baseline = None
                        self.camera.board_detector.detection_history.clear()
                        self.show_notification("🔄 Recalibrating board...", (100, 200, 255))
                elif event.key == pygame.K_m:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self._manual_board_mode = not self._manual_board_mode
                        self._manual_points = []
                        if self._manual_board_mode:
                            self.camera.board_detector.clear_manual_corners()
                            self.show_notification("📐 Click 4 corners: TL, TR, BR, BL", (255, 255, 100))
                        else:
                            self.show_notification("Manual mode OFF", (200, 200, 200))
                elif event.key == pygame.K_BACKSPACE:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.clear_manual_corners()
                    self._manual_board_mode = False
                    self._manual_points = []
                    self.show_notification("Manual board cleared", (200, 200, 200))
                elif event.key == pygame.K_d:
                    # TOGGLE DEBUG MODE
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.debug_mode = not self.camera.board_detector.debug_mode
                        state = "ON" if self.camera.board_detector.debug_mode else "OFF"
                        print(f"🐛 Debug mode: {state}")
                # SENSITIVITY ADJUSTMENT (1-9 keys)
                elif event.key == pygame.K_1:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.1
                        print(f"🎯 Sensitivity: 0.1 (very strict)")
                elif event.key == pygame.K_2:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.2
                        print(f"🎯 Sensitivity: 0.2")
                elif event.key == pygame.K_3:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.3
                        print(f"🎯 Sensitivity: 0.3")
                elif event.key == pygame.K_4:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.4
                        print(f"🎯 Sensitivity: 0.4")
                elif event.key == pygame.K_5:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.5
                        print(f"🎯 Sensitivity: 0.5 (balanced)")
                elif event.key == pygame.K_6:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.6
                        print(f"🎯 Sensitivity: 0.6")
                elif event.key == pygame.K_7:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.7
                        print(f"🎯 Sensitivity: 0.7")
                elif event.key == pygame.K_8:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.8
                        print(f"🎯 Sensitivity: 0.8")
                elif event.key == pygame.K_9:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.sensitivity = 0.9
                        print(f"🎯 Sensitivity: 0.9 (very sensitive)")
                # THRESHOLD ADJUSTMENT (T/G keys)
                elif event.key == pygame.K_t:
                    # Increase threshold (stricter)
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.dark_threshold = min(100, self.camera.board_detector.dark_threshold + 5)
                        print(f"🌑 Dark threshold: {self.camera.board_detector.dark_threshold}")
                elif event.key == pygame.K_g:
                    # Decrease threshold (more permissive)
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.dark_threshold = max(20, self.camera.board_detector.dark_threshold - 5)
                        print(f"🌑 Dark threshold: {self.camera.board_detector.dark_threshold}")
                elif event.key == pygame.K_q:
                    # Brightness UP
                    if self.camera:
                        self.camera.brightness = min(100, self.camera.brightness + 5)
                elif event.key == pygame.K_a:
                    # Brightness DOWN
                    if self.camera:
                        self.camera.brightness = max(-100, self.camera.brightness - 5)
                elif event.key == pygame.K_w:
                    # Contrast UP
                    if self.camera:
                        self.camera.contrast = min(3.0, self.camera.contrast + 0.1)
                elif event.key == pygame.K_s:
                    # Contrast DOWN
                    if self.camera:
                        self.camera.contrast = max(0.1, self.camera.contrast - 0.1)
                elif event.key == pygame.K_e:
                    if self.camera:
                        self.camera.brightness = 0
                        self.camera.contrast = 1.0
                elif event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_h:
                    if self.camera and hasattr(self.camera, 'set_hand_bpm_enabled'):
                        self.camera.set_hand_bpm_enabled(not self.camera.hand_bpm_enabled)
                elif event.key == pygame.K_l:
                    self._library_browser_open = not self._library_browser_open
                    if self._library_browser_open:
                        self._library_mode = "sound"
                        self._library_index = 0
                        self._pattern_index = 0
                # FASE 4: Mixer toggle
                elif event.key == pygame.K_x:
                    self._mixer_open = not self._mixer_open
                    state = "ON" if self._mixer_open else "OFF"
                    self.show_notification(f"🎚️ Mixer {state}", (100, 200, 255))
                # Kit + volume controls
                elif event.key == pygame.K_LEFT:
                    if self.audio_output:
                        new_kit = self.audio_output.cycle_kit(-1)
                        print(f"Kit: {new_kit}")
                elif event.key == pygame.K_RIGHT:
                    if self.audio_output:
                        new_kit = self.audio_output.cycle_kit(1)
                        print(f"Kit: {new_kit}")
                elif event.key == pygame.K_UP:
                    if self.audio_output and hasattr(self.audio_output, 'adjust_volume'):
                        self.audio_output.adjust_volume(0.05)
                elif event.key == pygame.K_DOWN:
                    if self.audio_output and hasattr(self.audio_output, 'adjust_volume'):
                        self.audio_output.adjust_volume(-0.05)
                elif event.key == pygame.K_0:
                    if self.audio_output:
                        if hasattr(self.audio_output, 'reset_filter'):
                            self.audio_output.reset_filter()
        
        return True
    
    def _update_bpm_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update BPM from slider position."""
        ratio = (x - slider_rect.x) / slider_rect.width
        ratio = max(0, min(1, ratio))
        self.sequencer.bpm = int(30 + ratio * (300 - 30))

    def _update_cutoff_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update filter cutoff from slider position."""
        if not self.audio_output or slider_rect.width <= 0:
            return
        ratio = (x - slider_rect.x) / slider_rect.width
        ratio = max(0.0, min(1.0, ratio))
        cutoff_min = getattr(self.audio_output, 'filter_cutoff_min', 80.0)
        cutoff_max = getattr(self.audio_output, 'filter_cutoff_max', 12000.0)
        cutoff = cutoff_min + ratio * (cutoff_max - cutoff_min)
        self.audio_output.set_filter_cutoff(cutoff)

    def _update_resonance_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update filter resonance from slider position."""
        if not self.audio_output or slider_rect.width <= 0:
            return
        ratio = (x - slider_rect.x) / slider_rect.width
        ratio = max(0.0, min(1.0, ratio))
        res_min = getattr(self.audio_output, 'filter_resonance_min', 0.5)
        res_max = getattr(self.audio_output, 'filter_resonance_max', 6.0)
        resonance = res_min + ratio * (res_max - res_min)
        self.audio_output.set_filter_resonance(resonance)

    def _apply_library_selection(self):
        if self._library_mode == "sound":
            if not self.audio_output or not hasattr(self.audio_output, 'sound_libraries'):
                return
            libs = self.audio_output.sound_libraries
            if not libs:
                return
            lib = libs[min(self._library_index, len(libs) - 1)]
            if not lib.get("active", True):
                return
            kit_id = lib.get("id")
            if kit_id:
                self.audio_output.set_kit(kit_id)
                kit_name = lib.get("name", kit_id)
                self.show_notification(f"🎵 Kit: {kit_name}", (100, 255, 150))
            self._library_browser_open = False
            return

        libs = self.pattern_libraries
        if not libs:
            return
        lib = libs[min(self._library_index, len(libs) - 1)]
        if not lib.get("active", True):
            return
        patterns = lib.get("patterns", []) or []
        if not patterns:
            return
        pattern = patterns[self._pattern_index % len(patterns)]
        grid = pattern.get("grid")
        if grid and hasattr(self.grid, "set_matrix"):
            self.grid.set_matrix(grid)
            pattern_name = pattern.get("name", "Pattern")
            self.show_notification(f"📋 Loaded: {pattern_name}", (150, 200, 255))
        kit_id = lib.get("kit")
        if kit_id and self.audio_output:
            self.audio_output.set_kit(kit_id)
        self._library_browser_open = False
    
    def _draw_performance_metrics(self):
        """
        FASE 2: Draw performance metrics overlay.
        Shows FPS, queue size, detection status.
        """
        if not self.camera:
            return
        
        try:
            stats = self.camera.get_performance_stats()
        except:
            return
        
        # Performance metrics box (top right corner)
        metrics_x = WINDOW_WIDTH - 200
        metrics_y = 5
        
        # Build metrics text
        capture_fps = stats.get('capture_fps', 0)
        detection_fps = stats.get('detection_fps', 0)
        queue_size = stats.get('frame_queue_size', 0)
        queue_max = stats.get('frame_queue_maxsize', 2)
        
        metrics_text = [
            f"Capture:  {capture_fps:.1f} FPS",
            f"Detection: {detection_fps:.1f} FPS",
            f"Queue: {queue_size}/{queue_max}"
        ]
        
        # Draw background panel
        metrics_height = len(metrics_text) * 22 + 10
        bg_rect = pygame.Rect(metrics_x - 5, metrics_y - 5, 205, metrics_height)
        bg_surf = pygame.Surface((205, metrics_height), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        self.screen.blit(bg_surf, (metrics_x - 5, metrics_y - 5))
        pygame.draw.rect(self.screen, (100, 200, 100), bg_rect, 1)
        
        # Draw text
        for i, line in enumerate(metrics_text):
            text_surf = self.font_small.render(line, True, (100, 255, 100))
            self.screen.blit(text_surf, (metrics_x, metrics_y + i * 22))
    
    def _draw_calibration_status(self):
        """
        FASE 2: Draw calibration status indicator.
        Shows if board is calibrated and stable.
        """
        if not self.camera or not hasattr(self.camera, 'board_detector'):
            return
        
        board_det = self.camera.board_detector
        
        # Calibration status box (top left corner)
        cal_x = 5
        cal_y = 5
        
        # Determine calibration state
        cal_data = board_det.get_calibration_data()
        stability = getattr(board_det, 'board_stability_score', 0.0)
        
        if cal_data:
            if stability > 0.7:
                status_text = "✓ CALIBRATED"
                status_color = (100, 255, 100)
            else:
                status_text = f"⚙ CALIBRATING ({stability:.0%})"
                status_color = (255, 200, 50)
        else:
            status_text = "⚠ NOT CALIBRATED"
            status_color = (255, 100, 100)
        
        # Background panel
        text_surf = self.font_small.render(status_text, True, status_color)
        bg_height = text_surf.get_height() + 6
        bg_rect = pygame.Rect(cal_x - 2, cal_y - 2, text_surf.get_width() + 6, bg_height)
        bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 150))
        self.screen.blit(bg_surf, bg_rect)
        pygame.draw.rect(self.screen, status_color, bg_rect, 1)
        
        # Draw text
        self.screen.blit(text_surf, (cal_x + 2, cal_y + 2))
    
    def _draw_hand_detection_status(self):
        """
        FASE 2: Draw hand detection status.
        Shows if hand tracking is active and BPM control available.
        """
        if not self.camera or not hasattr(self.camera, 'hand_detector'):
            return
        
        hand_det = self.camera.hand_detector
        hand_bpm = getattr(self.camera, 'hand_bpm_enabled', False)
        
        # Status position (top center)
        status_x = WINDOW_WIDTH // 2 - 100
        status_y = 5
        
        if hand_bpm:
            # Check if hand is being detected
            if hasattr(hand_det, 'state') and hand_det.state == 'active':
                status_text = "👋 HAND DETECTED"
                status_color = (100, 255, 100)
            else:
                status_text = "👋 Waiting for hand..."
    
    def _draw_notifications(self):
        """
        FASE 2: Draw temporary notification messages (errors, warnings, info).
        Displayed at bottom center, auto-fade after duration.
        """
        import time
        current_time = time.time()
        
        # Remove expired notifications
        self._notifications = [
            (msg, color, ts) for msg, color, ts in self._notifications
            if current_time - ts < self._notification_duration
        ]
        
        if not self._notifications:
            return
        
        # Draw notifications from bottom to top
        y_offset = CAMERA_HEIGHT - 30
        for msg, color, timestamp in reversed(self._notifications):
            # Calculate fade
            age = current_time - timestamp
            alpha = int(255 * (1.0 - age / self._notification_duration))
            alpha = max(0, min(255, alpha))
            
            # Render text
            text_surf = self.font.render(msg, True, color)
            text_rect = text_surf.get_rect(center=(WINDOW_WIDTH // 2, y_offset))
            
            # Background with fade
            padding = 10
            bg_rect = text_rect.inflate(padding * 2, padding)
            bg_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            bg_surf.fill((0, 0, 0, min(200, alpha)))
            self.screen.blit(bg_surf, bg_rect)
            
            # Border with fade
            border_color = (*color, alpha)
            border_surf = pygame.Surface(bg_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(border_surf, border_color, border_surf.get_rect(), 2)
            self.screen.blit(border_surf, bg_rect)
            
            # Text with fade
            text_surf.set_alpha(alpha)
            self.screen.blit(text_surf, text_rect)
            
            y_offset -= bg_rect.height + 5
    
    def _draw_mixer_panel(self):
        """
        FASE 4: Draw channel mixer panel on the right side.
        Shows when mixer is open (toggle with X key).
        """
        if not self.audio_output or not self._mixer_open:
            return
        
        # Panel background
        mixer_x = WINDOW_WIDTH
        panel_bg = pygame.Rect(mixer_x, 0, MIXER_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, (30, 30, 35), panel_bg)
        pygame.draw.rect(self.screen, (80, 80, 90), panel_bg, 2)
        
        # Title
        title = self.font.render("MIXER", True, (200, 200, 255))
        self.screen.blit(title, (mixer_x + 10, 10))
        
        # Draw 4 channel strips
        channel_width = 45
        start_x = mixer_x + 5
        start_y = 45
        strip_height = CAMERA_HEIGHT - 50
        
        for i in range(4):
            x = start_x + i * (channel_width + 2)
            self._draw_channel_strip(x, start_y, channel_width, strip_height, i)
    
    def _draw_channel_strip(self, x, y, width, height, channel_idx):
        """
        FASE 4: Draw individual channel strip with fader, mute, and FX knobs.
        """
        if not self.audio_output:
            return
        
        # Background
        bg = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (40, 42, 48), bg)
        pygame.draw.rect(self.screen, (70, 72, 78), bg, 1)
        
        # Label at top
        label_text = CHANNEL_NAMES[channel_idx]
        label = self.font_small.render(label_text, True, (255, 255, 100))
        label_rect = label.get_rect(center=(x + width // 2, y + 12))
        self.screen.blit(label, label_rect)
        
        # Volume fader (vertical)
        fader_x = x + width // 2 - 6
        fader_y = y + 30
        fader_height = height - 180
        volume = self.audio_output.channel_volumes[channel_idx]
        self._draw_vertical_fader(fader_x, fader_y, 12, fader_height, volume, channel_idx)
        
        # Volume percentage
        vol_pct = int(volume * 100)
        vol_text = self.font_small.render(f"{vol_pct}%", True, (200, 200, 200))
        vol_rect = vol_text.get_rect(center=(x + width // 2, y + fader_height + 40))
        self.screen.blit(vol_text, vol_rect)
        
        # Mute button
        mute_y = y + fader_height + 55
        is_muted = self.audio_output.channel_mutes[channel_idx]
        self._draw_mute_button(x + 5, mute_y, width - 10, 20, is_muted, channel_idx)
        
        # Delay knob
        delay_y = y + fader_height + 85
        delay_val = self.audio_output.channel_delay[channel_idx]
        self._draw_knob(x + width // 2 - 15, delay_y, 30, delay_val, "DLY", channel_idx, "delay")
        
        # Reverb knob
        reverb_y = y + fader_height + 135
        reverb_val = self.audio_output.channel_reverb[channel_idx]
        self._draw_knob(x + width // 2 - 15, reverb_y, 30, reverb_val, "REV", channel_idx, "reverb")
    
    def _draw_vertical_fader(self, x, y, width, height, value, channel_idx):
        """Draw vertical fader for volume control."""
        # Track background
        track = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (50, 50, 55), track)
        pygame.draw.rect(self.screen, (70, 70, 75), track, 1)
        
        # Fill (from bottom)
        fill_height = int(height * value)
        if fill_height > 0:
            fill = pygame.Rect(x + 1, y + height - fill_height, width - 2, fill_height)
            pygame.draw.rect(self.screen, (70, 150, 255), fill)
        
        # Thumb
        thumb_y = y + height - int(height * value)
        thumb = pygame.Rect(x - 2, thumb_y - 4, width + 4, 8)
        pygame.draw.rect(self.screen, (255, 255, 255), thumb)
        pygame.draw.rect(self.screen, (100, 100, 100), thumb, 1)
        
        # Store rect for mouse interaction
        if not hasattr(self, '_fader_rects'):
            self._fader_rects = {}
        self._fader_rects[channel_idx] = track
    
    def _draw_mute_button(self, x, y, width, height, is_muted, channel_idx):
        """Draw mute button."""
        btn = pygame.Rect(x, y, width, height)
        
        # Color based on state
        if is_muted:
            color = (200, 50, 50)
            text_color = (255, 255, 255)
        else:
            color = (60, 60, 65)
            text_color = (150, 150, 150)
        
        pygame.draw.rect(self.screen, color, btn)
        pygame.draw.rect(self.screen, (100, 100, 100), btn, 1)
        
        # Label
        label = self.font_small.render("M", True, text_color)
        label_rect = label.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(label, label_rect)
        
        # Store rect for mouse interaction
        if not hasattr(self, '_mute_rects'):
            self._mute_rects = {}
        self._mute_rects[channel_idx] = btn
    
    def _draw_knob(self, x, y, size, value, label, channel_idx, fx_type):
        """Draw circular knob for FX control."""
        center_x = x + size // 2
        center_y = y + size // 2
        radius = size // 2
        
        # Outer circle
        pygame.draw.circle(self.screen, (50, 50, 55), (center_x, center_y), radius)
        pygame.draw.circle(self.screen, (100, 100, 105), (center_x, center_y), radius, 2)
        
        # Value arc (270 degrees total, from -135 to +135)
        import math
        if value > 0:
            start_angle = -135 * math.pi / 180
            end_angle = start_angle + (270 * value * math.pi / 180)
            
            # Draw arc with multiple lines
            points = []
            steps = 20
            for i in range(steps + 1):
                angle = start_angle + (end_angle - start_angle) * i / steps
                px = center_x + int((radius - 3) * math.cos(angle))
                py = center_y + int((radius - 3) * math.sin(angle))
                points.append((px, py))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, (100, 200, 100), False, points, 3)
        
        # Indicator line
        angle = -135 + (270 * value)
        angle_rad = angle * math.pi / 180
        end_x = center_x + int((radius - 5) * math.cos(angle_rad))
        end_y = center_y + int((radius - 5) * math.sin(angle_rad))
        pygame.draw.line(self.screen, (255, 255, 255), (center_x, center_y), (end_x, end_y), 2)
        
        # Label below
        label_surf = self.font_small.render(label, True, (150, 150, 150))
        label_rect = label_surf.get_rect(center=(center_x, y + size + 10))
        self.screen.blit(label_surf, label_rect)
        
        # Value percentage
        val_pct = int(value * 100)
        val_surf = self.font_small.render(f"{val_pct}%", True, (180, 180, 180))
        val_rect = val_surf.get_rect(center=(center_x, y + size + 25))
        self.screen.blit(val_surf, val_rect)
        
        # Store rect for mouse interaction
        if not hasattr(self, '_knob_rects'):
            self._knob_rects = {}
        knob_rect = pygame.Rect(x, y, size, size)
        self._knob_rects[(channel_idx, fx_type)] = knob_rect
    
    def _handle_mixer_mousedown(self, pos):
        """
        FASE 4: Handle mouse clicks on mixer panel.
        Returns True if event was handled.
        """
        # Check mute buttons
        if hasattr(self, '_mute_rects'):
            for channel_idx, rect in self._mute_rects.items():
                if rect.collidepoint(pos):
                    self.audio_output.channel_mutes[channel_idx] = not self.audio_output.channel_mutes[channel_idx]
                    state = "MUTED" if self.audio_output.channel_mutes[channel_idx] else "ON"
                    self.show_notification(f"CH{channel_idx+1} {state}", (200, 100, 100) if self.audio_output.channel_mutes[channel_idx] else (100, 200, 100))
                    return True
        
        # Check faders
        if hasattr(self, '_fader_rects'):
            for channel_idx, rect in self._fader_rects.items():
                if rect.collidepoint(pos):
                    self._dragging_channel_fader = channel_idx
                    self._update_fader_from_pos(pos, rect, channel_idx)
                    return True
        
        # Check knobs
        if hasattr(self, '_knob_rects'):
            for (channel_idx, fx_type), rect in self._knob_rects.items():
                if rect.collidepoint(pos):
                    if fx_type == "delay":
                        self._dragging_delay_knob = channel_idx
                        self._update_knob_from_pos(pos, rect, channel_idx, fx_type)
                    elif fx_type == "reverb":
                        self._dragging_reverb_knob = channel_idx
                        self._update_knob_from_pos(pos, rect, channel_idx, fx_type)
                    return True
        
        return False
    
    def _handle_mixer_drag(self, pos):
        """FASE 4: Handle mouse dragging on mixer controls."""
        # Handle fader drag
        if self._dragging_channel_fader is not None and hasattr(self, '_fader_rects'):
            channel_idx = self._dragging_channel_fader
            if channel_idx in self._fader_rects:
                rect = self._fader_rects[channel_idx]
                self._update_fader_from_pos(pos, rect, channel_idx)
        
        # Handle delay knob drag
        if self._dragging_delay_knob is not None and hasattr(self, '_knob_rects'):
            channel_idx = self._dragging_delay_knob
            if (channel_idx, "delay") in self._knob_rects:
                rect = self._knob_rects[(channel_idx, "delay")]
                self._update_knob_from_pos(pos, rect, channel_idx, "delay")
        
        # Handle reverb knob drag
        if self._dragging_reverb_knob is not None and hasattr(self, '_knob_rects'):
            channel_idx = self._dragging_reverb_knob
            if (channel_idx, "reverb") in self._knob_rects:
                rect = self._knob_rects[(channel_idx, "reverb")]
                self._update_knob_from_pos(pos, rect, channel_idx, "reverb")
    
    def _update_fader_from_pos(self, pos, rect, channel_idx):
        """Update channel volume from vertical fader position."""
        # Clamp y to rect bounds
        y = max(rect.y, min(rect.y + rect.height, pos[1]))
        
        # Calculate value (inverted: top = 1.0, bottom = 0.0)
        ratio = 1.0 - (y - rect.y) / rect.height
        ratio = max(0.0, min(1.0, ratio))
        
        self.audio_output.channel_volumes[channel_idx] = ratio
    
    def _update_knob_from_pos(self, pos, rect, channel_idx, fx_type):
        """Update FX knob from mouse position."""
        import math
        
        # Calculate angle from center
        center_x = rect.x + rect.width // 2
        center_y = rect.y + rect.height // 2
        
        dx = pos[0] - center_x
        dy = pos[1] - center_y
        
        # Convert to angle (0 = right, 90 = down, etc.)
        angle_rad = math.atan2(dy, dx)
        angle_deg = angle_rad * 180 / math.pi
        
        # Normalize to -135 to +135 range
        # Map to 0.0 - 1.0
        # -135 = 0%, +135 = 100%
        normalized_angle = (angle_deg + 135) % 360
        if normalized_angle > 270:
            normalized_angle = 270
        
        value = normalized_angle / 270.0
        value = max(0.0, min(1.0, value))
        
        if fx_type == "delay":
            self.audio_output.channel_delay[channel_idx] = value
        elif fx_type == "reverb":
            self.audio_output.channel_reverb[channel_idx] = value

    def _draw_board_corners_indicator(self):
        """
        FASE 2: Draw visual indicator of board corners detection.
        Shows where the board is detected in the camera feed.
        """
        if not self.camera or not hasattr(self.camera, 'board_detector'):
            return
        
        board_det = self.camera.board_detector
        
        # Check if corners are detected
        if not hasattr(board_det, 'corners_norm') or board_det.corners_norm is None:
            return
        
        try:
            # Get current frame to determine size
            frame = self.camera.get_frame()
            if frame is None:
                return
            
            frame_h, frame_w = frame.shape[:2]
            
            # Convert normalized corners to screen coordinates
            corners_norm = board_det.corners_norm
            corners_screen = []
            
            for corner_norm in corners_norm:
                # Denormalize from [0,1] range
                x_norm, y_norm = corner_norm
                
                # Scale to frame size
                x_frame = x_norm * frame_w
                y_frame = y_norm * frame_h
                
                # Scale to screen size (might be different aspect ratio)
                if self._last_frame_shape:
                    last_h, last_w = self._last_frame_shape[:2]
                    scale_x = WINDOW_WIDTH / last_w
                    scale_y = CAMERA_HEIGHT / last_h
                    x_screen = int(x_frame * scale_x)
                    y_screen = int(y_frame * scale_y)
                else:
                    x_screen = int(x_frame)
                    y_screen = int(y_frame)
                
                corners_screen.append((x_screen, y_screen))
            
            if len(corners_screen) == 4:
                # Draw filled polygon with transparency
                poly_color = (50, 255, 50, 60)
                poly_surf = pygame.Surface((WINDOW_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(poly_surf, poly_color, corners_screen)
                self.screen.blit(poly_surf, (0, 0))
                
                # Draw corner points
                for i, corner in enumerate(corners_screen):
                    pygame.draw.circle(self.screen, (0, 255, 0), corner, 6)
                    pygame.draw.circle(self.screen, (255, 255, 255), corner, 6, 2)
                    
                    # Label corners: TL, TR, BR, BL
                    labels = ["TL", "TR", "BR", "BL"]
                    label_text = self.font_small.render(labels[i], True, (255, 255, 255))
                    self.screen.blit(label_text, (corner[0] + 10, corner[1] + 10))
                
                # Draw border lines
                for i in range(4):
                    p1 = corners_screen[i]
                    p2 = corners_screen[(i + 1) % 4]
                    pygame.draw.line(self.screen, (100, 255, 100), p1, p2, 2)
        except:
            pass

    def draw(self):
        """Draw everything."""
        # FASE 4: Dynamic window resize based on mixer state
        current_width = MIXER_ENABLED_WIDTH if self._mixer_open else WINDOW_WIDTH
        if self.screen.get_width() != current_width:
            self.screen = pygame.display.set_mode((current_width, WINDOW_HEIGHT))
        
        self.screen.fill(COLOR_BG)
        
        # FASE 2: Choose layout based on camera availability
        if self.camera:
            # Camera mode: small grid in corner, camera feed
            self._draw_camera_feed()
            self._draw_mini_grid()
            
            # FASE 2: UI/UX overlays
            self._draw_performance_metrics()
            self._draw_calibration_status()
            self._draw_hand_detection_status()
        else:
            # Virtual mode: large centered grid, no camera
            self._draw_large_grid()
        
        self._draw_library_browser()
        self._control_rects = self._draw_controls()
        
        # FASE 4: Draw mixer panel and notifications last (on top)
        self._draw_notifications()
        self._draw_mixer_panel()
    
    def run(self):
        """Main loop."""
        self._control_rects = (pygame.Rect(0, 0, 0, 0),) * 5
        
        while self.running:
            if not self.handle_events():
                self.running = False
                break
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        # FASE 4: Cleanup mixer references before pygame.quit()
        self._cleanup_mixer_refs()
        pygame.quit()
    
    def _cleanup_mixer_refs(self):
        """Clean up mixer references to avoid segfault on quit."""
        if hasattr(self, '_fader_rects'):
            self._fader_rects.clear()
        if hasattr(self, '_mute_rects'):
            self._mute_rects.clear()
        if hasattr(self, '_knob_rects'):
            self._knob_rects.clear()
