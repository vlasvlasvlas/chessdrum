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
COLOR_FILTER_LOW = (255, 100, 100)
COLOR_FILTER_HIGH = (100, 200, 255)
COLOR_FILTER_CENTER = (100, 255, 100)

# Layout constants - camera-first design
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 650
CONTROLS_HEIGHT = 80
CAMERA_HEIGHT = WINDOW_HEIGHT - CONTROLS_HEIGHT

# Mini grid settings (bottom right corner)
MINI_CELL = 18
MINI_MARGIN = 10
MINI_SIZE = MINI_CELL * 8
MINI_LABEL_WIDTH = 30

# Instrument labels
INSTRUMENT_LABELS = ['HH', 'CP', 'SD', 'KK']


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
        self._dragging_filter_slider = False
        self._manual_board_mode = False
        self._manual_points = []
        self._last_frame_shape = None
        
        # Register callback for step changes
        self.sequencer.on_step_change = self._on_step_change
    
    def _on_step_change(self, step: int):
        """Called by sequencer when step changes."""
        self._highlighted_step = step

    def _draw_camera_overlay(self):
        """Draw camera settings and detection status."""
        if not self.camera:
            return

        y_off = 10
        settings_font = self.font_small
        right_margin = WINDOW_WIDTH - 10

        def draw_line(text, color):
            nonlocal y_off
            shad = settings_font.render(text, True, (0, 0, 0))
            txt = settings_font.render(text, True, color)
            x = right_margin - txt.get_width()
            self.screen.blit(shad, (x + 2, y_off + 2))
            self.screen.blit(txt, (x, y_off))
            y_off += 25

        # Brightness
        b_text = f"Brightness (Q/A): {self.camera.brightness}"
        draw_line(b_text, (255, 255, 255))

        # Contrast
        c_text = f"Contrast (W/S): {self.camera.contrast:.1f}"
        draw_line(c_text, (255, 255, 255))

        # Detection parameters
        if hasattr(self.camera, 'board_detector') and self.camera.board_detector:
            # Sensitivity
            sens = self.camera.board_detector.sensitivity
            sens_text = f"Sensitivity (1-9): {sens:.1f}"
            color = (100, 255, 100) if 0.4 <= sens <= 0.6 else (255, 200, 100)
            draw_line(sens_text, color)

            # Dark threshold
            thresh = self.camera.board_detector.dark_threshold
            thresh_text = f"Dark Thresh (T/G): {thresh}"
            draw_line(thresh_text, (255, 255, 100))

            # Debug mode indicator
            if self.camera.board_detector.debug_mode:
                d_text = "DEBUG: ON (D to toggle)"
                draw_line(d_text, (255, 255, 0))

            # Calibration status
            if self.camera.board_detector.calibrated:
                cal_text = "Calibrated ✓ (R to reset)"
                color = (0, 255, 0)
            else:
                cal_text = "Calibrating... (R to reset)"
                color = (255, 200, 0)
            draw_line(cal_text, color)

            manual_corners = getattr(self.camera.board_detector, 'manual_corners_norm', None)
            if self._manual_board_mode:
                step = min(len(self._manual_points) + 1, 4)
                manual_text = f"Manual board: click {step}/4 (TL,TR,BR,BL)"
                draw_line(manual_text, (255, 255, 0))
            elif manual_corners is not None:
                manual_text = "Manual board: ON (Backspace to clear)"
                draw_line(manual_text, (255, 255, 0))

            draw_line("RED = TOP (Steps 1-8)", (255, 50, 50))
            draw_line("BLUE = BOTTOM (Steps 9-16)", (50, 50, 255))

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
        
        # Filter indicator (small)
        filter_x = btn_x + 90
        if self.audio_output and hasattr(self.audio_output, 'filter_value'):
            fv = self.audio_output.filter_value
            if fv < -0.1:
                filter_text = f"LP"
                color = COLOR_FILTER_LOW
            elif fv > 0.1:
                filter_text = f"HP"
                color = COLOR_FILTER_HIGH
            else:
                filter_text = "—"
                color = COLOR_FILTER_CENTER
            
            label = self.font.render(filter_text, True, color)
            self.screen.blit(label, (filter_x, controls_y + 22))

        # Kit label
        if self.audio_output and hasattr(self.audio_output, 'kit_name'):
            kit_text = self.font_small.render(f"Kit: {self.audio_output.kit_name}", True, COLOR_TEXT)
            self.screen.blit(kit_text, (20, controls_y + 50))
        
        # BPM indicator
        if self.camera and hasattr(self.camera, 'hand_detector') and self.camera.hand_detector:
            ind_x = WINDOW_WIDTH - 120
            if self.camera.hand_detector.hands_detected:
                pygame.draw.circle(self.screen, (0, 255, 0), (ind_x, controls_y + 30), 8)
                label = self.font_small.render("BPM", True, (0, 255, 0))
            else:
                pygame.draw.circle(self.screen, (80, 80, 80), (ind_x, controls_y + 30), 8)
                label = self.font_small.render("BPM", True, (100, 100, 100))
            self.screen.blit(label, (ind_x + 15, controls_y + 23))
        
        return btn_rect, pygame.Rect(slider_x, controls_y + 15, slider_width, 30)
    
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
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
        btn_rect, slider_rect = self._control_rects
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
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
                elif slider_rect.collidepoint(pos):
                    self._dragging_bpm_slider = True
                    self._update_bpm_from_pos(pos[0], slider_rect)
                else:
                    # Check mini grid
                    cell = self._get_mini_cell_from_pos(pos)
                    if cell:
                        self.grid.toggle(cell[0], cell[1])
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging_bpm_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self._dragging_bpm_slider:
                    self._update_bpm_from_pos(event.pos[0], slider_rect)
            
            elif event.type == pygame.KEYDOWN:
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
                        print("🔄 Board recalibration requested")
                elif event.key == pygame.K_m:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self._manual_board_mode = not self._manual_board_mode
                        self._manual_points = []
                        if self._manual_board_mode:
                            self.camera.board_detector.clear_manual_corners()
                            print("Manual board mode: ON")
                        else:
                            print("Manual board mode: OFF")
                elif event.key == pygame.K_BACKSPACE:
                    if self.camera and hasattr(self.camera, 'board_detector'):
                        self.camera.board_detector.clear_manual_corners()
                    self._manual_board_mode = False
                    self._manual_points = []
                    print("Manual board cleared")
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
                # Filter controls
                elif event.key == pygame.K_LEFT:
                    self._adjust_filter(-0.1)
                elif event.key == pygame.K_RIGHT:
                    self._adjust_filter(0.1)
                elif event.key == pygame.K_0:
                    if self.audio_output:
                        self.audio_output.filter_value = 0.0
                elif event.key == pygame.K_LEFTBRACKET:
                    if self.audio_output:
                        new_kit = self.audio_output.cycle_kit(-1)
                        print(f"Kit: {new_kit}")
                elif event.key == pygame.K_RIGHTBRACKET:
                    if self.audio_output:
                        new_kit = self.audio_output.cycle_kit(1)
                        print(f"Kit: {new_kit}")
        
        return True
    
    def _update_bpm_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update BPM from slider position."""
        ratio = (x - slider_rect.x) / slider_rect.width
        ratio = max(0, min(1, ratio))
        self.sequencer.bpm = int(30 + ratio * (300 - 30))
    
    def _adjust_filter(self, delta: float):
        """Adjust filter value."""
        if self.audio_output and hasattr(self.audio_output, 'filter_value'):
            new_val = self.audio_output.filter_value + delta
            self.audio_output.filter_value = max(-1, min(1, new_val))
    
    def draw(self):
        """Draw everything."""
        self.screen.fill(COLOR_BG)
        
        self._draw_camera_feed()
        self._draw_mini_grid()
        self._control_rects = self._draw_controls()
    
    def run(self):
        """Main loop."""
        self._control_rects = (pygame.Rect(0, 0, 0, 0),) * 2
        
        while self.running:
            if not self.handle_events():
                self.running = False
                break
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
