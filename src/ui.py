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
        self._dragging_cutoff_slider = False
        self._dragging_resonance_slider = False
        self._manual_board_mode = False
        self._manual_points = []
        self._last_frame_shape = None
        self._library_browser_open = False
        self._library_mode = "sound"
        self._library_index = 0
        self._pattern_index = 0

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
    
    def handle_events(self) -> bool:
        """Handle pygame events."""
        btn_rect, slider_rect, cutoff_rect, resonance_rect, hand_toggle_rect = self._control_rects
        
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
                    # Check mini grid
                    cell = self._get_mini_cell_from_pos(pos)
                    if cell:
                        self.grid.toggle(cell[0], cell[1])
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging_bpm_slider = False
                self._dragging_cutoff_slider = False
                self._dragging_resonance_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
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
                elif event.key == pygame.K_h:
                    if self.camera and hasattr(self.camera, 'set_hand_bpm_enabled'):
                        self.camera.set_hand_bpm_enabled(not self.camera.hand_bpm_enabled)
                elif event.key == pygame.K_l:
                    self._library_browser_open = not self._library_browser_open
                    if self._library_browser_open:
                        self._library_mode = "sound"
                        self._library_index = 0
                        self._pattern_index = 0
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
        kit_id = lib.get("kit")
        if kit_id and self.audio_output:
            self.audio_output.set_kit(kit_id)
        self._library_browser_open = False
    
    def draw(self):
        """Draw everything."""
        self.screen.fill(COLOR_BG)
        
        self._draw_camera_feed()
        self._draw_mini_grid()
        self._draw_library_browser()
        self._control_rects = self._draw_controls()
    
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
        
        pygame.quit()
