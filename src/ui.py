"""
Pygame-based UI for ChessDrum.
Camera-first layout with small grid overlay.
"""
import pygame
import numpy as np
import cv2
from typing import Optional

from grid import Grid, EMPTY, WHITE, BLACK, INSTRUMENTS
from sequencer import Sequencer

# Colors
COLOR_BG = (30, 30, 35)
COLOR_BOARD_LIGHT = (139, 195, 74)   # Green like Reversi board
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
        
        # Register callback for step changes
        self.sequencer.on_step_change = self._on_step_change
    
    def _on_step_change(self, step: int):
        """Called by sequencer when step changes."""
        self._highlighted_step = step
    
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
            return

        # Create surface from frame (RGB)
        # Flip mainly handled in camera capture loop, but Pygame needs it right way up
        # CV2 is BGR, Pygame needs RGB. and Pygame surface is (W,H)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        
        # Scale to fit window width? Or keep aspect ratio?
        # Current UI assumes WINDOW_WIDTH x CAMERA_HEIGHT
        frame_surface = pygame.transform.scale(frame_surface, (WINDOW_WIDTH, CAMERA_HEIGHT))
        self.screen.blit(frame_surface, (0, 0))
        
        # --- DRAW OVERLAY ---
        if self.camera and hasattr(self.camera, 'board_detector') and self.camera.board_detector.corners is not None:
            corners = self.camera.board_detector.corners
            
            # Scale corners to UI space
            # Camera frame size vs UI size
            h, w = frame.shape[:2]
            scale_x = WINDOW_WIDTH / w
            scale_y = CAMERA_HEIGHT / h
            
            # Draw Legend
            legend_font = self.font
            # Top-Left corner of board
            tl = (corners[0][0] * scale_x, corners[0][1] * scale_y)
            # Bottom-Left corner
            bl = (corners[3][0] * scale_x, corners[3][1] * scale_y)
            
            # Orientation Helper text
            # Draw near top edge
            top_mid_x = (corners[0][0] + corners[1][0]) / 2 * scale_x
            top_mid_y = (corners[0][1] + corners[1][1]) / 2 * scale_y
            label_surf = legend_font.render("RED = TOP (Steps 1-8)", True, (255, 50, 50))
            self.screen.blit(label_surf, (top_mid_x - label_surf.get_width()//2, top_mid_y - 30))
            
            # Draw near bottom edge
            bot_mid_x = (corners[3][0] + corners[2][0]) / 2 * scale_x
            bot_mid_y = (corners[3][1] + corners[2][1]) / 2 * scale_y
            label_surf = legend_font.render("BLUE = BOTTOM (Steps 9-16)", True, (50, 50, 255))
            self.screen.blit(label_surf, (bot_mid_x - label_surf.get_width()//2, bot_mid_y + 10))
            
            # --- DRAW PLAYHEAD ---
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
            
            # Draw Transparent Overlay
            overlay = pygame.Surface((WINDOW_WIDTH, CAMERA_HEIGHT), pygame.SRCALPHA)
            pygame.draw.polygon(overlay, (200, 200, 200, 100), ui_poly)  # White/Gray transparent
            pygame.draw.polygon(overlay, (255, 255, 0, 200), ui_poly, 3) # Yellow border
            self.screen.blit(overlay, (0, 0))

        # Camera Settings Overlay
        if self.camera:
            y_off = 10
            settings_font = self.font_small
            
            # Brightness
            b_text = f"Brightness (Q/A): {self.camera.brightness}"
            shad = settings_font.render(b_text, True, (0,0,0))
            txt = settings_font.render(b_text, True, (255,255,255))
            self.screen.blit(shad, (12, y_off+2)); self.screen.blit(txt, (10, y_off))
            y_off += 25
            
            # Contrast
            c_text = f"Contrast (W/S): {self.camera.contrast:.1f}"
            shad = settings_font.render(c_text, True, (0,0,0))
            txt = settings_font.render(c_text, True, (255,255,255))
            self.screen.blit(shad, (12, y_off+2)); self.screen.blit(txt, (10, y_off))
            
        
    def _draw_mini_grid(self):
        """Draw small grid overlay in bottom right corner of camera view."""
        # Position in bottom right of camera area
        base_x = WINDOW_WIDTH - MINI_SIZE - MINI_MARGIN
        base_y = CAMERA_HEIGHT - MINI_SIZE - MINI_MARGIN
        
        # Semi-transparent background
        bg_surface = pygame.Surface((MINI_SIZE + 4, MINI_SIZE + 4), pygame.SRCALPHA)
        bg_surface.fill((0, 0, 0, 180))
        self.screen.blit(bg_surface, (base_x - 2, base_y - 2))
        
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
        
        # Hands indicator
        if self.camera and hasattr(self.camera, 'hand_detector') and self.camera.hand_detector:
            ind_x = WINDOW_WIDTH - 120
            if self.camera.hand_detector.hands_detected:
                pygame.draw.circle(self.screen, (0, 255, 0), (ind_x, controls_y + 30), 8)
                label = self.font_small.render("Hands", True, (0, 255, 0))
            else:
                pygame.draw.circle(self.screen, (80, 80, 80), (ind_x, controls_y + 30), 8)
                label = self.font_small.render("Hands", True, (100, 100, 100))
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
                elif event.key == pygame.K_q:
                # Brightness UP
                    if self.camera: self.camera.brightness = min(100, self.camera.brightness + 5)
                elif event.key == pygame.K_a:
                # Brightness DOWN
                    if self.camera: self.camera.brightness = max(-100, self.camera.brightness - 5)
                elif event.key == pygame.K_w:
                # Contrast UP
                    if self.camera: self.camera.contrast = min(3.0, self.camera.contrast + 0.1)
                elif event.key == pygame.K_s:
                # Contrast DOWN
                    if self.camera: self.camera.contrast = max(0.1, self.camera.contrast - 0.1)
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
