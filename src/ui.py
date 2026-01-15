"""
Pygame-based UI for ChessDrum.
Displays the 8x8 grid with pieces and playhead.
Includes rotation slider for filter control.
"""
import pygame
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
COLOR_PLAYHEAD_BG = (255, 193, 7, 80)
COLOR_TEXT = (255, 255, 255)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 149, 237)
COLOR_BUTTON_ACTIVE = (65, 105, 225)
COLOR_SLIDER_BG = (60, 60, 65)
COLOR_SLIDER_FILL = (70, 130, 180)
COLOR_FILTER_LOW = (255, 100, 100)   # Red for low-pass
COLOR_FILTER_HIGH = (100, 200, 255)  # Blue for high-pass
COLOR_FILTER_CENTER = (100, 255, 100)  # Green for center

# Layout constants
CELL_SIZE = 60
BOARD_MARGIN = 40
BOARD_SIZE = CELL_SIZE * 8
CONTROLS_HEIGHT = 140  # Increased for second row of controls
WINDOW_WIDTH = BOARD_SIZE + BOARD_MARGIN * 2
WINDOW_HEIGHT = BOARD_SIZE + BOARD_MARGIN * 2 + CONTROLS_HEIGHT

# Instrument labels
INSTRUMENT_LABELS = ['HH', 'CP', 'SD', 'KK']  # Short labels for each row


class UI:
    """Pygame-based UI for the drum sequencer."""
    
    def __init__(self, grid: Grid, sequencer: Sequencer, audio_output=None, config=None):
        self.grid = grid
        self.sequencer = sequencer
        self.audio_output = audio_output  # For filter control
        self.config = config
        
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
        self.font = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 36)
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
    
    def _get_cell_rect(self, row: int, col: int) -> pygame.Rect:
        """Get the rectangle for a cell."""
        x = BOARD_MARGIN + col * CELL_SIZE
        y = BOARD_MARGIN + row * CELL_SIZE
        return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
    
    def _get_cell_from_pos(self, pos: tuple) -> Optional[tuple]:
        """Get (row, col) from mouse position, or None if outside board."""
        x, y = pos
        col = (x - BOARD_MARGIN) // CELL_SIZE
        row = (y - BOARD_MARGIN) // CELL_SIZE
        if 0 <= row < 8 and 0 <= col < 8:
            return (row, col)
        return None
    
    def _draw_board(self):
        """Draw the chessboard pattern."""
        for row in range(8):
            for col in range(8):
                rect = self._get_cell_rect(row, col)
                # Alternate colors like a chessboard
                if (row + col) % 2 == 0:
                    color = COLOR_BOARD_LIGHT
                else:
                    color = COLOR_BOARD_DARK
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
    
    def _draw_pieces(self):
        """Draw all pieces on the board."""
        for row in range(8):
            for col in range(8):
                state = self.grid.get_cell(row, col)
                if state != EMPTY:
                    rect = self._get_cell_rect(row, col)
                    center = rect.center
                    radius = CELL_SIZE // 2 - 6
                    
                    # Draw piece
                    color = COLOR_PIECE_WHITE if state == WHITE else COLOR_PIECE_BLACK
                    pygame.draw.circle(self.screen, color, center, radius)
                    pygame.draw.circle(self.screen, COLOR_PIECE_BORDER, center, radius, 2)
    
    def _draw_playhead(self):
        """Draw the playhead indicator."""
        step = self._highlighted_step
        is_upper, col = self.grid.get_step_column(step)
        
        # Highlight the column
        x = BOARD_MARGIN + col * CELL_SIZE
        
        if is_upper:
            # Upper half (rows 0-3)
            y = BOARD_MARGIN
            height = CELL_SIZE * 4
        else:
            # Lower half (rows 4-7)
            y = BOARD_MARGIN + CELL_SIZE * 4
            height = CELL_SIZE * 4
        
        # Semi-transparent highlight
        highlight_surface = pygame.Surface((CELL_SIZE, height), pygame.SRCALPHA)
        highlight_surface.fill((255, 193, 7, 60))
        self.screen.blit(highlight_surface, (x, y))
        
        # Border
        pygame.draw.rect(self.screen, COLOR_PLAYHEAD, (x, y, CELL_SIZE, height), 3)
    
    def _draw_labels(self):
        """Draw instrument labels on the side."""
        for i, label in enumerate(INSTRUMENT_LABELS):
            # Upper half
            y1 = BOARD_MARGIN + i * CELL_SIZE + CELL_SIZE // 2
            text1 = self.font_small.render(label, True, COLOR_TEXT)
            self.screen.blit(text1, (8, y1 - text1.get_height() // 2))
            
            # Lower half
            y2 = BOARD_MARGIN + (i + 4) * CELL_SIZE + CELL_SIZE // 2
            text2 = self.font_small.render(label, True, COLOR_TEXT)
            self.screen.blit(text2, (8, y2 - text2.get_height() // 2))
        
        # Step numbers on top
        for col in range(8):
            x = BOARD_MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            text = self.font_small.render(str(col + 1), True, COLOR_TEXT)
            self.screen.blit(text, (x - text.get_width() // 2, 20))
        
        # "Bar 1" / "Bar 2" labels
        bar1_text = self.font_small.render("Bar 1", True, COLOR_TEXT)
        bar2_text = self.font_small.render("Bar 2", True, COLOR_TEXT)
        self.screen.blit(bar1_text, (WINDOW_WIDTH - 50, BOARD_MARGIN + 2 * CELL_SIZE))
        self.screen.blit(bar2_text, (WINDOW_WIDTH - 50, BOARD_MARGIN + 6 * CELL_SIZE))
    
    def _draw_controls(self):
        """Draw BPM slider, play/stop button, and filter slider."""
        controls_y = BOARD_MARGIN * 2 + BOARD_SIZE + 10
        
        # Play/Stop button
        btn_rect = pygame.Rect(BOARD_MARGIN, controls_y, 80, 40)
        btn_color = COLOR_BUTTON_ACTIVE if self.sequencer.is_playing else COLOR_BUTTON
        pygame.draw.rect(self.screen, btn_color, btn_rect, border_radius=5)
        btn_text = "Stop" if self.sequencer.is_playing else "Play"
        text = self.font.render(btn_text, True, COLOR_TEXT)
        text_rect = text.get_rect(center=btn_rect.center)
        self.screen.blit(text, text_rect)
        
        # Clear button
        clear_rect = pygame.Rect(BOARD_MARGIN + 90, controls_y, 80, 40)
        pygame.draw.rect(self.screen, COLOR_BUTTON, clear_rect, border_radius=5)
        clear_text = self.font.render("Clear", True, COLOR_TEXT)
        self.screen.blit(clear_text, clear_text.get_rect(center=clear_rect.center))
        
        # BPM Slider
        slider_x = BOARD_MARGIN + 200
        slider_width = 200
        bpm_slider_rect = pygame.Rect(slider_x, controls_y + 15, slider_width, 10)
        pygame.draw.rect(self.screen, COLOR_SLIDER_BG, bpm_slider_rect, border_radius=5)
        
        # Slider fill
        bpm_ratio = (self.sequencer.bpm - 30) / (300 - 30)
        fill_width = int(slider_width * bpm_ratio)
        fill_rect = pygame.Rect(slider_x, controls_y + 15, fill_width, 10)
        pygame.draw.rect(self.screen, COLOR_SLIDER_FILL, fill_rect, border_radius=5)
        
        # Slider handle
        handle_x = slider_x + fill_width
        pygame.draw.circle(self.screen, COLOR_TEXT, (handle_x, controls_y + 20), 8)
        
        # BPM label
        bpm_label = self.font.render(f"BPM: {self.sequencer.bpm}", True, COLOR_TEXT)
        self.screen.blit(bpm_label, (slider_x + slider_width + 20, controls_y + 10))
        
        # === FILTER/ROTATION SLIDER (second row) ===
        filter_y = controls_y + 50
        filter_slider_x = BOARD_MARGIN
        filter_slider_width = WINDOW_WIDTH - BOARD_MARGIN * 2 - 100
        
        # Label
        filter_label = self.font.render("🔄 Rotation Filter:", True, COLOR_TEXT)
        self.screen.blit(filter_label, (filter_slider_x, filter_y))
        
        # Slider background with gradient effect
        filter_slider_y = filter_y + 25
        filter_slider_rect = pygame.Rect(filter_slider_x, filter_slider_y, filter_slider_width, 14)
        
        # Draw gradient background (red -> green -> blue)
        for i in range(filter_slider_width):
            ratio = i / filter_slider_width
            if ratio < 0.5:
                # Red to Green
                r = int(255 * (1 - ratio * 2))
                g = int(200 * ratio * 2)
                b = int(100 * (1 - ratio * 2))
            else:
                # Green to Blue
                r = int(100 * (ratio - 0.5) * 2)
                g = int(200 * (1 - (ratio - 0.5) * 2))
                b = int(255 * (ratio - 0.5) * 2)
            pygame.draw.line(self.screen, (r, g, b), 
                           (filter_slider_x + i, filter_slider_y),
                           (filter_slider_x + i, filter_slider_y + 14))
        
        # Center marker
        center_x = filter_slider_x + filter_slider_width // 2
        pygame.draw.line(self.screen, COLOR_TEXT, 
                        (center_x, filter_slider_y - 3),
                        (center_x, filter_slider_y + 17), 2)
        
        # Handle position based on filter value
        if self.audio_output and hasattr(self.audio_output, 'filter_value'):
            filter_val = self.audio_output.filter_value
        else:
            filter_val = 0.0
        
        handle_ratio = (filter_val + 1) / 2  # Convert -1..+1 to 0..1
        handle_x = filter_slider_x + int(filter_slider_width * handle_ratio)
        
        # Handle
        pygame.draw.circle(self.screen, COLOR_TEXT, (handle_x, filter_slider_y + 7), 10)
        pygame.draw.circle(self.screen, COLOR_BG, (handle_x, filter_slider_y + 7), 6)
        
        # Filter value label
        if filter_val < -0.1:
            filter_text = f"LP: {abs(filter_val):.1f}"
            text_color = COLOR_FILTER_LOW
        elif filter_val > 0.1:
            filter_text = f"HP: {filter_val:.1f}"
            text_color = COLOR_FILTER_HIGH
        else:
            filter_text = "OFF"
            text_color = COLOR_FILTER_CENTER
        
        filter_val_label = self.font.render(filter_text, True, text_color)
        self.screen.blit(filter_val_label, (filter_slider_x + filter_slider_width + 10, filter_slider_y))
        
        return btn_rect, clear_rect, pygame.Rect(slider_x, controls_y, slider_width, 40), \
               pygame.Rect(filter_slider_x, filter_slider_y - 5, filter_slider_width, 24)
    
    def handle_events(self) -> bool:
        """Handle pygame events. Returns False if should quit."""
        btn_rect, clear_rect, bpm_slider_rect, filter_slider_rect = self._control_rects
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                
                # Check buttons
                if btn_rect.collidepoint(pos):
                    self.sequencer.toggle()
                elif clear_rect.collidepoint(pos):
                    self.grid.clear()
                elif bpm_slider_rect.collidepoint(pos):
                    self._dragging_bpm_slider = True
                    self._update_bpm_from_pos(pos[0], bpm_slider_rect)
                elif filter_slider_rect.collidepoint(pos):
                    self._dragging_filter_slider = True
                    self._update_filter_from_pos(pos[0], filter_slider_rect)
                else:
                    # Check board
                    cell = self._get_cell_from_pos(pos)
                    if cell:
                        self.grid.toggle(cell[0], cell[1])
            
            elif event.type == pygame.MOUSEBUTTONUP:
                self._dragging_bpm_slider = False
                self._dragging_filter_slider = False
            
            elif event.type == pygame.MOUSEMOTION:
                if self._dragging_bpm_slider:
                    self._update_bpm_from_pos(event.pos[0], bpm_slider_rect)
                elif self._dragging_filter_slider:
                    self._update_filter_from_pos(event.pos[0], filter_slider_rect)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.sequencer.toggle()
                elif event.key == pygame.K_c:
                    self.grid.clear()
                elif event.key == pygame.K_ESCAPE:
                    return False
                # Arrow keys for filter
                elif event.key == pygame.K_LEFT:
                    self._adjust_filter(-0.1)
                elif event.key == pygame.K_RIGHT:
                    self._adjust_filter(0.1)
                elif event.key == pygame.K_DOWN:
                    self._adjust_filter(-0.05)  # Fine adjustment
                elif event.key == pygame.K_UP:
                    self._adjust_filter(0.05)
                elif event.key == pygame.K_0:
                    # Reset filter to center
                    if self.audio_output and hasattr(self.audio_output, 'filter_value'):
                        self.audio_output.filter_value = 0.0
        
        return True
    
    def _update_bpm_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update BPM based on slider position."""
        ratio = (x - slider_rect.x) / slider_rect.width
        ratio = max(0, min(1, ratio))
        self.sequencer.bpm = int(30 + ratio * (300 - 30))
    
    def _update_filter_from_pos(self, x: int, slider_rect: pygame.Rect):
        """Update filter value based on slider position."""
        if self.audio_output and hasattr(self.audio_output, 'filter_value'):
            ratio = (x - slider_rect.x) / slider_rect.width
            ratio = max(0, min(1, ratio))
            # Convert 0..1 to -1..+1
            self.audio_output.filter_value = (ratio * 2) - 1
    
    def _adjust_filter(self, delta: float):
        """Adjust filter value by delta."""
        if self.audio_output and hasattr(self.audio_output, 'filter_value'):
            new_val = self.audio_output.filter_value + delta
            self.audio_output.filter_value = max(-1, min(1, new_val))
    
    def draw(self):
        """Draw everything."""
        self.screen.fill(COLOR_BG)
        
        self._draw_board()
        self._draw_playhead()
        self._draw_pieces()
        self._draw_labels()
        self._control_rects = self._draw_controls()
    
    def run(self):
        """Main loop."""
        self._control_rects = (pygame.Rect(0, 0, 0, 0),) * 4  # Placeholder
        
        while self.running:
            if not self.handle_events():
                self.running = False
                break
            
            self.draw()
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS
        
        pygame.quit()
