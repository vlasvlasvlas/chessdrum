"""
Grid model for the 8x8 Reversi board.
Maps to a 4-instrument x 16-step drum sequencer.
"""
import numpy as np

# Cell states
EMPTY = 0
WHITE = 1  # Strong accent (velocity 127)
BLACK = 2  # Weak accent (velocity 80)

# Instrument mapping (row indices for each instrument)
# Using upper half (rows 0-3) as first 8 steps
# Using lower half (rows 4-7) as steps 9-16
INSTRUMENTS = ['hihat', 'clap', 'snare', 'kick']


class Grid:
    """8x8 grid representing the Reversi/chess board."""
    
    def __init__(self):
        # 8x8 numpy array, all cells start empty
        self.cells = np.zeros((8, 8), dtype=np.int8)
    
    def toggle(self, row: int, col: int) -> int:
        """
        Toggle cell state: EMPTY -> WHITE -> BLACK -> EMPTY
        Returns the new state.
        """
        if 0 <= row < 8 and 0 <= col < 8:
            current = self.cells[row, col]
            new_state = (current + 1) % 3
            self.cells[row, col] = new_state
            return new_state
        return EMPTY
    
    def set_cell(self, row: int, col: int, state: int):
        """Set a specific cell state."""
        if 0 <= row < 8 and 0 <= col < 8:
            self.cells[row, col] = state
    
    def get_cell(self, row: int, col: int) -> int:
        """Get cell state."""
        if 0 <= row < 8 and 0 <= col < 8:
            return self.cells[row, col]
        return EMPTY
    
    def clear(self):
        """Clear all cells."""
        self.cells.fill(EMPTY)
    
    def get_pattern(self) -> list:
        """
        Convert 8x8 grid to 4-instrument x 16-step pattern.
        
        Mapping:
        - Row 0 (cols 0-7) = Hi-Hat steps 1-8
        - Row 4 (cols 0-7) = Hi-Hat steps 9-16
        - Row 1 (cols 0-7) = Clap steps 1-8
        - Row 5 (cols 0-7) = Clap steps 9-16
        - Row 2 (cols 0-7) = Snare steps 1-8
        - Row 6 (cols 0-7) = Snare steps 9-16
        - Row 3 (cols 0-7) = Kick steps 1-8
        - Row 7 (cols 0-7) = Kick steps 9-16
        
        Returns:
            List of 4 lists (one per instrument), each with 16 values (0, 1, or 2)
        """
        pattern = []
        for i in range(4):  # 4 instruments
            upper_row = self.cells[i, :]       # Steps 1-8 (row 0-3)
            lower_row = self.cells[i + 4, :]   # Steps 9-16 (row 4-7)
            instrument_pattern = np.concatenate([upper_row, lower_row]).tolist()
            pattern.append(instrument_pattern)
        return pattern
    
    def get_step_column(self, step: int) -> tuple:
        """
        Get the row and column indices for a given step (0-15).
        Used for highlighting the playhead.
        
        Returns:
            (is_upper_half, col) - is_upper_half is True for steps 0-7
        """
        if step < 8:
            return (True, step)
        else:
            return (False, step - 8)
