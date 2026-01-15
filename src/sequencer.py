"""
Sequencer engine for ChessDrum.
Handles timing and playback of the drum pattern.
"""
import time
import threading
from typing import Callable, Optional, Protocol


class SoundOutput(Protocol):
    """Protocol for sound output (MIDI or Audio)."""
    def trigger(self, instrument: str, cell_state: int) -> None: ...


class Sequencer:
    """
    Drum sequencer that plays a 16-step pattern.
    """
    
    def __init__(self, grid, output: SoundOutput):
        self.grid = grid
        self.output = output
        
        # Import here to avoid circular imports
        from grid import INSTRUMENTS
        self.instruments = INSTRUMENTS
        
        # Timing
        self._bpm = 120
        self._steps = 16
        self._current_step = 0
        
        # Playback state
        self._playing = False
        self._thread: Optional[threading.Thread] = None
        
        # Callback for UI updates
        self.on_step_change: Optional[Callable[[int], None]] = None
    
    @property
    def bpm(self) -> int:
        return self._bpm
    
    @bpm.setter
    def bpm(self, value: int):
        self._bpm = max(30, min(300, value))
    
    @property
    def step_duration(self) -> float:
        """Duration of one step in seconds (sixteenth note)."""
        # BPM is quarter notes per minute
        # 16th note = 1/4 of a quarter note
        quarter_duration = 60.0 / self._bpm
        return quarter_duration / 4
    
    @property
    def current_step(self) -> int:
        return self._current_step
    
    @property
    def is_playing(self) -> bool:
        return self._playing
    
    def start(self):
        """Start playback."""
        if self._playing:
            return
        
        self._playing = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop playback."""
        self._playing = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._current_step = 0
        if self.on_step_change:
            self.on_step_change(self._current_step)
    
    def toggle(self):
        """Toggle play/stop."""
        if self._playing:
            self.stop()
        else:
            self.start()
    
    def _playback_loop(self):
        """Main playback loop running in a separate thread."""
        while self._playing:
            # Get current pattern
            pattern = self.grid.get_pattern()
            
            # Trigger sounds for current step
            for i, instrument in enumerate(self.instruments):
                cell_state = pattern[i][self._current_step]
                self.output.trigger(instrument, cell_state)
            
            # Notify UI
            if self.on_step_change:
                self.on_step_change(self._current_step)
            
            # Wait for next step
            time.sleep(self.step_duration)
            
            # Advance step
            self._current_step = (self._current_step + 1) % self._steps
    
    def set_step(self, step: int):
        """Manually set the current step (for scrubbing)."""
        self._current_step = step % self._steps
        if self.on_step_change:
            self.on_step_change(self._current_step)
