"""
MIDI output module for ChessDrum.
Sends drum notes to a virtual MIDI port.
"""
import mido
from mido import Message

from grid import EMPTY, BLACK, WHITE

# General MIDI Drum Map (Channel 10)
DRUM_NOTES = {
    'kick': 36,    # Bass Drum 1
    'snare': 38,   # Acoustic Snare
    'clap': 39,    # Hand Clap
    'hihat': 42,   # Closed Hi-Hat
}

# Velocity values for piece types
VELOCITY_WHITE = 127  # Strong accent
VELOCITY_BLACK = 80   # Soft accent


class MidiOutput:
    """Handles MIDI output for the drum sequencer."""
    
    def __init__(self, port_name: str = "ChessDrum"):
        self.port_name = port_name
        self.port = None
        self._open_port()
    
    def _open_port(self):
        """Try to open a virtual MIDI port."""
        try:
            # Try to open a virtual port (works on macOS/Linux)
            self.port = mido.open_output(self.port_name, virtual=True)
            print(f"✓ Virtual MIDI port '{self.port_name}' created")
        except Exception as e:
            print(f"⚠ Could not create virtual MIDI port: {e}")
            # Try to use an existing port
            available = mido.get_output_names()
            if available:
                self.port = mido.open_output(available[0])
                print(f"✓ Using existing MIDI port: {available[0]}")
            else:
                print("✗ No MIDI ports available")
                self.port = None
    
    def play_note(self, instrument: str, velocity: int = 127):
        """
        Send a drum note.
        
        Args:
            instrument: One of 'kick', 'snare', 'clap', 'hihat'
            velocity: MIDI velocity (0-127)
        """
        if self.port is None:
            return
        
        note = DRUM_NOTES.get(instrument, 36)
        
        # Note on
        msg_on = Message('note_on', note=note, velocity=velocity, channel=9)
        self.port.send(msg_on)
        
        # Note off (immediate for drums)
        msg_off = Message('note_off', note=note, velocity=0, channel=9)
        self.port.send(msg_off)
    
    def trigger(self, instrument: str, cell_state: int):
        """
        Trigger a drum sound based on cell state.
        
        Args:
            instrument: Instrument name
            cell_state: 0=empty (no sound), 1=black (soft), 2=white (loud)
        """
        if cell_state == EMPTY:
            return
        
        if cell_state == BLACK:
            velocity = VELOCITY_BLACK
        elif cell_state == WHITE:
            velocity = VELOCITY_WHITE
        else:
            return
        self.play_note(instrument, velocity)
    
    def close(self):
        """Close the MIDI port."""
        if self.port:
            self.port.close()
            self.port = None
    
    def __del__(self):
        self.close()
