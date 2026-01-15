#!/usr/bin/env python3
"""
ChessDrum - Main Entry Point

A drum sequencer where an 8x8 chessboard grid controls a 4-instrument,
16-step drum pattern.

Usage:
    python3 src/main.py          # Uses settings from config.json
    python3 src/main.py --midi   # Force MIDI mode

Controls:
    - Click on cells to place/toggle pieces (white → black → empty)
    - Space: Play/Stop
    - C: Clear all pieces
    - ESC: Quit
    - Arrows: Adjust filter
    - 0: Reset filter
"""
import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from grid import Grid
from sequencer import Sequencer
from ui import UI
from config import config


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ChessDrum - Chess-based Drum Machine')
    parser.add_argument('--midi', action='store_true', 
                        help='Use MIDI output instead of audio samples')
    parser.add_argument('--no-filter', action='store_true',
                        help='Disable filter effect')
    args = parser.parse_args()
    
    # Get project name from config
    project_name = config.get('project', 'name', default='ChessDrum')
    
    print("=" * 50)
    print(f"  {project_name} 🎵")
    print("=" * 50)
    print()
    print("Controls:")
    print("  • Click cells to toggle pieces (white → black → empty)")
    print("  • White = loud (velocity 127)")
    print("  • Black = soft (velocity 80)")
    print("  • Space = Play/Stop")
    print("  • C = Clear")
    print("  • ESC = Quit")
    print("  • ←/→ = Adjust filter")
    print("  • 0 = Reset filter")
    print()
    print("Mapping:")
    print("  • Upper half (rows 1-4) = Steps 1-8")
    print("  • Lower half (rows 5-8) = Steps 9-16")
    print("  • HH=Hi-Hat, CP=Clap, SD=Snare, KK=Kick")
    print()
    
    # Initialize components
    grid = Grid()
    
    # Choose output mode (CLI args override config)
    use_midi = args.midi or config.get('midi', 'enabled', default=False)
    use_audio = not use_midi and config.get('audio', 'enabled', default=True)
    
    if use_midi:
        from midi_output import MidiOutput
        output = MidiOutput()
        print("Mode: MIDI output")
        audio_ref = None
    else:
        from audio_output import AudioOutput
        # Pass config for filter settings
        filter_config = {
            'filter': config.filter,
            'audio': config.audio
        }
        # Override filter enabled if --no-filter flag
        if args.no_filter:
            filter_config['filter']['enabled'] = False
        
        output = AudioOutput(config=filter_config)
        print("Mode: Audio samples")
        audio_ref = output
    
    sequencer = Sequencer(grid, output)
    
    # Set default BPM from config
    sequencer.bpm = config.get('sequencer', 'default_bpm', default=120)
    
    # Pass audio_output to UI for filter control
    ui = UI(grid, sequencer, audio_output=audio_ref, config=config)
    
    # Run the app
    try:
        ui.run()
    finally:
        sequencer.stop()
        output.close()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
