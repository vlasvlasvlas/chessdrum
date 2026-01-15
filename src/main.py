#!/usr/bin/env python3
"""
ChessDrum - Main Entry Point

A drum sequencer where an 8x8 chessboard grid controls a 4-instrument,
16-step drum pattern.

Usage:
    python3 src/main.py              # Virtual mode (no camera)
    python3 src/main.py --camera     # Camera mode (detect board + hands)
    python3 src/main.py --midi       # MIDI output mode

Controls:
    - Click on cells to place/toggle pieces (white → black → empty)
    - Space: Play/Stop
    - C: Clear all pieces
    - ESC: Quit
    - Arrows: Adjust filter
    - 0: Reset filter (or use camera rotation)
    - Show 2 hands: Control BPM by distance
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
    parser.add_argument('--camera', action='store_true',
                        help='Enable camera for board/hand detection')
    parser.add_argument('--cam', type=int, default=None,
                        help='Camera device ID (0=built-in, 1=external, etc.)')
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
        if args.no_filter:
            filter_config['filter']['enabled'] = False
        
        output = AudioOutput(config=filter_config)
        print("Mode: Audio samples")
        audio_ref = output
    
    sequencer = Sequencer(grid, output)
    sequencer.bpm = config.get('sequencer', 'default_bpm', default=120)
    
    # Initialize camera if requested
    camera_controller = None
    use_camera = args.camera or config.get('camera', 'enabled', default=False)
    
    if use_camera:
        try:
            from camera import CameraController
            
            camera_config = config.camera
            # Use --cam argument if provided, otherwise use config
            device_id = args.cam if args.cam is not None else camera_config.get('device_id', 0)
            
            camera_controller = CameraController(
                device_id=device_id,
                config=camera_config
            )
            
            # Set up callbacks
            def on_bpm_change(bpm):
                sequencer.bpm = bpm
            
            def on_rotation_change(rotation):
                if audio_ref:
                    audio_ref.filter_value = rotation
            
            def on_pieces_change(pieces):
                # Update grid with detected pieces
                # pieces is 8x8 numpy array: 0=empty, 1=white, 2=black
                if pieces is not None:
                    for row in range(8):
                        for col in range(8):
                            grid.set_cell(row, col, int(pieces[row, col]))
            
            camera_controller.on_bpm_change = on_bpm_change
            camera_controller.on_rotation_change = on_rotation_change
            camera_controller.on_pieces_change = on_pieces_change
            
            if camera_controller.start():
                print("Camera: Enabled (show 2 hands for BPM)")
            else:
                camera_controller = None
                print("Camera: Failed to start")
        except ImportError as e:
            print(f"Camera: Not available ({e})")
            camera_controller = None
    
    # Pass audio_output and camera to UI
    ui = UI(grid, sequencer, audio_output=audio_ref, config=config, camera=camera_controller)
    
    # Run the app
    try:
        ui.run()
    finally:
        sequencer.stop()
        output.close()
        if camera_controller:
            camera_controller.stop()
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    main()
