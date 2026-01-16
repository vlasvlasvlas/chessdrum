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
    - Click on cells to place/toggle pieces (empty → black → empty)
    - Space: Play/Stop
    - C: Clear all pieces
    - ESC: Quit
    - Left/Right: Switch sound kit
    - Up/Down: Master volume
    - 0: Reset filter (cutoff/resonance)
    - Two open hands: Control BPM by distance
"""
import sys
import os
import argparse
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize logging FIRST (before other imports)
from logger import init_logging, get_logger
logger = get_logger(__name__)

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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging (DEBUG level)')
    args = parser.parse_args()
    
    # Initialize logging system
    init_logging(verbose=args.verbose)
    logger.info("ChessDrum starting...")
    
    # Get project name from config
    project_name = config.get('project', 'name', default='ChessDrum')
    
    print("=" * 50)
    print(f"  {project_name} 🎵")
    print("=" * 50)
    print()
    
    # Determine camera usage first
    use_camera = args.camera or config.get('camera', 'enabled', default=False)
    logger.info(f"Camera mode: {use_camera}")
    
    print("Controls:")
    print("  • Click cells to toggle pieces (empty → black → empty)")
    print("  • Black = hit (velocity 80)")
    print("  • Space = Play/Stop")
    print("  • C = Clear")
    print("  • ESC = Quit")
    print("  • ←/→ = Switch sound kit (audio only)")
    print("  • ↑/↓ = Master volume (audio only)")
    print("  • 0 = Reset filter (cutoff/resonance)")
    print("  • Cutoff/Resonance sliders are on-screen below BPM")
    print("  • L = Library browser (sound + pattern presets)")
    print("  • H = Toggle hand BPM detection")
    if use_camera:
        print()
        print("Camera Controls:")
        print("  • Q/A = Brightness ±5")
        print("  • W/S = Contrast ±0.1")
        print("  • E = Reset brightness/contrast")
        print("  • 1-9 = Sensitivity (1=strict, 5=balanced, 9=sensitive)")
        print("  • T/G = Dark threshold ±5 (higher=stricter)")
        print("  • R = Recalibrate board (use when lighting changes)")
        print("  • D = Toggle debug mode (shows warped board)")
        print("  • M = Manual board mode (click 4 corners)")
        print("  • Backspace = Clear manual board")
        print("  • Two open hands = BPM (20-220)")
        print("  • H = Toggle hand BPM detection")
    print()
    print("Mapping:")
    print("  • Upper half (rows 1-4) = Steps 1-8")
    print("  • Lower half (rows 5-8) = Steps 9-16")
    print("  • HH=Hi-Hat, CP=Clap, SD=Snare, KK=Kick")
    print()
    
    logger.debug("Initializing grid...")
    # Initialize components
    grid = Grid()
    
    # Choose output mode (CLI args override config)
    use_midi = args.midi or config.get('midi', 'enabled', default=False)
    logger.info(f"MIDI mode: {use_midi}")
    
    if use_midi:
        logger.debug("Importing MIDI output...")
        from midi_output import MidiOutput
        output = MidiOutput()
        print("Mode: MIDI output")
        logger.info("MIDI output initialized")
        audio_ref = None
    else:
        logger.debug("Importing Audio output...")
        from audio_output import AudioOutput
        # Pass config for filter settings
        filter_config = {
            'filter': config.filter,
            'audio': config.audio,
            'libraries': config.libraries
        }
        if args.no_filter:
            filter_config['filter']['enabled'] = False
            logger.info("Filter disabled by command line argument")
        
        output = AudioOutput(config=filter_config)
        print("Mode: Audio samples")
        logger.info("Audio output initialized")
        audio_ref = output
    
    logger.debug("Initializing sequencer...")
    sequencer = Sequencer(grid, output)
    sequencer.bpm = config.get('sequencer', 'default_bpm', default=120)
    logger.info(f"Sequencer initialized at {sequencer.bpm} BPM")
    
    # Initialize camera if requested (use_camera already defined earlier)
    camera_controller = None
    
    if use_camera:
        try:
            logger.debug("Importing vision module...")
            from vision import CameraController
            
            camera_config = config.camera
            # Use --cam argument if provided, otherwise use config
            device_id = args.cam if args.cam is not None else camera_config.get('device_id', 0)
            logger.info(f"Initializing camera on device {device_id}")
            
            camera_controller = CameraController(
                device_id=device_id,
                config=camera_config
            )
            
            # Set up callbacks
            def on_bpm_change(bpm):
                logger.debug(f"BPM changed to {bpm} via hand gesture")
                sequencer.bpm = bpm
            
            def on_pieces_change(pieces):
                # Update grid with detected pieces
                # pieces is 8x8 numpy array: 0=empty, 1=black, 2=white
                if pieces is not None:
                    piece_count = np.count_nonzero(pieces)
                    logger.debug(f"Pieces detected: {piece_count}")
                    for row in range(8):
                        for col in range(8):
                            grid.set_cell(row, col, int(pieces[row, col]))
            
            camera_controller.on_bpm_change = on_bpm_change
            camera_controller.on_pieces_change = on_pieces_change
            
            if camera_controller.start():
                print("Camera: Enabled (two open hands for BPM)")
                logger.info("Camera started successfully")
            else:
                camera_controller = None
                print("Camera: Failed to start")
                logger.error("Camera failed to start")
        except ImportError as e:
            print(f"Camera: Not available ({e})")
            logger.error(f"Camera import failed: {e}", exc_info=True)
            camera_controller = None
        except Exception as e:
            print(f"Camera: Error ({e})")
            logger.error(f"Camera initialization error: {e}", exc_info=True)
            camera_controller = None
    
    # Pass audio_output and camera to UI
    logger.debug("Initializing UI...")
    ui = UI(grid, sequencer, audio_output=audio_ref, config=config, camera=camera_controller)
    logger.info("UI initialized")
    
    # FASE 2: Show camera error in UI if failed
    if use_camera and camera_controller is None:
        ui.show_notification("⚠ Camera failed to start", (255, 150, 50))
    
    # Run the app
    logger.info("Starting main loop...")
    try:
        ui.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.critical(f"Fatal error in main loop: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down...")
        sequencer.stop()
        output.close()
        if camera_controller:
            camera_controller.stop()
        logger.info("Cleanup complete")
        print("\n👋 Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Catch any unhandled exceptions at the top level
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        print(f"\n❌ Fatal error: {e}")
        print("Check logs/chessdrum.log for details")
        sys.exit(1)
