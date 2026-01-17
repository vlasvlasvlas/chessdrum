# ChessDrum 🎵♟️

A drum sequencer where an 8x8 chessboard controls a 4-instrument, 16-step drum pattern. Place pieces on the board to create beats!

## 🎬 Demo

https://github.com/vlasvlasvlas/chessdrum/assets/xxxxx/chessdrum-demo.mp4

*ChessDrum in action: Camera detection with real-time board tracking, mixer controls, and live performance (79 seconds).*

**Alternative:** [Download video directly](docs/images/chessdrum-demo.mp4)

## 📸 Screenshots

![ChessDrum Screenshot](docs/images/chessdrum-screenshot.png)
*Camera mode with board detection overlay and real-time controls.*

## ✨ Highlights

### Core Features
- **Virtual & Camera modes**: Use physical board or interactive grid
- **4-Channel Mixer** (NEW): Individual volume, mute, delay & reverb per channel
- **6 built-in sound kits**: classic, real, dnb, ethnic, 8bit, bizarre
- **Sound + pattern libraries**: JSON-based with in-app browser (press L)
- **Synth filter**: Real-time cutoff/resonance control
- **MIDI output**: Send to DAW

### Camera Detection (NEW)
- **Producer-consumer architecture**: 30 FPS capture & detection on separate threads
- **Adaptive board detection**: Dynamic sizing (200-800px), stability scoring
- **Manual calibration**: 4-click corner override
- **Real-time tuning**: Sensitivity/threshold/brightness/contrast
- **Hand BPM control**: Two open hands to adjust tempo (20-220 BPM)

### UI/UX (NEW)
- **Performance metrics**: Live FPS overlay (capture/detection)
- **Calibration status**: Visual indicator (calibrating/ready)
- **Notification system**: Toast messages for actions
- **Board corner visualization**: Green overlay showing detected area
- **Virtual mode**: Large clickable grid when no camera available

### Stability & Logging (NEW)
- **Rotating log files**: 10MB max, 5 backups (logs/chessdrum.log)
- **Color-coded console**: DEBUG/INFO/WARNING/ERROR levels
- **Graceful fallbacks**: MediaPipe optional, stable on ARM64 macOS
- **Version pinning**: opencv-python==4.8.1.78, mediapipe==0.10.9

## 🎮 How It Works

```
┌─────────────────────────────────────┐
│           8x8 CHESSBOARD            │
├─────────────────────────────────────┤
│  Row 0-3: Steps 1-8   (Bar 1)      │
│  Row 4-7: Steps 9-16  (Bar 2)      │
├─────────────────────────────────────┤
│  Each row = 1 instrument:           │
│    Row 0,4 = Hi-Hat                 │
│    Row 1,5 = Clap                   │
│    Row 2,6 = Snare                  │
│    Row 3,7 = Kick                   │
├─────────────────────────────────────┤
│  ⚫ Black piece = hit (velocity 80)  │
│  Empty = silence                    │
│  (Uses BLACK pieces for contrast)   │
└─────────────────────────────────────┘
```

## � Requirements

- **Python**: 3.9+ (tested on 3.9.6)
- **OS**: macOS, Linux, Windows
  - ARM64 macOS (M1/M2): Fully supported with pinned versions
- **Camera**: Optional (0 or 1 USB/built-in camera for detection mode)
- **Dependencies**: See [requirements.txt](requirements.txt)
  - `pygame` (GUI + audio)
  - `opencv-python==4.8.1.78` (camera, version pinned)
  - `mediapipe==0.10.9` (hand tracking, version pinned, optional)
  - `numpy`, `scipy` (audio processing)
  - `mido`, `python-rtmidi` (MIDI output)

## �🚀 Quick Start

### Using the Launcher (Recommended)
```bash
# Interactive launcher with camera detection
./run.sh
```

The launcher will:
- ✅ Activate virtual environment automatically
- 🔍 Detect available cameras
- 🎮 Let you choose virtual or camera mode
- 📹 List all camera options if multiple found
- 📝 Ask about verbose logging

### Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Virtual mode (no camera)
python3 src/main.py

# Camera mode
python3 src/main.py --camera --cam 0

# Camera mode with verbose logging
python3 src/main.py --camera --cam 0 --verbose
```

## 🎛️ Controls

### Basic Controls
| Input | Action |
|-------|--------|
| **Click cell** | Toggle: empty → black → empty |
| **Space** | Play/Stop |
| **C** | Clear board |
| **ESC** | Quit |
| **X** | **Toggle Mixer Panel** (NEW) |

### Mixer Controls (NEW - Press X to open)
| Input | Action |
|-------|--------|
| **Vertical faders** | Adjust channel volume (drag with mouse) |
| **M buttons** | Mute/unmute individual channels (click) |
| **DLY knobs** | Delay wet/dry mix per channel (drag circular) |
| **REV knobs** | Reverb wet/dry mix per channel (drag circular) |

Channels: **1.HH** (Hi-Hat), **2.CP** (Clap), **3.SD** (Snare), **4.KK** (Kick)

### Sound & Filter Controls
| Input | Action |
|-------|--------|
| **← / →** | Previous/next sound kit |
| **↑ / ↓** | Master volume |
| **0** | Reset filter (cutoff/resonance) |
| **L** | Library browser (sound + pattern presets) |
| **H** | Toggle hand BPM detection |

Filter is controlled by the on-screen Cutoff and Resonance sliders (below BPM).
Drag the sliders with the mouse to shape the tone.
Use **H** or the **Hands BPM** button to disable hand BPM while moving pieces.

### Camera Controls (when --camera mode)
| Input | Action |
|-------|--------|
| **Q / A** | Brightness +5 / -5 |
| **W / S** | Contrast +0.1 / -0.1 |
| **E** | Reset brightness/contrast |
| **R** | **Recalibrate board baseline** (use when lighting changes) |
| **D** | **Toggle debug mode** (shows warped board analysis) |
| **M** | **Manual board mode** (click 4 corners: TL,TR,BR,BL) |
| **Backspace** | **Clear manual board** (return to auto detect) |
| **Right click** | **Undo last corner** (manual mode) |
| **1-9** | **Sensitivity** (1=strict, 5=balanced, 9=sensitive) |
| **T / G** | **Dark threshold** +5 / -5 (adjust piece detection) |
| **2 open hands** | **BPM control** (20-220, only when both hands are open and held briefly) |
| **H** | **Hand BPM toggle** (disable hand detection when placing pieces) |

## 🔊 Audio Engine & Effects

### Synth Filter
Use the two sliders under BPM:

- **Cutoff**: Lowpass cutoff frequency (lower = darker, higher = brighter)
- **Resonance**: Boosts the cutoff area for a sharper, more "squelchy" sound

### Channel Mixer (NEW - Press X)

The mixer provides **per-channel control** for all 4 instruments:

#### Volume Faders
- **Vertical faders** for precise volume control (0-100%)
- Real-time visual feedback with fill indicator
- Separate control for HH, CP, SD, KK

#### Mute Buttons
- **Click [M]** to mute/unmute individual channels
- Visual feedback: Red = muted, Gray = active
- Instant notification on state change

#### Delay Effect
- **Delay knob** per channel (0-100% wet/dry mix)
- 250ms delay time with 40% feedback
- Circular knob with arc indicator showing amount
- Creates echo/repeat effects

#### Reverb Effect
- **Reverb knob** per channel (0-100% wet/dry mix)
- Convolution reverb with exponential decay impulse
- Adds space and depth to individual sounds
- Independent control per instrument

#### FX Signal Chain
```
Input Sample
    ↓
Volume Control
    ↓
Mute Check
    ↓
Delay Processing (if > 0%)
    ↓
Reverb Processing (if > 0%)
    ↓
Output
```

All effects are processed in **real-time** during playback with zero latency.

## 🧰 Sound Kits

Switch kits with **← / →** (audio mode only). Available kits:

- **classic**: original synthetic kit (balanced).
- **real**: punchy acoustic-style kit.
- **dnb**: tight, fast drum-and-bass vibe.
- **ethnic**: softer, percussive, more 'organic'.
- **8bit**: chiptune/bitcrushed kit.
- **bizarre**: experimental, noisy, and weird.
- **hybrid_*:** built-in drum kits with HH/CP converted to notes.

All kits are generated in code (no external samples). You can also add your own WAV or synth kits in `libraries/sound_libraries.json`.

## 📚 Pattern Libraries

Pattern presets live in `libraries/pattern_libraries.json` and can be applied from the in-app library browser (press **L**).
Each library can include up to 4 patterns and can optionally pick a sound kit.

Included presets: Chroma C Pulse, D Shuffle, E Steps, F Bounce, G Drive (drums + notes).
See [docs/LIBRARIES.md](docs/LIBRARIES.md) for examples and templates.

### Library Browser

Press **L** to open the browser:

- **Tab** switches between Sound and Pattern lists
- **Up/Down** selects a library
- **Left/Right** switches patterns (pattern mode)
- **Enter** applies the selection

## ⚙️ Configuration

All settings in `config.json`:

```json
{
  "audio": {
    "enabled": true,
    "kit": "classic",
    "sample_rate": 44100,
    "buffer_size": 512,
    "volume": 1.0,
    "channels": 32,
    "instrument_gain": {
      "kick": 1.0,
      "snare": 1.2,
      "hihat": 1.0,
      "clap": 1.0
    }
  },
  "camera": {
    "enabled": false,
    "device_id": 1,
    "brightness": 0,
    "contrast": 1.0,
    "manual_corners": null,
    "hand_bpm_enabled": true,
    "bpm_min_distance": 80,
    "bpm_max_distance": 880,
    "detection_sensitivity": 0.5,
    "dark_threshold": 50
  },
  "midi": {
    "enabled": false,
    "port_name": "ChessDrum"
  },
  "filter": {
    "enabled": true,
    "min_freq": 80,
    "max_freq": 12000,
    "resonance": 3.0
  },
  "mixer": {
    "channel_volumes": [1.0, 1.0, 1.0, 1.0],
    "channel_mutes": [false, false, false, false],
    "channel_delay": [0.0, 0.0, 0.0, 0.0],
    "channel_reverb": [0.0, 0.0, 0.0, 0.0],
    "delay_time_ms": 250,
    "delay_feedback": 0.4
  },
  "sequencer": {
    "default_bpm": 120
  },
  "libraries": {
    "sound_file": "libraries/sound_libraries.json",
    "pattern_file": "libraries/pattern_libraries.json"
  }
}
```

### Options

| Setting | Description |
|---------|-------------|
| `audio.enabled` | Use built-in synth sounds |
| `audio.kit` | Sound kit id from `libraries/sound_libraries.json` (e.g., `classic`, `real`, `hybrid_c_major`) |
| `audio.buffer_size` | Audio buffer size (latency vs stability) |
| `audio.volume` | Master volume (0.0 - 1.0) |
| `audio.channels` | Mixer channels (overlap headroom for longer samples) |
| `audio.instrument_gain` | Per-instrument gain map (`kick`, `snare`, `hihat`, `clap`) |
| `mixer.channel_volumes` | Initial volume per channel [HH, CP, SD, KK] (0.0-1.0) (NEW) |
| `mixer.channel_mutes` | Initial mute state per channel (true/false) (NEW) |
| `mixer.channel_delay` | Initial delay mix per channel (0.0-1.0) (NEW) |
| `mixer.channel_reverb` | Initial reverb mix per channel (0.0-1.0) (NEW) |
| `mixer.delay_time_ms` | Delay time in milliseconds (NEW) |
| `mixer.delay_feedback` | Delay feedback amount (0.0-1.0) (NEW) |
| `midi.enabled` | Output MIDI to DAW |
| `camera.brightness` | Image brightness offset |
| `camera.contrast` | Image contrast multiplier |
| `camera.manual_corners` | Manual board corners (normalized TL,TR,BR,BL) |
| `camera.hand_bpm_enabled` | Start with hand BPM detection enabled |
| `camera.bpm_min_distance` | Hand distance → minimum BPM |
| `camera.bpm_max_distance` | Hand distance → maximum BPM |
| `filter.min_freq` | Lowest cutoff (Hz) for the Cutoff slider |
| `filter.max_freq` | Highest cutoff (Hz) for the Cutoff slider |
| `filter.resonance` | Default resonance value |
| `libraries.sound_file` | Sound library JSON file |
| `libraries.pattern_file` | Pattern library JSON file |

## 🎹 MIDI Mode

```bash
python3 src/main.py --midi
```

Creates virtual MIDI port "ChessDrum" for your DAW.

| Instrument | MIDI Note |
|------------|-----------|
| Kick | 36 |
| Snare | 38 |
| Clap | 39 |
| Hi-Hat | 42 |

## 📷 Camera Detection Mode

Run with `--camera` flag to use physical chessboard detection:

```bash
python3 src/main.py --camera
```

### 🎛️ Camera Tuning Tool
Use the built-in tool to adjust brightness/contrast and set manual board corners:

```bash
python3 test_camera.py
```

Tool shortcuts:
- **Q/A** brightness, **W/S** contrast
- **Left click** 4 corners (TL, TR, BR, BL), **Right click** undo last
- **C** clear corners, **Space** snapshot, **ESC/X** quit
- **P** print a config snippet with `manual_corners`

### ✅ Camera Workflow (recommended)
1. Start: `python3 src/main.py --camera`
2. Press **D** to see debug warped view
3. Adjust **Q/A** and **W/S** until squares are clear
4. If auto-detect fails, press **M** and click 4 corners
5. Show **two open hands** to set BPM, then lower hands to lock it

### 🔧 Troubleshooting Detection Issues

#### Board Not Detected
- Ensure good lighting (even, not too bright/dark)
- Board should fill 15-50% of camera view
- Try different camera angles (straight overhead is best)
- Press **R** to recalibrate after moving camera/board
- Use **M** to set manual board corners if auto detection fails

#### False Positives / Missing Pieces
1. **Adjust Sensitivity (1-9 keys)**:
   - Press **5** for balanced (recommended start)
   - Press **3-4** if too many false positives
   - Press **6-7** if pieces not detected

2. **Adjust Dark Threshold (T/G keys)**:
   - Press **T** to increase (stricter, fewer false positives)
   - Press **G** to decrease (more permissive, detect lighter pieces)
   - Watch on-screen value: aim for 40-60 range

3. **Optimize Image Quality**:
   - Press **Q/A** to adjust brightness
   - Press **W/S** to adjust contrast
   - Press **D** to see debug view with detection values

4. **Recalibrate**:
   - Press **R** after any camera/board movement
   - Press **R** when lighting conditions change

#### Detection Features
- ✅ **Black piece detection**: Optimized for contrast
- ✅ **Real-time adjustable thresholds**: Keys 1-9, T/G
- ✅ **On-screen parameters**: See sensitivity and threshold values
- ✅ **Adaptive algorithm**: Auto-calibrates to lighting
- ✅ **Temporal filtering**: 5/7 frames required (reduces flicker)
- ✅ **Debug mode**: Visual feedback for troubleshooting
- ✅ **Enhanced playhead overlay**: Yellow highlight on physical board
- ✅ **Manual board corners**: 4-click fallback when auto detect fails
- ✅ **Two open hands**: BPM control gated to avoid accidental changes

### On-Screen Indicators
- **Performance Metrics (NEW)**: Capture FPS / Detection FPS (camera mode)
- **Calibration Status (NEW)**: ⏳ Calibrating... / ✓ Calibrated (green check)
- **Hand Detection (NEW)**: 🤚 Hands: X / 🎯 BPM ACTIVE (two open hands)
- **Board Corners (NEW)**: Green polygon overlay showing detected board area
- **Mixer Open**: Panel slides in from right side (200px width)
- **Sensitivity (1-9)**: Shows current detection sensitivity (0.1-0.9)
- **Dark Thresh (T/G)**: Shows threshold value (20-100)
- **Pieces: X**: Count of detected black pieces
- **DEBUG: ON**: Yellow when debug mode active
- **Manual board**: Shows when manual corners are enabled

### Config Options
```json
{
  "camera": {
    "enabled": false,
    "device_id": 1,
    "debug_mode": false,
    "brightness": 0,
    "contrast": 1.0,
    "manual_corners": null,
    "bpm_min_distance": 80,
    "bpm_max_distance": 880,
    "detection_sensitivity": 0.5,
    "dark_threshold": 50,
    "board_detection": true,
    "hand_detection": true
  }
}
```

### Recommended Settings by Environment

**Bright Room (Daylight)**:
- Sensitivity: **4-5** (keys 4-5)
- Dark Threshold: **55-65** (press T multiple times)
- Brightness: **-10 to 0** (press A)

**Normal Indoor Lighting**:
- Sensitivity: **5-6** (keys 5-6)
- Dark Threshold: **45-55** (default range)
- Brightness: **0** (default)

**Dim Lighting**:
- Sensitivity: **6-7** (keys 6-7)
- Dark Threshold: **35-45** (press G multiple times)
- Brightness: **+10 to +20** (press Q)

## 🧠 Technical Details

### Architecture (Refactored in FASE 1)
- **Modular vision package**: `src/vision/` with separate modules
  - `hand_detector.py`: MediaPipe hand tracking (250 lines)
  - `board_detector.py`: Adaptive board detection (600+ lines)
  - `camera_controller.py`: Producer-consumer orchestration (345 lines)
- **Producer-consumer pattern**: Camera capture (29.9 FPS) and detection (30.0 FPS) on separate threads
- **Queue-based communication**: `queue.Queue(maxsize=2)` prevents frame backlog

### Board Detection
- **Adaptive sizing**: 200-800px warp based on board area (15-50% of frame)
- **Stability scoring**: Smooths corner jitter, adapts filtering strength
- **Otsu thresholding**: Auto-adjusts to lighting conditions
- **Manual corners**: 4-point override stored as normalized coordinates (TL,TR,BR,BL)
- **Calibration persistence**: `vision/camera_controller.py` saves/loads board baseline

### Piece Detection
- **Per-cell center sampling**: 3x3 grid average for robustness
- **Dark-pixel ratio + baseline deviation**: Adaptive to lighting changes
- **Temporal filter**: 5/7 frame consensus reduces flicker
- **Real-time tunable**: Sensitivity (1-9) and dark threshold (T/G keys)

### Hand Detection & BPM
- **MediaPipe Hands**: 21-landmark tracking per hand
- **Two-hand gating**: Only adjusts BPM when both hands open (prevents accidental changes)
- **Distance mapping**: Hand distance (pixels) → BPM (20-220)
- **Graceful fallback**: Continues without MediaPipe if not available (ARM64 compatible)

### Audio Engine
- **Pygame mixer**: 44.1kHz, 512 sample buffer, 32 channels
- **Real-time FX processing**: Channel mixer with delay/reverb
  - **Delay**: 250ms with 40% feedback using `collections.deque` buffers
  - **Reverb**: Convolution with exponential decay impulse responses
  - **Signal chain**: Volume → Mute → Delay → Reverb
- **SciPy filter**: Butterworth lowpass with resonance boost
- **JSON library system**: Sound kits + pattern presets

### Logging System (FASE 0)
- **Rotating file handler**: 10MB max, 5 backups in `logs/`
- **Color-coded console**: Custom formatter with ANSI colors
- **Multi-level**: DEBUG/INFO/WARNING/ERROR
- **Module-specific**: Each module has its own logger

### UI System
- **Pygame**: 800x600 base window (1000x600 with mixer open)
- **Dynamic resize**: Window expands when mixer opens (X key)
- **Notification system**: Toast messages (3s auto-fade, bottom-left)
- **Dual mode**: Camera view with mini-grid OR large virtual grid
- **Performance overlays**: FPS, calibration, hand status

## 📁 Project Structure

```
chessdrum/
├── run.sh               # Interactive launcher (NEW)
├── config.json          # All settings
├── requirements.txt     # Dependencies (pinned versions)
├── docs/
│   ├── images/          # README screenshots
│   └── LIBRARIES.md     # How to create new libraries
├── libraries/           # Sound + pattern library JSON files
├── logs/                # Rotating log files (NEW)
│   └── chessdrum.log    # Main log (10MB max, 5 backups)
├── src/
│   ├── main.py          # Entry point
│   ├── logger.py        # Logging system (NEW)
│   ├── config.py        # Config loader
│   ├── grid.py          # 8x8 board model
│   ├── sequencer.py     # Playback engine
│   ├── audio_output.py  # Synth + filter + mixer FX (NEW)
│   ├── midi_output.py   # MIDI output
│   ├── ui.py            # Pygame interface + mixer panel (NEW)
│   └── vision/          # Vision module (NEW - FASE 1)
│       ├── __init__.py
│       ├── hand_detector.py      # MediaPipe hand tracking
│       ├── board_detector.py     # Adaptive board detection
│       └── camera_controller.py  # Producer-consumer pattern
└── README.md
```


## 🔮 Future Ideas

### Audio & Mixing
- Per-step velocity levels (multiple piece colors)
- Per-instrument reverb/delay send amounts
- Compressor/limiter on master output
- External WAV sample loader
- Pattern chaining & automation

### Sequencing
- Probability steps (% chance to trigger)
- Ratcheting (sub-divisions per step)
- Conditional fills (every N bars)
- Save/load pattern banks
- Export patterns to MIDI file
- Multiple pages/patterns per session

### Camera & Detection
- Hand gestures to toggle sections or change kit
- Multi-board support (2+ boards for longer patterns)
- Colored piece detection for velocity/variation
- Auto-save calibration per board ID

### Melodic Extensions
- Toggle melodic mode (pitch per row)
- Second board for melody/bass
- Chord mode (detect multiple pieces → chord)
- Scale snapping (pentatonic, major, minor, etc.)

### UI/UX
- Dark/light theme toggle
- Customizable color schemes per kit
- Fullscreen mode
- Pattern visualization (waveform/spectrum)
- Tutorial mode with hints

## 📜 License

MIT
