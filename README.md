# ChessDrum 🎵♟️

A drum sequencer where an 8x8 chessboard controls a 4-instrument, 16-step drum pattern. Place pieces on the board to create beats!

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
│  ⚪ White piece = LOUD hit (127)    │
│  ⚫ Black piece = soft hit (80)     │
│  Empty = silence                    │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run
python3 src/main.py
```

## 🎛️ Controls

| Input | Action |
|-------|--------|
| **Click cell** | Toggle: empty → white → black → empty |
| **Space** | Play/Stop |
| **C** | Clear board |
| **← / →** | Filter: left=dark, right=bright |
| **0** | Reset filter to center |
| **ESC** | Quit |

## 🔊 Synth Filter

The rotation slider controls a classic lowpass/highpass filter:

```
  ◀── DARK ────── CENTER ────── BRIGHT ──▶
      (80Hz)      (neutral)      (12kHz)
        LP          OFF            HP
```

- **Left**: Lowpass filter, muffled/dark sound
- **Center**: No filter, natural sound
- **Right**: Highpass filter, bright/thin sound
- **Resonance**: Adds "squelchy" peak at cutoff frequency

## ⚙️ Configuration

All settings in `config.json`:

```json
{
  "audio": {
    "enabled": true,
    "sample_rate": 44100
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
  "sequencer": {
    "default_bpm": 120
  }
}
```

### Options

| Setting | Description |
|---------|-------------|
| `audio.enabled` | Use built-in synth sounds |
| `midi.enabled` | Output MIDI to DAW |
| `filter.min_freq` | Lowest cutoff (Hz) at left position |
| `filter.max_freq` | Highest cutoff (Hz) at right position |
| `filter.resonance` | Q factor (1=flat, 3+=resonant) |

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

## 📁 Project Structure

```
chessdrum/
├── config.json          # All settings
├── requirements.txt     # Dependencies
├── src/
│   ├── main.py          # Entry point
│   ├── config.py        # Config loader
│   ├── grid.py          # 8x8 board model
│   ├── sequencer.py     # Playback engine
│   ├── audio_output.py  # Synth + filter
│   ├── midi_output.py   # MIDI output
│   └── ui.py            # Pygame interface
└── README.md
```

## 🗺️ Roadmap

- [x] Virtual sequencer with GUI
- [x] Built-in synth sounds
- [x] Synth filter with resonance
- [x] JSON configuration
- [ ] **Camera detection** (OpenCV)
  - Detect physical chessboard
  - Detect pieces (white/black/empty)
  - Board rotation → filter control
  - Distance/tilt → BPM control?

## 📜 License

MIT
