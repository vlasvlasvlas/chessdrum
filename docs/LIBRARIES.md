# Library Guide

This guide explains how to create new sound libraries (kits) and pattern libraries (preset grids).
All files are plain JSON so they are easy to share, version, and extend.

## File Locations

- Sound libraries: `libraries/sound_libraries.json`
- Pattern libraries: `libraries/pattern_libraries.json`

Paths are configured in `config.json` under `libraries`.

```
"libraries": {
  "sound_file": "libraries/sound_libraries.json",
  "pattern_file": "libraries/pattern_libraries.json"
}
```

## Sound Libraries (Kits)

Each sound library defines one kit with metadata and a `type`.
The UI library browser (press `L`) shows these entries and lets you switch kits.

### Common Fields

- `id`: unique id (used in config and pattern libraries)
- `name`: display name
- `description`: short description
- `author`: who made it
- `date`: creation date (YYYY-MM-DD)
- `active`: true/false (inactive entries are hidden from cycling)
- `type`: `builtin`, `hybrid`, `synth`, or `samples`

### Type: builtin

Uses the built-in generator by `id` (`classic`, `real`, `dnb`, `ethnic`, `8bit`, `bizarre`).

```
{
  "id": "classic",
  "name": "Classic",
  "type": "builtin",
  "active": true
}
```

### Type: hybrid

Starts from a built-in drum kit and replaces HH/CP (or any instrument) with notes.
This is the easiest way to mix drums + melody.

```
{
  "id": "hybrid_c_major",
  "name": "Hybrid C Major",
  "type": "hybrid",
  "base_kit": "classic",
  "notes": {
    "hihat": "C4",
    "clap": "E4"
  },
  "synth": {
    "waveform": "sine",
    "duration": 0.35,
    "decay": 6.0,
    "attack": 0.01
  }
}
```

### Type: synth

Pure synth-note kit. Each instrument becomes a pitched tone.

```
{
  "id": "custom_synth",
  "name": "Custom Synth",
  "type": "synth",
  "notes": {
    "kick": "C3",
    "snare": "G3",
    "hihat": "C4",
    "clap": "E4"
  },
  "synth": {
    "waveform": "triangle",
    "duration": 0.4,
    "decay": 5.0,
    "attack": 0.02
  }
}
```

Supported `notes` formats: `C4`, `D#4`, `Eb3`, `G3` (A4 = 440Hz).

Supported `synth.waveform` values:

- `sine`
- `square`
- `triangle`
- `saw`

### Type: samples

Loads WAV files from disk (relative to repo or absolute paths).

```
{
  "id": "custom_samples",
  "name": "My Samples",
  "type": "samples",
  "samples": {
    "kick": "samples/kick.wav",
    "snare": "samples/snare.wav",
    "hihat": "samples/hihat.wav",
    "clap": "samples/clap.wav"
  },
  "gain": {
    "snare": 1.2
  }
}
```

## Pattern Libraries (Presets)

Each pattern library is a collection of up to 4 patterns.
Patterns are 8x8 grids with 0/1/2 values (empty, normal hit, accent).

### Grid Mapping (important)

The grid follows the same 8x8 mapping as the main app:

- Row 0 = Hi-Hat (steps 1-8)
- Row 1 = Clap (steps 1-8)
- Row 2 = Snare (steps 1-8)
- Row 3 = Kick (steps 1-8)
- Row 4 = Hi-Hat (steps 9-16)
- Row 5 = Clap (steps 9-16)
- Row 6 = Snare (steps 9-16)
- Row 7 = Kick (steps 9-16)

Cell values:

- `0` = empty
- `1` = normal hit
- `2` = accent (louder)

```
{
  "id": "chroma_c_pulse",
  "name": "Chroma C Pulse",
  "description": "Straight pulse with C/E notes.",
  "author": "ChessDrum",
  "date": "2025-02-01",
  "active": true,
  "kit": "hybrid_c_major",
  "patterns": [
    {
      "name": "Pulse 1",
      "description": "Even notes with a steady backbeat.",
      "grid": [
        [1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1],
        [0,0,0,0,1,0,0,0],
        [1,0,0,0,1,0,0,0],
        [1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1],
        [0,0,0,0,1,0,0,0],
        [1,0,0,0,1,0,0,0]
      ]
    }
  ]
}
```

## Gain and Volume Tips

- Global volume: `audio.volume` in `config.json` (0.0 - 1.0)
- Per-instrument gain: `audio.instrument_gain`
- Per-kit gain: use `gain` inside a sound library entry

Example:

```
"gain": {
  "snare": 1.3,
  "kick": 1.0,
  "hihat": 0.9,
  "clap": 1.0
}
```

## Melody + Drums (2 tracks each)

Use a `hybrid` kit to keep snare/kick as drums and turn hi-hat/clap into notes.
That gives 2 drum tracks + 2 melodic tracks without changing the grid mapping.

## Active/Inactive Libraries

Set `active: false` to hide a library from cycling. The browser still shows it
but you cannot apply it until you set it back to `true`.

## Tips

- Keep `active: false` for experiments you want hidden.
- Use `hybrid` kits for drum + melody mixes (HH/CP as notes, SN/KK as drums).
- For louder snares, add `"snare": 1.2` in `audio.instrument_gain` or library `gain`.

## UI Controls

- Press `L` to open the library browser.
- Use Up/Down to select, Enter to apply.
- In Pattern mode, use Left/Right to switch patterns.
- Tab switches between Sound and Pattern lists.
