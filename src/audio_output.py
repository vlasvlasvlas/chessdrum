"""
Audio output module for ChessDrum.
Classic synth-style filter with configurable cutoff and resonance.
"""
import os
import numpy as np
import pygame
from scipy import signal
from typing import Optional, Dict, Any

from libraries import load_sound_libraries

from grid import EMPTY, BLACK, WHITE

SAMPLE_RATE = 44100

NOTE_OFFSETS = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def _normalize(wave: np.ndarray, peak: float = 0.9) -> np.ndarray:
    max_val = float(np.max(np.abs(wave))) if wave.size else 0.0
    if max_val <= 1e-8:
        return wave
    return wave / max_val * peak


def _to_int16(wave: np.ndarray, peak: float = 0.9) -> np.ndarray:
    normalized = _normalize(wave, peak)
    return (np.clip(normalized, -1.0, 1.0) * 32767).astype(np.int16)


def _mix(*waves: np.ndarray) -> np.ndarray:
    if not waves:
        return np.array([], dtype=np.float32)
    max_len = max(len(w) for w in waves)
    mix = np.zeros(max_len, dtype=np.float32)
    for wave in waves:
        mix[:len(wave)] += wave
    return mix


def _soft_clip(wave: np.ndarray, drive: float = 1.0) -> np.ndarray:
    return np.tanh(wave * drive)


def _noise_burst(duration: float, decay: float, amp: float = 1.0) -> np.ndarray:
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    noise = np.random.uniform(-1, 1, samples).astype(np.float32)
    env = np.exp(-t * decay)
    return noise * env * amp


def _sine_sweep(duration: float, f_start: float, f_end: float, decay: float, amp: float = 1.0) -> np.ndarray:
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    f_start = max(1.0, f_start)
    f_end = max(1.0, f_end)
    freqs = np.exp(np.linspace(np.log(f_start), np.log(f_end), samples))
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    env = np.exp(-t * decay)
    return np.sin(phase) * env * amp


def _square_sweep(duration: float, f_start: float, f_end: float, decay: float, amp: float = 1.0) -> np.ndarray:
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    f_start = max(1.0, f_start)
    f_end = max(1.0, f_end)
    freqs = np.exp(np.linspace(np.log(f_start), np.log(f_end), samples))
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    env = np.exp(-t * decay)
    return np.sign(np.sin(phase)) * env * amp


def _highpass(wave: np.ndarray, cutoff: float) -> np.ndarray:
    nyquist = SAMPLE_RATE / 2
    norm = min(max(cutoff / nyquist, 0.01), 0.99)
    try:
        b, a = signal.butter(2, norm, btype='high')
        return signal.lfilter(b, a, wave).astype(np.float32)
    except Exception:
        return wave


def _bandpass(wave: np.ndarray, low: float, high: float) -> np.ndarray:
    nyquist = SAMPLE_RATE / 2
    low_n = min(max(low / nyquist, 0.01), 0.98)
    high_n = min(max(high / nyquist, low_n + 0.01), 0.99)
    try:
        b, a = signal.butter(2, [low_n, high_n], btype='band')
        return signal.lfilter(b, a, wave).astype(np.float32)
    except Exception:
        return wave


def _bitcrush(wave: np.ndarray, bits: int = 6, downsample: int = 4) -> np.ndarray:
    bits = max(2, bits)
    step = 2 ** (bits - 1)
    crushed = np.round(wave * step) / step
    if downsample > 1:
        crushed = crushed[::downsample]
        crushed = np.repeat(crushed, downsample)[:len(wave)]
    return crushed.astype(np.float32)


def _note_to_freq(note: str) -> Optional[float]:
    note = note.strip().upper()
    if not note:
        return None
    # Extract octave (last char(s))
    octave_str = ""
    for ch in reversed(note):
        if ch.isdigit():
            octave_str = ch + octave_str
        else:
            break
    if not octave_str:
        return None
    name = note[:-len(octave_str)].strip()
    if not name:
        return None
    semitone = NOTE_OFFSETS.get(name)
    if semitone is None:
        return None
    try:
        octave = int(octave_str)
    except ValueError:
        return None
    midi = (octave + 1) * 12 + semitone
    return 440.0 * (2 ** ((midi - 69) / 12))


def _waveform(phase: np.ndarray, kind: str) -> np.ndarray:
    kind = (kind or "sine").lower()
    if kind == "square":
        return np.sign(np.sin(phase))
    if kind == "saw":
        return 2.0 * (phase / (2 * np.pi) % 1.0) - 1.0
    if kind == "triangle":
        return 2.0 * np.abs(2.0 * (phase / (2 * np.pi) % 1.0) - 1.0) - 1.0
    return np.sin(phase)


def _synth_note(note: str, duration: float = 0.35, decay: float = 5.0,
                attack: float = 0.01, waveform: str = "sine", amp: float = 0.9) -> np.ndarray:
    freq = _note_to_freq(note)
    if freq is None:
        return np.array([], dtype=np.float32)
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    phase = 2 * np.pi * freq * t
    wave = _waveform(phase, waveform)
    env = np.exp(-t * decay)
    if attack > 0:
        attack_samples = max(1, int(SAMPLE_RATE * attack))
        ramp = np.linspace(0, 1, attack_samples)
        env[:attack_samples] *= ramp
    return wave * env * amp


class SynthFilter:
    """
    Classic synth-style resonant lowpass filter.

    Cutoff controls brightness; resonance boosts near cutoff.
    """
    
    def __init__(self, min_freq=80, max_freq=12000, resonance=2.0, order=4):
        """
        Args:
            min_freq: Minimum cutoff frequency in Hz
            max_freq: Maximum cutoff frequency in Hz
            resonance: Q factor (1.0 = flat, 2.0+ = resonant peak)
            order: Filter order (higher = steeper rolloff)
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resonance = resonance
        self.order = order
    
    def get_default_cutoff(self) -> float:
        """Geometric mean of min/max for neutral cutoff."""
        return float(np.sqrt(self.min_freq * self.max_freq))

    def apply(self, audio_data: np.ndarray, cutoff: Optional[float] = None,
              resonance: Optional[float] = None) -> np.ndarray:
        """
        Apply lowpass filter with explicit cutoff/resonance.
        """
        if cutoff is None:
            cutoff = self.get_default_cutoff()
        if resonance is None:
            resonance = self.resonance

        cutoff = max(self.min_freq, min(self.max_freq, float(cutoff)))
        resonance = max(0.1, float(resonance))

        # Convert to float
        audio_float = audio_data.astype(np.float32) / 32767.0

        # Normalize for scipy
        nyquist = SAMPLE_RATE / 2
        normalized = min(max(cutoff / nyquist, 0.01), 0.99)

        try:
            # Lowpass filter
            b, a = signal.butter(self.order, normalized, btype='low')
            filtered = signal.filtfilt(b, a, audio_float)

            # Add resonance (boost at cutoff frequency)
            if resonance > 1.0 and len(audio_float) > 50:
                q = min(20.0, max(0.5, resonance * 2.0))
                try:
                    b_peak, a_peak = signal.iirpeak(normalized, Q=q)
                    peak = signal.filtfilt(b_peak, a_peak, audio_float)
                    amount = (resonance - 1.0) * 0.9
                    filtered = filtered + peak * amount
                except Exception:
                    bw = max(0.05, 0.2 / resonance)
                    bp_low = max(0.01, normalized * (1 - bw))
                    bp_high = min(0.99, normalized * (1 + bw))
                    if bp_low < bp_high:
                        b_bp, a_bp = signal.butter(2, [bp_low, bp_high], btype='band')
                        resonant = signal.filtfilt(b_bp, a_bp, audio_float)
                        amount = (resonance - 1.0) * 0.7
                        filtered = filtered + resonant * amount

            # Soft clip to prevent distortion
            filtered = np.tanh(filtered * 1.2) / 1.2

            return (np.clip(filtered, -1, 1) * 32767).astype(np.int16)

        except Exception:
            return audio_data


def generate_kick(duration=0.4):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    freq = 150 * np.exp(-t * 10) + 50
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    envelope = np.exp(-t * 8)
    wave = np.sin(phase) * envelope
    click_samples = int(0.01 * SAMPLE_RATE)
    if click_samples > 0:
        click = np.sin(2 * np.pi * 1000 * t[:click_samples]) * np.exp(-t[:click_samples] * 100)
        wave[:click_samples] += click * 0.5
    return (wave * 32767 * 0.8).astype(np.int16)


def generate_snare(duration=0.28):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    noise = np.random.uniform(-1, 1, samples)
    noise_env = np.exp(-t * 15)
    tone = np.sin(2 * np.pi * 200 * t)
    tone_env = np.exp(-t * 20)
    wave = noise * noise_env * 0.5 + tone * tone_env * 0.5
    return (wave * 32767 * 0.7).astype(np.int16)


def generate_hihat(duration=0.12):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    noise = np.random.uniform(-1, 1, samples)
    envelope = np.exp(-t * 30)
    wave = noise * envelope
    return (wave * 32767 * 0.5).astype(np.int16)


def generate_clap(duration=0.22):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    wave = np.zeros(samples)
    for i in range(4):
        delay = int(i * 0.01 * SAMPLE_RATE)
        burst_len = min(int(0.02 * SAMPLE_RATE), samples - delay)
        if burst_len > 0:
            burst_t = np.linspace(0, 0.02, burst_len, False)
            burst = np.random.uniform(-1, 1, burst_len) * np.exp(-burst_t * 50)
            wave[delay:delay + burst_len] += burst
    envelope = np.exp(-t * 12)
    wave = wave * envelope
    return (wave * 32767 * 0.6 / 4).astype(np.int16)


def _kit_classic():
    return {
        'kick': generate_kick(),
        'snare': generate_snare(),
        'hihat': generate_hihat(),
        'clap': generate_clap(),
    }


def _kit_real():
    kick = _mix(
        _sine_sweep(0.45, 140, 50, decay=7, amp=1.0),
        _noise_burst(0.06, decay=50, amp=0.18),
        _sine_sweep(0.03, 900, 200, decay=50, amp=0.35),
    )
    kick = _soft_clip(kick, 1.2)

    snare_t = np.linspace(0, 0.32, int(SAMPLE_RATE * 0.32), False)
    snare_tone = np.sin(2 * np.pi * 180 * snare_t) * np.exp(-snare_t * 12)
    snare_noise = _noise_burst(0.32, decay=16, amp=0.8)
    snare_noise = _highpass(snare_noise, 900)
    snare = _soft_clip(snare_tone * 0.4 + snare_noise, 1.1)

    hihat = _noise_burst(0.15, decay=45, amp=0.6)
    hihat = _highpass(hihat, 7000)

    clap = np.zeros(int(SAMPLE_RATE * 0.26), dtype=np.float32)
    for i, amp in enumerate([0.7, 0.5, 0.35, 0.25]):
        offset = int(i * 0.012 * SAMPLE_RATE)
        burst = _noise_burst(0.06, decay=35, amp=amp)
        end = min(len(clap), offset + len(burst))
        clap[offset:end] += burst[:end - offset]
    clap = _highpass(clap, 1200)

    return {
        'kick': _to_int16(kick, 0.9),
        'snare': _to_int16(snare, 0.85),
        'hihat': _to_int16(hihat, 0.6),
        'clap': _to_int16(clap, 0.7),
    }


def _kit_dnb():
    kick = _mix(
        _sine_sweep(0.33, 180, 45, decay=10, amp=1.1),
        _sine_sweep(0.022, 1200, 250, decay=55, amp=0.5),
        _noise_burst(0.04, decay=60, amp=0.15),
    )
    kick = _soft_clip(kick, 1.4)

    snare_t = np.linspace(0, 0.28, int(SAMPLE_RATE * 0.28), False)
    snare_tone = np.sin(2 * np.pi * 220 * snare_t) * np.exp(-snare_t * 18)
    snare_noise = _noise_burst(0.28, decay=22, amp=0.9)
    snare_noise = _bandpass(snare_noise, 1500, 8000)
    snare = _soft_clip(snare_tone * 0.35 + snare_noise, 1.2)

    hihat = _noise_burst(0.11, decay=55, amp=0.55)
    hihat = _highpass(hihat, 8500)

    clap = _noise_burst(0.16, decay=28, amp=0.7)
    clap = _highpass(clap, 1400)

    return {
        'kick': _to_int16(kick, 0.9),
        'snare': _to_int16(snare, 0.8),
        'hihat': _to_int16(hihat, 0.55),
        'clap': _to_int16(clap, 0.6),
    }


def _kit_ethnic():
    kick = _mix(
        _sine_sweep(0.5, 220, 70, decay=5, amp=1.0),
        _sine_sweep(0.28, 140, 90, decay=6, amp=0.5),
    )
    kick = _soft_clip(kick, 1.1)

    snare_t = np.linspace(0, 0.36, int(SAMPLE_RATE * 0.36), False)
    snare_tone = np.sin(2 * np.pi * 260 * snare_t) * np.exp(-snare_t * 8)
    snare_noise = _noise_burst(0.36, decay=10, amp=0.4)
    snare = _soft_clip(snare_tone + snare_noise, 1.1)

    hihat = _noise_burst(0.24, decay=20, amp=0.6)
    hihat = _bandpass(hihat, 3000, 7000)

    clap_t = np.linspace(0, 0.18, int(SAMPLE_RATE * 0.18), False)
    clap = np.sin(2 * np.pi * 520 * clap_t) * np.exp(-clap_t * 18)

    return {
        'kick': _to_int16(kick, 0.9),
        'snare': _to_int16(snare, 0.8),
        'hihat': _to_int16(hihat, 0.6),
        'clap': _to_int16(clap, 0.6),
    }


def _kit_8bit():
    kick = _square_sweep(0.28, 200, 60, decay=9, amp=1.0)
    kick = _bitcrush(kick, bits=5, downsample=6)

    snare = _noise_burst(0.22, decay=18, amp=0.9)
    snare = _bitcrush(snare, bits=4, downsample=5)

    hihat = _noise_burst(0.08, decay=35, amp=0.8)
    hihat = _highpass(hihat, 9000)
    hihat = _bitcrush(hihat, bits=3, downsample=6)

    clap = _mix(
        _square_sweep(0.09, 800, 300, decay=20, amp=0.6),
        _noise_burst(0.09, decay=30, amp=0.4),
    )
    clap = _bitcrush(clap, bits=4, downsample=5)

    return {
        'kick': _to_int16(kick, 0.8),
        'snare': _to_int16(snare, 0.7),
        'hihat': _to_int16(hihat, 0.5),
        'clap': _to_int16(clap, 0.6),
    }


def _kit_bizarre():
    kick = _mix(
        _sine_sweep(0.4, 90, 280, decay=6, amp=0.9),
        _square_sweep(0.22, 600, 80, decay=12, amp=0.4),
    )
    kick = _soft_clip(kick, 1.6)

    snare_len = int(SAMPLE_RATE * 0.32)
    snare_t = np.linspace(0, 0.32, snare_len, False)
    chirp = signal.chirp(snare_t, f0=600, f1=80, t1=0.32, method='linear')
    snare = _mix(chirp * np.exp(-snare_t * 7), _noise_burst(0.32, decay=12, amp=0.6))
    snare = _soft_clip(snare, 1.4)

    hihat = _mix(
        _noise_burst(0.16, decay=30, amp=0.6),
        np.sin(2 * np.pi * 4300 * snare_t[:int(SAMPLE_RATE * 0.16)]) * np.exp(-snare_t[:int(SAMPLE_RATE * 0.16)] * 25) * 0.3,
    )
    hihat = _bandpass(hihat, 2500, 9000)

    clap = _mix(
        _noise_burst(0.14, decay=40, amp=0.6),
        _sine_sweep(0.14, 900, 120, decay=15, amp=0.4),
    )
    clap = _soft_clip(clap, 1.3)

    return {
        'kick': _to_int16(kick, 0.85),
        'snare': _to_int16(snare, 0.8),
        'hihat': _to_int16(hihat, 0.55),
        'clap': _to_int16(clap, 0.65),
    }


KIT_LIBRARY = {
    'classic': _kit_classic,
    'real': _kit_real,
    'dnb': _kit_dnb,
    'ethnic': _kit_ethnic,
    '8bit': _kit_8bit,
    'bizarre': _kit_bizarre,
}

KIT_ORDER = ['classic', 'real', 'dnb', 'ethnic', '8bit', 'bizarre']


class AudioOutput:
    """Audio playback with classic synth filter."""
    
    def __init__(self, config=None):
        # Load filter config
        if config and 'filter' in config:
            fc = config['filter']
            self.filter_enabled = fc.get('enabled', True)
            min_freq = fc.get('min_freq', 80)
            max_freq = fc.get('max_freq', 12000)
            resonance = fc.get('resonance', 2.0)
            order = fc.get('order', 4)
        else:
            self.filter_enabled = True
            min_freq = 80
            max_freq = 12000
            resonance = 2.0
            order = 4
        
        # Audio config
        if config and 'audio' in config:
            sample_rate = config['audio'].get('sample_rate', SAMPLE_RATE)
            buffer_size = config['audio'].get('buffer_size', 512)
            kit_name = config['audio'].get('kit', 'classic')
            master_volume = config['audio'].get('volume', 1.0)
            mixer_channels = config['audio'].get('channels', 32)
            instrument_gain = config['audio'].get('instrument_gain', {}) or {}
        else:
            sample_rate = SAMPLE_RATE
            buffer_size = 512
            kit_name = 'classic'
            master_volume = 1.0
            mixer_channels = 32
            instrument_gain = {}

        try:
            master_volume = float(master_volume)
        except (TypeError, ValueError):
            master_volume = 1.0
        self.master_volume = max(0.0, min(1.0, master_volume))

        try:
            mixer_channels = int(mixer_channels)
        except (TypeError, ValueError):
            mixer_channels = 32
        mixer_channels = max(4, min(128, mixer_channels))
        
        # Create filter
        self.synth_filter = SynthFilter(
            min_freq=min_freq,
            max_freq=max_freq,
            resonance=resonance,
            order=order
        )

        self.filter_cutoff_min = float(min_freq)
        self.filter_cutoff_max = float(max_freq)
        self.filter_resonance_min = 0.5
        self.filter_resonance_max = 8.0
        self.filter_cutoff_default = self.synth_filter.get_default_cutoff()
        self.filter_resonance_default = float(resonance)
        self.filter_resonance_default = max(
            self.filter_resonance_min,
            min(self.filter_resonance_max, self.filter_resonance_default),
        )
        self.filter_cutoff = self.filter_cutoff_default
        self.filter_resonance = self.filter_resonance_default
        
        # Init pygame mixer
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
        pygame.mixer.set_num_channels(mixer_channels)
        
        # Load sound library metadata
        libraries_config = (config or {}).get('libraries', {}) if isinstance(config, dict) else {}
        sound_file = libraries_config.get('sound_file')
        self.sound_libraries = load_sound_libraries(sound_file)
        self.sound_library_by_id = {lib["id"]: lib for lib in self.sound_libraries}
        self.available_kits = [lib["id"] for lib in self.sound_libraries if lib.get("active", True)]
        if not self.available_kits:
            self.available_kits = KIT_ORDER[:]

        kit_name = kit_name.lower()
        if kit_name not in self.available_kits:
            kit_name = self.available_kits[0]
        self.kit_name = kit_name
        self.kit_display_name = self.sound_library_by_id.get(kit_name, {}).get("name", kit_name)
        self.kit_gain = self._get_kit_gain(kit_name)

        # Instrument gain
        self.instrument_gain: Dict[str, float] = {}
        for key in ["kick", "snare", "hihat", "clap"]:
            try:
                self.instrument_gain[key] = float(instrument_gain.get(key, 1.0))
            except (TypeError, ValueError):
                self.instrument_gain[key] = 1.0

        # Generate sounds
        self.raw_samples = {}
        self.sounds = {}
        self._generate_sounds()
        
        print("✓ Audio samples generated")
        print(f"✓ Kit: {self.kit_name}")
        if self.filter_enabled:
            print(f"✓ Synth Filter: {min_freq}Hz - {max_freq}Hz, Resonance: {resonance}")
    
    def _generate_sounds(self):
        self.raw_samples = self._build_kit_samples(self.kit_name)
        for name, data in self.raw_samples.items():
            stereo = np.column_stack([data, data]).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(stereo)
        if self.filter_enabled:
            self._apply_filter_to_sounds()

    def _build_kit_samples(self, kit_id: str) -> Dict[str, np.ndarray]:
        lib = self.sound_library_by_id.get(kit_id)
        if not lib:
            kit_fn = KIT_LIBRARY.get(kit_id, _kit_classic)
            return kit_fn()

        lib_type = lib.get("type", "builtin")
        if lib_type == "builtin":
            kit_fn = KIT_LIBRARY.get(kit_id, _kit_classic)
            return kit_fn()
        if lib_type == "samples":
            return self._build_sample_kit(lib)
        if lib_type == "synth":
            return self._build_synth_kit(lib)
        if lib_type == "hybrid":
            return self._build_hybrid_kit(lib)

        kit_fn = KIT_LIBRARY.get(kit_id, _kit_classic)
        return kit_fn()

    def _build_sample_kit(self, lib: Dict[str, Any]) -> Dict[str, np.ndarray]:
        samples = lib.get("samples", {}) or {}
        fallback = _kit_classic()
        kit = {}
        for instrument in ["kick", "snare", "hihat", "clap"]:
            path = samples.get(instrument)
            data = None
            if path:
                data = self._load_sample(path)
            if data is None or len(data) == 0:
                data = fallback.get(instrument, np.zeros(1, dtype=np.int16))
            kit[instrument] = data
        return kit

    def _build_synth_kit(self, lib: Dict[str, Any]) -> Dict[str, np.ndarray]:
        synth = lib.get("synth", {}) or {}
        waveform = synth.get("waveform", "sine")
        duration = float(synth.get("duration", 0.35))
        decay = float(synth.get("decay", 5.0))
        attack = float(synth.get("attack", 0.01))
        notes = lib.get("notes", {}) or {}
        kit = {}
        for instrument in ["kick", "snare", "hihat", "clap"]:
            note = notes.get(instrument, "C4")
            wave = _synth_note(note, duration=duration, decay=decay,
                               attack=attack, waveform=waveform, amp=0.9)
            kit[instrument] = _to_int16(wave, 0.9) if wave.size else np.zeros(1, dtype=np.int16)
        return kit

    def _build_hybrid_kit(self, lib: Dict[str, Any]) -> Dict[str, np.ndarray]:
        base_id = str(lib.get("base_kit", "classic")).lower()
        base_fn = KIT_LIBRARY.get(base_id, _kit_classic)
        kit = base_fn()
        synth = lib.get("synth", {}) or {}
        waveform = synth.get("waveform", "sine")
        duration = float(synth.get("duration", 0.35))
        decay = float(synth.get("decay", 5.0))
        attack = float(synth.get("attack", 0.01))
        notes = lib.get("notes", {}) or {}
        for instrument, note in notes.items():
            if instrument not in kit:
                continue
            wave = _synth_note(note, duration=duration, decay=decay,
                               attack=attack, waveform=waveform, amp=0.9)
            if wave.size:
                kit[instrument] = _to_int16(wave, 0.9)
        return kit

    def _load_sample(self, path: str) -> Optional[np.ndarray]:
        if not path:
            return None
        abs_path = path
        if not os.path.isabs(path):
            abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            return None
        try:
            sound = pygame.mixer.Sound(abs_path)
            arr = pygame.sndarray.array(sound)
            if arr.ndim == 2:
                mono = np.mean(arr, axis=1).astype(np.int16)
            else:
                mono = arr.astype(np.int16)
            return mono
        except Exception:
            return None

    def set_kit(self, name: str) -> bool:
        if not name:
            return False
        name = name.lower()
        if name not in self.sound_library_by_id and name not in KIT_LIBRARY:
            return False
        if name == self.kit_name:
            return True
        self.kit_name = name
        self.kit_display_name = self.sound_library_by_id.get(name, {}).get("name", name)
        self.kit_gain = self._get_kit_gain(name)
        self._generate_sounds()
        return True

    def cycle_kit(self, direction: int = 1) -> str:
        if not self.available_kits:
            return self.kit_name
        if self.kit_name not in self.available_kits:
            self.kit_name = self.available_kits[0]
        idx = self.available_kits.index(self.kit_name)
        idx = (idx + direction) % len(self.available_kits)
        self.set_kit(self.available_kits[idx])
        return self.kit_name
    
    def _apply_filter_to_sounds(self):
        if not self.filter_enabled:
            for name, raw_data in self.raw_samples.items():
                stereo = np.column_stack([raw_data, raw_data]).astype(np.int16)
                self.sounds[name] = pygame.sndarray.make_sound(stereo)
            return

        for name, raw_data in self.raw_samples.items():
            filtered = self.synth_filter.apply(
                raw_data,
                cutoff=self.filter_cutoff,
                resonance=self.filter_resonance,
            )
            stereo = np.column_stack([filtered, filtered]).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(stereo)
    
    def play_sound(self, instrument: str, velocity: int = 127):
        sound = self.sounds.get(instrument)
        if sound:
            kit_gain = self.kit_gain.get(instrument, 1.0)
            inst_gain = self.instrument_gain.get(instrument, 1.0)
            volume = (velocity / 127.0) * self.master_volume * kit_gain * inst_gain
            channel = sound.play()
            if channel is not None:
                channel.set_volume(volume)

    def trigger(self, instrument: str, cell_state: int):
        if cell_state == EMPTY:
            return
        if cell_state == BLACK:
            velocity = 80
        elif cell_state == WHITE:
            velocity = 127
        else:
            return
        self.play_sound(instrument, velocity)

    def set_filter_cutoff(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        value = max(self.filter_cutoff_min, min(self.filter_cutoff_max, value))
        if abs(value - self.filter_cutoff) < 5.0:
            return
        self.filter_cutoff = value
        if self.filter_enabled:
            self._apply_filter_to_sounds()

    def set_filter_resonance(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        value = max(self.filter_resonance_min, min(self.filter_resonance_max, value))
        if abs(value - self.filter_resonance) < 0.05:
            return
        self.filter_resonance = value
        if self.filter_enabled:
            self._apply_filter_to_sounds()

    def reset_filter(self):
        self.filter_cutoff = self.filter_cutoff_default
        self.filter_resonance = self.filter_resonance_default
        if self.filter_enabled:
            self._apply_filter_to_sounds()

    def _get_kit_gain(self, kit_id: str) -> Dict[str, float]:
        gain = {}
        lib = self.sound_library_by_id.get(kit_id, {})
        lib_gain = lib.get("gain", {}) or {}
        for key in ["kick", "snare", "hihat", "clap"]:
            try:
                gain[key] = float(lib_gain.get(key, 1.0))
            except (TypeError, ValueError):
                gain[key] = 1.0
        return gain

    def set_master_volume(self, value: float):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return
        self.master_volume = max(0.0, min(1.0, value))

    def adjust_volume(self, delta: float):
        self.set_master_volume(self.master_volume + delta)
    
    def close(self):
        pygame.mixer.quit()
