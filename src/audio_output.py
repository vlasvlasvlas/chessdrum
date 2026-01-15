"""
Audio output module for ChessDrum.
Classic synth-style filter with configurable cutoff and resonance.
"""
import numpy as np
import pygame
from scipy import signal

from grid import EMPTY, BLACK, WHITE

SAMPLE_RATE = 44100


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


class SynthFilter:
    """
    Classic synth-style resonant lowpass filter.
    
    - Rotation left (-1) = LOW cutoff frequency (dark, muffled)
    - Center (0) = MID cutoff (normal)  
    - Rotation right (+1) = HIGH cutoff frequency (bright, open)
    
    Resonance adds a peak at the cutoff frequency (the classic "squelchy" sound).
    """
    
    def __init__(self, min_freq=80, max_freq=12000, resonance=2.0, order=4):
        """
        Args:
            min_freq: Minimum cutoff frequency in Hz (at rotation = -1)
            max_freq: Maximum cutoff frequency in Hz (at rotation = +1) 
            resonance: Q factor (1.0 = flat, 2.0+ = resonant peak)
            order: Filter order (higher = steeper rolloff)
        """
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.resonance = resonance
        self.order = order
    
    def get_cutoff(self, rotation: float) -> float:
        """
        Calculate cutoff frequency from rotation value.
        Uses exponential curve for more natural feel.
        
        rotation: -1 (low) to +1 (high)
        """
        # Normalize to 0-1 range
        t = (rotation + 1) / 2  # -1 -> 0, 0 -> 0.5, +1 -> 1
        
        # Exponential interpolation for more musical response
        # Low values change slowly, high values change faster
        freq_ratio = self.max_freq / self.min_freq
        cutoff = self.min_freq * (freq_ratio ** t)
        
        return cutoff
    
    def apply(self, audio_data: np.ndarray, rotation: float) -> np.ndarray:
        """
        Apply lowpass filter with current rotation.
        
        Args:
            audio_data: Audio samples (int16)
            rotation: -1.0 to +1.0 
                     -1 = low cutoff (dark/muffled)
                      0 = center (no filter, natural sound)
                     +1 = high cutoff (bright, more highs)
        
        Returns:
            Filtered audio samples (int16)
        """
        # At center, return unfiltered
        if abs(rotation) < 0.05:
            return audio_data
        
        # Convert to float
        audio_float = audio_data.astype(np.float32) / 32767.0
        
        # Calculate cutoff frequency
        # Center (0) = center_freq, Left (-1) = min_freq, Right (+1) = max_freq
        center_freq = np.sqrt(self.min_freq * self.max_freq)  # Geometric mean
        
        if rotation < 0:
            # Going left: center_freq down to min_freq
            t = -rotation  # 0 to 1
            cutoff = center_freq * ((self.min_freq / center_freq) ** t)
        else:
            # Going right: center_freq up to max_freq
            t = rotation  # 0 to 1
            cutoff = center_freq * ((self.max_freq / center_freq) ** t)
        
        # Normalize for scipy
        nyquist = SAMPLE_RATE / 2
        normalized = min(max(cutoff / nyquist, 0.01), 0.99)
        
        try:
            # For left rotation (dark): use lowpass filter
            # For right rotation (bright): boost highs slightly
            if rotation < 0:
                # Lowpass filter
                b, a = signal.butter(self.order, normalized, btype='low')
                filtered = signal.filtfilt(b, a, audio_float)
                
                # Add resonance (boost at cutoff frequency)
                if self.resonance > 1.0 and len(audio_float) > 50:
                    bw = 0.15
                    bp_low = max(0.01, normalized * (1 - bw))
                    bp_high = min(0.99, normalized * (1 + bw))
                    
                    if bp_low < bp_high:
                        b_bp, a_bp = signal.butter(2, [bp_low, bp_high], btype='band')
                        resonant = signal.filtfilt(b_bp, a_bp, audio_float)
                        amount = (self.resonance - 1) * abs(rotation) * 0.5
                        filtered = filtered + resonant * amount
            else:
                # Highpass filter to remove lows and make it brighter
                b, a = signal.butter(self.order, normalized, btype='high')
                filtered = signal.filtfilt(b, a, audio_float)
                
                # Add some of the original back for body
                filtered = filtered * 0.7 + audio_float * 0.3
                
                # Add resonance
                if self.resonance > 1.0 and len(audio_float) > 50:
                    bw = 0.15
                    bp_low = max(0.01, normalized * (1 - bw))
                    bp_high = min(0.99, normalized * (1 + bw))
                    
                    if bp_low < bp_high:
                        b_bp, a_bp = signal.butter(2, [bp_low, bp_high], btype='band')
                        resonant = signal.filtfilt(b_bp, a_bp, audio_float)
                        amount = (self.resonance - 1) * rotation * 0.4
                        filtered = filtered + resonant * amount
            
            # Soft clip to prevent distortion
            filtered = np.tanh(filtered * 1.2) / 1.2
            
            return (np.clip(filtered, -1, 1) * 32767).astype(np.int16)
        
        except Exception as e:
            return audio_data


def generate_kick(duration=0.3):
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


def generate_snare(duration=0.2):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    noise = np.random.uniform(-1, 1, samples)
    noise_env = np.exp(-t * 15)
    tone = np.sin(2 * np.pi * 200 * t)
    tone_env = np.exp(-t * 20)
    wave = noise * noise_env * 0.5 + tone * tone_env * 0.5
    return (wave * 32767 * 0.7).astype(np.int16)


def generate_hihat(duration=0.1):
    samples = int(SAMPLE_RATE * duration)
    t = np.linspace(0, duration, samples, False)
    noise = np.random.uniform(-1, 1, samples)
    envelope = np.exp(-t * 30)
    wave = noise * envelope
    return (wave * 32767 * 0.5).astype(np.int16)


def generate_clap(duration=0.15):
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
        _sine_sweep(0.35, 140, 50, decay=7, amp=1.0),
        _noise_burst(0.04, decay=50, amp=0.18),
        _sine_sweep(0.02, 900, 200, decay=50, amp=0.35),
    )
    kick = _soft_clip(kick, 1.2)

    snare_t = np.linspace(0, 0.25, int(SAMPLE_RATE * 0.25), False)
    snare_tone = np.sin(2 * np.pi * 180 * snare_t) * np.exp(-snare_t * 12)
    snare_noise = _noise_burst(0.25, decay=16, amp=0.8)
    snare_noise = _highpass(snare_noise, 900)
    snare = _soft_clip(snare_tone * 0.4 + snare_noise, 1.1)

    hihat = _noise_burst(0.12, decay=45, amp=0.6)
    hihat = _highpass(hihat, 7000)

    clap = np.zeros(int(SAMPLE_RATE * 0.2), dtype=np.float32)
    for i, amp in enumerate([0.7, 0.5, 0.35, 0.25]):
        offset = int(i * 0.012 * SAMPLE_RATE)
        burst = _noise_burst(0.05, decay=35, amp=amp)
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
        _sine_sweep(0.25, 180, 45, decay=10, amp=1.1),
        _sine_sweep(0.018, 1200, 250, decay=55, amp=0.5),
        _noise_burst(0.03, decay=60, amp=0.15),
    )
    kick = _soft_clip(kick, 1.4)

    snare_t = np.linspace(0, 0.2, int(SAMPLE_RATE * 0.2), False)
    snare_tone = np.sin(2 * np.pi * 220 * snare_t) * np.exp(-snare_t * 18)
    snare_noise = _noise_burst(0.2, decay=22, amp=0.9)
    snare_noise = _bandpass(snare_noise, 1500, 8000)
    snare = _soft_clip(snare_tone * 0.35 + snare_noise, 1.2)

    hihat = _noise_burst(0.08, decay=55, amp=0.55)
    hihat = _highpass(hihat, 8500)

    clap = _noise_burst(0.12, decay=28, amp=0.7)
    clap = _highpass(clap, 1400)

    return {
        'kick': _to_int16(kick, 0.9),
        'snare': _to_int16(snare, 0.8),
        'hihat': _to_int16(hihat, 0.55),
        'clap': _to_int16(clap, 0.6),
    }


def _kit_ethnic():
    kick = _mix(
        _sine_sweep(0.4, 220, 70, decay=5, amp=1.0),
        _sine_sweep(0.2, 140, 90, decay=6, amp=0.5),
    )
    kick = _soft_clip(kick, 1.1)

    snare_t = np.linspace(0, 0.28, int(SAMPLE_RATE * 0.28), False)
    snare_tone = np.sin(2 * np.pi * 260 * snare_t) * np.exp(-snare_t * 8)
    snare_noise = _noise_burst(0.28, decay=10, amp=0.4)
    snare = _soft_clip(snare_tone + snare_noise, 1.1)

    hihat = _noise_burst(0.18, decay=20, amp=0.6)
    hihat = _bandpass(hihat, 3000, 7000)

    clap_t = np.linspace(0, 0.12, int(SAMPLE_RATE * 0.12), False)
    clap = np.sin(2 * np.pi * 520 * clap_t) * np.exp(-clap_t * 18)

    return {
        'kick': _to_int16(kick, 0.9),
        'snare': _to_int16(snare, 0.8),
        'hihat': _to_int16(hihat, 0.6),
        'clap': _to_int16(clap, 0.6),
    }


def _kit_8bit():
    kick = _square_sweep(0.2, 200, 60, decay=9, amp=1.0)
    kick = _bitcrush(kick, bits=5, downsample=6)

    snare = _noise_burst(0.16, decay=18, amp=0.9)
    snare = _bitcrush(snare, bits=4, downsample=5)

    hihat = _noise_burst(0.06, decay=35, amp=0.8)
    hihat = _highpass(hihat, 9000)
    hihat = _bitcrush(hihat, bits=3, downsample=6)

    clap = _mix(
        _square_sweep(0.06, 800, 300, decay=20, amp=0.6),
        _noise_burst(0.06, decay=30, amp=0.4),
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
        _sine_sweep(0.3, 90, 280, decay=6, amp=0.9),
        _square_sweep(0.15, 600, 80, decay=12, amp=0.4),
    )
    kick = _soft_clip(kick, 1.6)

    snare_len = int(SAMPLE_RATE * 0.24)
    snare_t = np.linspace(0, 0.24, snare_len, False)
    chirp = signal.chirp(snare_t, f0=600, f1=80, t1=0.24, method='linear')
    snare = _mix(chirp * np.exp(-snare_t * 7), _noise_burst(0.24, decay=12, amp=0.6))
    snare = _soft_clip(snare, 1.4)

    hihat = _mix(
        _noise_burst(0.12, decay=30, amp=0.6),
        np.sin(2 * np.pi * 4300 * snare_t[:int(SAMPLE_RATE * 0.12)]) * np.exp(-snare_t[:int(SAMPLE_RATE * 0.12)] * 25) * 0.3,
    )
    hihat = _bandpass(hihat, 2500, 9000)

    clap = _mix(
        _noise_burst(0.1, decay=40, amp=0.6),
        _sine_sweep(0.1, 900, 120, decay=15, amp=0.4),
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
        else:
            sample_rate = SAMPLE_RATE
            buffer_size = 512
            kit_name = 'classic'
        
        # Create filter
        self.synth_filter = SynthFilter(
            min_freq=min_freq,
            max_freq=max_freq,
            resonance=resonance,
            order=order
        )
        
        # Init pygame mixer
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
        
        # Sound kits
        kit_name = kit_name.lower()
        if kit_name not in KIT_LIBRARY:
            kit_name = 'classic'
        self.kit_name = kit_name
        self.available_kits = KIT_ORDER[:]

        # Generate sounds
        self.raw_samples = {}
        self._filter_value = 0.0  # Start at center (neutral/no filter)
        self.sounds = {}
        self._generate_sounds()
        
        print("✓ Audio samples generated")
        print(f"✓ Kit: {self.kit_name}")
        if self.filter_enabled:
            print(f"✓ Synth Filter: {min_freq}Hz - {max_freq}Hz, Resonance: {resonance}")
    
    @property
    def filter_value(self) -> float:
        return self._filter_value
    
    @filter_value.setter
    def filter_value(self, value: float):
        if not self.filter_enabled:
            return
        new_value = max(-1.0, min(1.0, value))
        if abs(new_value - self._filter_value) > 0.02:
            self._filter_value = new_value
            self._apply_filter_to_sounds()
            # Debug: show current cutoff
            cutoff = self.synth_filter.get_cutoff(new_value)
            # print(f"Cutoff: {cutoff:.0f} Hz")
    
    def _generate_sounds(self):
        kit_fn = KIT_LIBRARY.get(self.kit_name, _kit_classic)
        self.raw_samples = kit_fn()
        for name, data in self.raw_samples.items():
            stereo = np.column_stack([data, data]).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(stereo)
        if self.filter_enabled and abs(self._filter_value) >= 0.05:
            self._apply_filter_to_sounds()

    def set_kit(self, name: str) -> bool:
        if not name:
            return False
        name = name.lower()
        if name not in KIT_LIBRARY:
            return False
        if name == self.kit_name:
            return True
        self.kit_name = name
        self._generate_sounds()
        return True

    def cycle_kit(self, direction: int = 1) -> str:
        if not self.available_kits:
            return self.kit_name
        idx = self.available_kits.index(self.kit_name)
        idx = (idx + direction) % len(self.available_kits)
        self.set_kit(self.available_kits[idx])
        return self.kit_name
    
    def _apply_filter_to_sounds(self):
        if not self.filter_enabled:
            return
        
        for name, raw_data in self.raw_samples.items():
            filtered = self.synth_filter.apply(raw_data, self._filter_value)
            stereo = np.column_stack([filtered, filtered]).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(stereo)
    
    def play_sound(self, instrument: str, velocity: int = 127):
        sound = self.sounds.get(instrument)
        if sound:
            volume = velocity / 127.0
            sound.set_volume(volume)
            sound.play()
    
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
    
    def close(self):
        pygame.mixer.quit()
