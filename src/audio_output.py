"""
Audio output module for ChessDrum.
Classic synth-style filter with configurable cutoff and resonance.
"""
import numpy as np
import pygame
from scipy import signal

SAMPLE_RATE = 44100


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
        else:
            sample_rate = SAMPLE_RATE
            buffer_size = 512
        
        # Create filter
        self.synth_filter = SynthFilter(
            min_freq=min_freq,
            max_freq=max_freq,
            resonance=resonance,
            order=order
        )
        
        # Init pygame mixer
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=buffer_size)
        
        # Generate sounds
        self.raw_samples = {}
        self._filter_value = 0.0  # Start at center (neutral/no filter)
        self.sounds = {}
        self._generate_sounds()
        
        print("✓ Audio samples generated")
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
        self.raw_samples = {
            'kick': generate_kick(),
            'snare': generate_snare(),
            'hihat': generate_hihat(),
            'clap': generate_clap(),
        }
        for name, data in self.raw_samples.items():
            stereo = np.column_stack([data, data]).astype(np.int16)
            self.sounds[name] = pygame.sndarray.make_sound(stereo)
    
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
        if cell_state == 0:
            return
        velocity = 127 if cell_state == 1 else 80
        self.play_sound(instrument, velocity)
    
    def close(self):
        pygame.mixer.quit()
