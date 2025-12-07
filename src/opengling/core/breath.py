"""Breath and lip smack detection for cleaner audio editing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from opengling.core.models import EditDecision, EditType, ProcessingConfig

logger = logging.getLogger(__name__)


class BreathDetector:
    """Detects breath sounds and lip smacks in audio for removal."""
    
    # Typical breath characteristics
    MIN_BREATH_DURATION = 0.05  # 50ms minimum
    MAX_BREATH_DURATION = 0.4   # 400ms maximum
    BREATH_FREQ_LOW = 100       # Hz - breaths are low frequency
    BREATH_FREQ_HIGH = 1000     # Hz
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def detect_breaths(
        self,
        audio_path: Path | str,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> list[EditDecision]:
        """
        Detect breath sounds and lip smacks in audio.
        
        Breaths typically have these characteristics:
        - Short duration (50-400ms)
        - Low frequency content (100-1000 Hz)
        - Occur between speech segments
        - Lower amplitude than speech
        
        Args:
            audio_path: Path to audio file
            min_duration: Minimum breath duration to detect
            max_duration: Maximum breath duration to detect
            
        Returns:
            List of EditDecisions marking breath regions for removal
        """
        if not getattr(self.config, 'remove_breaths', False):
            return []
            
        audio_path = Path(audio_path)
        min_dur = min_duration or self.MIN_BREATH_DURATION
        max_dur = max_duration or self.MAX_BREATH_DURATION
        
        logger.info(f"Detecting breaths in {audio_path.name}")
        
        try:
            return self._detect_with_spectral_analysis(audio_path, min_dur, max_dur)
        except Exception as e:
            logger.warning(f"Breath detection failed: {e}")
            return []
    
    def _detect_with_spectral_analysis(
        self,
        audio_path: Path,
        min_duration: float,
        max_duration: float,
    ) -> list[EditDecision]:
        """Detect breaths using spectral analysis."""
        try:
            from pydub import AudioSegment
            from scipy import signal
            from scipy.fft import rfft, rfftfreq
        except ImportError:
            raise ImportError(
                "pydub and scipy are required. "
                "Install with: pip install pydub scipy"
            )
        
        # Load audio
        audio = AudioSegment.from_file(str(audio_path))
        sample_rate = audio.frame_rate
        audio = audio.set_channels(1)  # Mono
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        
        # Calculate speech threshold (breaths are quieter)
        speech_threshold = np.percentile(np.abs(samples), 90) * 0.3
        
        # Window size for analysis (50ms windows)
        window_ms = 50
        window_samples = int(sample_rate * window_ms / 1000)
        hop_samples = window_samples // 2
        
        breaths = []
        breath_start = None
        breath_samples = []
        
        for i in range(0, len(samples) - window_samples, hop_samples):
            window = samples[i:i + window_samples]
            window_time = i / sample_rate
            
            # Calculate window properties
            window_rms = np.sqrt(np.mean(window ** 2))
            
            # Check if this could be a breath
            is_quiet = window_rms < speech_threshold
            is_breath_like = False
            
            if is_quiet and window_rms > 0.01:  # Not silent, but quiet
                # Spectral analysis
                spectrum = np.abs(rfft(window))
                freqs = rfftfreq(len(window), 1/sample_rate)
                
                # Find energy in breath frequency band
                breath_mask = (freqs >= self.BREATH_FREQ_LOW) & (freqs <= self.BREATH_FREQ_HIGH)
                speech_mask = freqs > self.BREATH_FREQ_HIGH
                
                breath_energy = np.sum(spectrum[breath_mask])
                speech_energy = np.sum(spectrum[speech_mask])
                
                # Breaths have more low-frequency content
                if breath_energy > 0:
                    ratio = speech_energy / (breath_energy + 1e-10)
                    is_breath_like = ratio < 0.5  # More low freq than high
            
            # Track breath regions
            if is_breath_like:
                if breath_start is None:
                    breath_start = window_time
                breath_samples.append(window_rms)
            else:
                if breath_start is not None:
                    breath_end = window_time
                    breath_duration = breath_end - breath_start
                    
                    # Check duration constraints
                    if min_duration <= breath_duration <= max_duration:
                        breaths.append(EditDecision(
                            start=breath_start,
                            end=breath_end,
                            edit_type=EditType.SILENCE,  # Treat as silence type
                            keep=False,
                            reason=f"Breath detected ({breath_duration*1000:.0f}ms)",
                            confidence=0.7,
                        ))
                    
                    breath_start = None
                    breath_samples = []
        
        logger.info(f"Found {len(breaths)} breath sounds")
        return breaths


class LipSmackDetector:
    """Detects lip smacks and mouth sounds."""
    
    # Lip smack characteristics
    MIN_DURATION = 0.02   # 20ms
    MAX_DURATION = 0.15   # 150ms
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def detect_lip_smacks(
        self,
        audio_path: Path | str,
    ) -> list[EditDecision]:
        """
        Detect lip smacks in audio.
        
        Lip smacks are characterized by:
        - Very short duration (20-150ms)
        - Sharp transient at start
        - Higher frequency content than breaths
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of EditDecisions marking lip smacks for removal
        """
        if not getattr(self.config, 'remove_lip_smacks', False):
            return []
            
        audio_path = Path(audio_path)
        
        logger.info(f"Detecting lip smacks in {audio_path.name}")
        
        try:
            return self._detect_transients(audio_path)
        except Exception as e:
            logger.warning(f"Lip smack detection failed: {e}")
            return []
    
    def _detect_transients(self, audio_path: Path) -> list[EditDecision]:
        """Detect sharp transients that could be lip smacks."""
        try:
            from pydub import AudioSegment
            from scipy import signal
        except ImportError:
            return []
        
        audio = AudioSegment.from_file(str(audio_path))
        sample_rate = audio.frame_rate
        audio = audio.set_channels(1)
        
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        
        # Detect transients using onset detection
        # Simple energy-based onset detection
        window_ms = 10
        window_samples = int(sample_rate * window_ms / 1000)
        
        # Calculate local energy
        energy = np.array([
            np.sum(samples[i:i+window_samples]**2)
            for i in range(0, len(samples) - window_samples, window_samples)
        ])
        
        # Find sharp increases (transients)
        energy_diff = np.diff(energy)
        threshold = np.percentile(np.abs(energy_diff), 95)
        
        lip_smacks = []
        min_samples = int(self.MIN_DURATION * sample_rate)
        max_samples = int(self.MAX_DURATION * sample_rate)
        
        i = 0
        while i < len(energy_diff):
            if energy_diff[i] > threshold:
                # Found potential onset
                start_time = i * window_samples / sample_rate
                
                # Look for offset
                j = i + 1
                while j < len(energy_diff) and j < i + max_samples // window_samples:
                    if energy_diff[j] < -threshold * 0.5:
                        # Found offset
                        end_time = j * window_samples / sample_rate
                        duration = end_time - start_time
                        
                        if self.MIN_DURATION <= duration <= self.MAX_DURATION:
                            lip_smacks.append(EditDecision(
                                start=start_time,
                                end=end_time,
                                edit_type=EditType.SILENCE,
                                keep=False,
                                reason=f"Lip smack detected ({duration*1000:.0f}ms)",
                                confidence=0.6,
                            ))
                        break
                    j += 1
                i = j
            else:
                i += 1
        
        logger.info(f"Found {len(lip_smacks)} lip smacks")
        return lip_smacks


def detect_mouth_sounds(
    audio_path: Path | str,
    config: ProcessingConfig,
) -> list[EditDecision]:
    """
    Convenience function to detect all mouth sounds (breaths + lip smacks).
    
    Args:
        audio_path: Path to audio file
        config: Processing configuration
        
    Returns:
        Combined list of detected mouth sounds
    """
    breath_detector = BreathDetector(config)
    lip_smack_detector = LipSmackDetector(config)
    
    breaths = breath_detector.detect_breaths(audio_path)
    lip_smacks = lip_smack_detector.detect_lip_smacks(audio_path)
    
    # Combine and sort by time
    all_sounds = breaths + lip_smacks
    all_sounds.sort(key=lambda x: x.start)
    
    return all_sounds

