"""Silence detection using audio analysis and VAD."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from opengling.core.models import EditDecision, EditType, ProcessingConfig

logger = logging.getLogger(__name__)


class SilenceDetector:
    """Detects silence regions in audio using WebRTC VAD and amplitude analysis."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def detect_silences(
        self,
        audio_path: Path | str,
        min_silence_duration: Optional[float] = None,
        padding: Optional[float] = None,
    ) -> list[EditDecision]:
        """
        Detect silence regions in audio file.
        
        Args:
            audio_path: Path to audio file
            min_silence_duration: Minimum silence duration to detect (seconds)
            padding: Padding to add around speech regions (seconds)
            
        Returns:
            List of EditDecisions marking silence regions for removal
        """
        # Check if silence removal is enabled
        if not self.config.remove_silences:
            logger.info("Silence removal disabled, skipping detection")
            return []
        
        audio_path = Path(audio_path)
        min_silence = min_silence_duration or self.config.silence_threshold
        padding = padding or self.config.silence_padding
        
        # Try WebRTC VAD first, fall back to amplitude-based detection
        try:
            return self._detect_with_webrtc_vad(audio_path, min_silence, padding)
        except Exception as e:
            logger.warning(f"WebRTC VAD failed: {e}, falling back to amplitude detection")
            return self._detect_with_amplitude(audio_path, min_silence, padding)
    
    def _detect_with_webrtc_vad(
        self,
        audio_path: Path,
        min_silence: float,
        padding: float,
    ) -> list[EditDecision]:
        """Detect silence using WebRTC Voice Activity Detection."""
        try:
            import webrtcvad
            from pydub import AudioSegment
        except ImportError:
            raise ImportError(
                "webrtcvad and pydub are required. "
                "Install with: pip install webrtcvad pydub"
            )
        
        logger.info("Detecting silences with WebRTC VAD...")
        
        # Load and prepare audio
        audio = AudioSegment.from_file(str(audio_path))
        
        # WebRTC VAD requires specific format: 16-bit mono, 8/16/32/48 kHz
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit
        
        # Use 16kHz for best VAD performance
        target_sample_rate = 16000
        audio = audio.set_frame_rate(target_sample_rate)
        
        # Create VAD with aggressiveness 2 (0-3, higher = more aggressive filtering)
        vad = webrtcvad.Vad(2)
        
        # Process in 30ms frames (WebRTC VAD requirement)
        frame_duration_ms = 30
        frame_size = int(target_sample_rate * frame_duration_ms / 1000)
        
        raw_audio = audio.raw_data
        num_frames = len(raw_audio) // (frame_size * 2)  # 2 bytes per sample
        
        # Detect speech frames
        speech_frames = []
        for i in range(num_frames):
            start_byte = i * frame_size * 2
            end_byte = start_byte + frame_size * 2
            frame = raw_audio[start_byte:end_byte]
            
            if len(frame) == frame_size * 2:
                is_speech = vad.is_speech(frame, target_sample_rate)
                frame_time = i * frame_duration_ms / 1000.0
                speech_frames.append((frame_time, is_speech))
        
        # Find silence regions (consecutive non-speech frames)
        silences = []
        silence_start = None
        
        for frame_time, is_speech in speech_frames:
            if not is_speech:
                if silence_start is None:
                    silence_start = frame_time
            else:
                if silence_start is not None:
                    silence_end = frame_time
                    silence_duration = silence_end - silence_start
                    
                    if silence_duration >= min_silence:
                        # Apply padding
                        padded_start = silence_start + padding
                        padded_end = silence_end - padding
                        
                        if padded_end > padded_start:
                            silences.append(EditDecision(
                                start=padded_start,
                                end=padded_end,
                                edit_type=EditType.SILENCE,
                                keep=False,
                                reason=f"Silence detected ({silence_duration:.2f}s)",
                                confidence=0.9,
                            ))
                    
                    silence_start = None
        
        # Handle silence at end of audio
        if silence_start is not None:
            silence_end = len(audio) / 1000.0
            silence_duration = silence_end - silence_start
            
            if silence_duration >= min_silence:
                padded_start = silence_start + padding
                padded_end = silence_end - padding
                
                if padded_end > padded_start:
                    silences.append(EditDecision(
                        start=padded_start,
                        end=padded_end,
                        edit_type=EditType.SILENCE,
                        keep=False,
                        reason=f"Silence detected ({silence_duration:.2f}s)",
                        confidence=0.9,
                    ))
        
        logger.info(f"Found {len(silences)} silence regions")
        return silences
    
    def _detect_with_amplitude(
        self,
        audio_path: Path,
        min_silence: float,
        padding: float,
    ) -> list[EditDecision]:
        """Fallback: Detect silence using amplitude threshold."""
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence
        except ImportError:
            raise ImportError("pydub is required. Install with: pip install pydub")
        
        logger.info("Detecting silences with amplitude analysis...")
        
        audio = AudioSegment.from_file(str(audio_path))
        
        # Detect silence regions (returns list of [start_ms, end_ms])
        # silence_thresh in dBFS (typically -40 to -50 for speech)
        silence_ranges = detect_silence(
            audio,
            min_silence_len=int(min_silence * 1000),  # Convert to ms
            silence_thresh=-40,  # dBFS
            seek_step=10,  # ms
        )
        
        silences = []
        for start_ms, end_ms in silence_ranges:
            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            duration = end_sec - start_sec
            
            # Apply padding
            padded_start = start_sec + padding
            padded_end = end_sec - padding
            
            if padded_end > padded_start:
                silences.append(EditDecision(
                    start=padded_start,
                    end=padded_end,
                    edit_type=EditType.SILENCE,
                    keep=False,
                    reason=f"Silence detected ({duration:.2f}s)",
                    confidence=0.8,
                ))
        
        logger.info(f"Found {len(silences)} silence regions")
        return silences


def merge_overlapping_regions(regions: list[EditDecision]) -> list[EditDecision]:
    """Merge overlapping edit regions."""
    if not regions:
        return []
    
    # Sort by start time
    sorted_regions = sorted(regions, key=lambda r: r.start)
    
    merged = [sorted_regions[0]]
    
    for current in sorted_regions[1:]:
        previous = merged[-1]
        
        # Check if overlapping or adjacent
        if current.start <= previous.end + 0.05:  # 50ms tolerance
            # Merge by extending the end
            merged[-1] = EditDecision(
                start=previous.start,
                end=max(previous.end, current.end),
                edit_type=previous.edit_type,
                keep=previous.keep,
                reason=f"{previous.reason}; {current.reason}",
                confidence=min(previous.confidence, current.confidence),
            )
        else:
            merged.append(current)
    
    return merged

