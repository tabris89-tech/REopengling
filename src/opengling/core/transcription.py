"""Transcription engine using faster-whisper for word-level timestamps."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Iterator

from opengling.core.models import TranscriptSegment, TranscriptWord, ProcessingConfig

logger = logging.getLogger(__name__)


class TranscriptionEngine:
    """Handles audio transcription using faster-whisper."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._model = None
        
    def _get_device_and_compute(self) -> tuple[str, str]:
        """Determine device and compute type."""
        device = self.config.device
        compute_type = self.config.compute_type
        
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
            
        return device, compute_type
    
    def _load_model(self):
        """Lazy load the Whisper model."""
        if self._model is not None:
            return
            
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper is required for transcription. "
                "Install with: pip install faster-whisper"
            )
        
        device, compute_type = self._get_device_and_compute()
        
        logger.info(
            f"Loading Whisper model '{self.config.whisper_model}' "
            f"on {device} with {compute_type}"
        )
        
        self._model = WhisperModel(
            self.config.whisper_model,
            device=device,
            compute_type=compute_type,
        )
        
    def transcribe(
        self,
        audio_path: Path | str,
        language: Optional[str] = None,
    ) -> list[TranscriptSegment]:
        """
        Transcribe audio file with word-level timestamps.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            language: Language code or None for auto-detection
            
        Returns:
            List of transcript segments with word-level timing
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        language = language or self.config.language
        
        logger.info(f"Transcribing {audio_path.name}...")
        
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,  # Use VAD to filter out non-speech
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )
        
        detected_language = info.language
        logger.info(f"Detected language: {detected_language} (probability: {info.language_probability:.2f})")
        
        result = []
        for segment in segments:
            words = []
            if segment.words:
                for word in segment.words:
                    words.append(TranscriptWord(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        confidence=word.probability,
                    ))
            
            # Calculate segment confidence as average word confidence
            avg_confidence = (
                sum(w.confidence for w in words) / len(words)
                if words else 0.5
            )
            
            result.append(TranscriptSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words,
                confidence=avg_confidence,
                language=detected_language,
            ))
        
        logger.info(f"Transcription complete: {len(result)} segments")
        return result
    
    def transcribe_streaming(
        self,
        audio_path: Path | str,
        language: Optional[str] = None,
    ) -> Iterator[TranscriptSegment]:
        """
        Transcribe audio file with streaming results.
        
        Yields segments as they are transcribed.
        """
        self._load_model()
        
        audio_path = Path(audio_path)
        language = language or self.config.language
        
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
        
        detected_language = info.language
        
        for segment in segments:
            words = []
            if segment.words:
                for word in segment.words:
                    words.append(TranscriptWord(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        confidence=word.probability,
                    ))
            
            avg_confidence = (
                sum(w.confidence for w in words) / len(words)
                if words else 0.5
            )
            
            yield TranscriptSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=words,
                confidence=avg_confidence,
                language=detected_language,
            )


def get_full_transcript(segments: list[TranscriptSegment]) -> str:
    """Combine all segments into a full transcript."""
    return " ".join(seg.text for seg in segments)


def get_word_list(segments: list[TranscriptSegment]) -> list[TranscriptWord]:
    """Extract all words from segments into a flat list."""
    words = []
    for segment in segments:
        words.extend(segment.words)
    return words

