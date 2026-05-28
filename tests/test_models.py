"""Tests for OpenGling data models."""

import pytest
from pathlib import Path


class TestTranscriptWord:
    """Tests for TranscriptWord dataclass."""
    
    def test_creation(self):
        from opengling.core.models import TranscriptWord
        
        word = TranscriptWord(
            word="hello",
            start=0.0,
            end=0.5,
            confidence=0.95,
        )
        
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.95
    
    def test_duration(self):
        from opengling.core.models import TranscriptWord
        
        word = TranscriptWord(word="test", start=1.0, end=2.5, confidence=0.9)
        assert word.duration == 1.5


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""
    
    def test_creation(self):
        from opengling.core.models import TranscriptSegment
        
        segment = TranscriptSegment(
            text="Hello world",
            start=0.0,
            end=1.5,
        )
        
        assert segment.text == "Hello world"
        assert segment.duration == 1.5
        assert segment.confidence == 1.0
        assert segment.language == "en"
    
    def test_with_words(self, sample_transcript):
        segment = sample_transcript[0]
        
        assert len(segment.words) == 6
        assert segment.words[0].word == "Hello"
        assert segment.words[1].word == "um"


class TestEditDecision:
    """Tests for EditDecision dataclass."""
    
    def test_creation(self):
        from opengling.core.models import EditDecision, EditType
        
        edit = EditDecision(
            start=1.0,
            end=2.0,
            edit_type=EditType.SILENCE,
            keep=False,
            reason="Silence detected",
        )
        
        assert edit.duration == 1.0
        assert edit.edit_type == EditType.SILENCE
        assert not edit.keep
    
    def test_edit_types(self):
        from opengling.core.models import EditType
        
        assert EditType.SILENCE.value == "silence"
        assert EditType.FILLER_WORD.value == "filler_word"
        assert EditType.BAD_TAKE.value == "bad_take"
        assert EditType.MANUAL.value == "manual"


class TestProcessingConfig:
    """Tests for ProcessingConfig dataclass."""
    
    def test_default_values(self):
        from opengling.core.models import ProcessingConfig
        
        config = ProcessingConfig()
        
        assert config.remove_silences is True
        assert config.silence_threshold == 0.5
        assert config.remove_fillers is True
        assert config.detect_bad_takes is True
        assert config.remove_noise is False
        assert config.auto_zoom is False
        assert config.whisper_model == "large-v3"
    
    def test_custom_values(self):
        from opengling.core.models import ProcessingConfig
        
        config = ProcessingConfig(
            silence_threshold=0.3,
            remove_noise=True,
            whisper_model="medium",
        )
        
        assert config.silence_threshold == 0.3
        assert config.remove_noise is True
        assert config.whisper_model == "medium"
    
    def test_filler_words_default(self):
        from opengling.core.models import ProcessingConfig
        
        config = ProcessingConfig()
        
        assert "um" in config.filler_words
        assert "uh" in config.filler_words
        assert "like" in config.filler_words


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""
    
    def test_time_saved(self):
        from opengling.core.models import ProcessingResult
        
        result = ProcessingResult(
            input_path=Path("test.mp4"),
            original_duration=100.0,
            edited_duration=80.0,
        )
        
        assert result.time_saved == 20.0
        assert result.time_saved_percentage == 20.0
    
    def test_time_saved_zero_duration(self):
        from opengling.core.models import ProcessingResult
        
        result = ProcessingResult(
            input_path=Path("test.mp4"),
            original_duration=0.0,
            edited_duration=0.0,
        )
        
        assert result.time_saved == 0.0
        assert result.time_saved_percentage == 0.0


class TestExportFormat:
    """Tests for ExportFormat enum."""
    
    def test_values(self):
        from opengling.core.models import ExportFormat
        
        assert ExportFormat.MP4.value == "mp4"
        assert ExportFormat.FCPXML.value == "fcpxml"
        assert ExportFormat.PREMIERE_XML.value == "premiere_xml"
        assert ExportFormat.DAVINCI_EDL.value == "davinci_edl"
        assert ExportFormat.SRT.value == "srt"
        assert ExportFormat.VTT.value == "vtt"

