"""Tests for silence detection."""

import pytest
from pathlib import Path


class TestSilenceDetector:
    """Tests for SilenceDetector class."""
    
    def test_init(self, default_config):
        from opengling.core.silence import SilenceDetector
        
        detector = SilenceDetector(default_config)
        assert detector.config == default_config
    
    def test_disabled_returns_empty(self, default_config, sample_audio_path):
        from opengling.core.silence import SilenceDetector
        from opengling.core.models import ProcessingConfig
        
        config = ProcessingConfig(remove_silences=False)
        detector = SilenceDetector(config)
        
        result = detector.detect_silences(sample_audio_path)
        assert result == []
    
    @pytest.mark.skipif(
        not pytest.importorskip("webrtcvad", reason="webrtcvad not installed"),
        reason="webrtcvad not installed"
    )
    def test_detect_silences(self, default_config, sample_audio_path):
        from opengling.core.silence import SilenceDetector
        
        detector = SilenceDetector(default_config)
        silences = detector.detect_silences(sample_audio_path)
        
        # Should detect some silences
        assert isinstance(silences, list)
        
        for silence in silences:
            assert silence.start < silence.end
            assert not silence.keep
            assert "Silence" in silence.reason


class TestMergeOverlappingRegions:
    """Tests for merge_overlapping_regions function."""
    
    def test_empty_list(self):
        from opengling.core.silence import merge_overlapping_regions
        
        result = merge_overlapping_regions([])
        assert result == []
    
    def test_no_overlap(self, sample_edit_decisions):
        from opengling.core.silence import merge_overlapping_regions
        
        # These don't overlap
        result = merge_overlapping_regions(sample_edit_decisions)
        assert len(result) == len(sample_edit_decisions)
    
    def test_overlapping(self):
        from opengling.core.silence import merge_overlapping_regions
        from opengling.core.models import EditDecision, EditType
        
        edits = [
            EditDecision(start=1.0, end=2.0, edit_type=EditType.SILENCE, keep=False, reason="A"),
            EditDecision(start=1.5, end=2.5, edit_type=EditType.SILENCE, keep=False, reason="B"),
        ]
        
        result = merge_overlapping_regions(edits)
        
        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].end == 2.5
    
    def test_adjacent(self):
        from opengling.core.silence import merge_overlapping_regions
        from opengling.core.models import EditDecision, EditType
        
        # Adjacent within 50ms tolerance
        edits = [
            EditDecision(start=1.0, end=2.0, edit_type=EditType.SILENCE, keep=False, reason="A"),
            EditDecision(start=2.02, end=3.0, edit_type=EditType.SILENCE, keep=False, reason="B"),
        ]
        
        result = merge_overlapping_regions(edits)
        
        # Should merge due to 50ms tolerance
        assert len(result) == 1
        assert result[0].start == 1.0
        assert result[0].end == 3.0

