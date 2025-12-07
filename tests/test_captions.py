"""Tests for caption export."""

import pytest
from pathlib import Path


class TestSRTExport:
    """Tests for SRT caption export."""
    
    def test_export_srt(self, sample_transcript, temp_dir):
        from opengling.export.captions import export_srt
        
        output_path = temp_dir / "test.srt"
        result = export_srt(sample_transcript, output_path)
        
        assert result.exists()
        assert result.suffix == ".srt"
        
        content = result.read_text()
        
        # Check SRT format
        assert "1\n" in content  # Sequence number
        assert "-->" in content  # Timestamp separator
        assert "Hello" in content
    
    def test_srt_time_format(self):
        from opengling.export.captions import seconds_to_srt_time
        
        # Test various times
        assert seconds_to_srt_time(0.0) == "00:00:00,000"
        assert seconds_to_srt_time(1.5) == "00:00:01,500"
        assert seconds_to_srt_time(65.123) == "00:01:05,123"
        assert seconds_to_srt_time(3661.5) == "01:01:01,500"


class TestVTTExport:
    """Tests for VTT caption export."""
    
    def test_export_vtt(self, sample_transcript, temp_dir):
        from opengling.export.captions import export_vtt
        
        output_path = temp_dir / "test.vtt"
        result = export_vtt(sample_transcript, output_path)
        
        assert result.exists()
        assert result.suffix == ".vtt"
        
        content = result.read_text()
        
        # Check VTT format
        assert "WEBVTT" in content  # Header
        assert "-->" in content
        assert "Hello" in content
    
    def test_vtt_time_format(self):
        from opengling.export.captions import seconds_to_vtt_time
        
        # VTT uses . instead of , for milliseconds
        assert seconds_to_vtt_time(0.0) == "00:00:00.000"
        assert seconds_to_vtt_time(1.5) == "00:00:01.500"


class TestTextWrapping:
    """Tests for text wrapping function."""
    
    def test_short_text(self):
        from opengling.export.captions import wrap_text
        
        result = wrap_text("Hello world", max_chars=42)
        assert result == ["Hello world"]
    
    def test_long_text(self):
        from opengling.export.captions import wrap_text
        
        long_text = "This is a very long sentence that should be wrapped across multiple lines for better readability"
        result = wrap_text(long_text, max_chars=42)
        
        assert len(result) > 1
        for line in result:
            assert len(line) <= 42


class TestCaptionAdjustment:
    """Tests for caption timing adjustment."""
    
    def test_adjust_captions_for_edits(self):
        from opengling.export.captions import adjust_captions_for_edits
        from opengling.core.models import TranscriptSegment
        
        segments = [
            TranscriptSegment(text="First", start=0.0, end=1.0),
            TranscriptSegment(text="Second", start=2.0, end=3.0),  # After cut region
            TranscriptSegment(text="Third", start=4.0, end=5.0),
        ]
        
        # Keep regions: 0-1.5 and 3.5-5
        # Cuts: 1.5-3.5 (2 seconds removed)
        keep_regions = [(0.0, 1.5), (3.5, 5.0)]
        
        adjusted = adjust_captions_for_edits(segments, keep_regions)
        
        # First segment should be unchanged (within first keep region)
        assert adjusted[0].text == "First"
        assert adjusted[0].start == 0.0
        
        # Segments after cut should have adjusted timing
        # The second keep region starts at timeline offset 1.5 (first region duration)
        # Third segment (4.0-5.0) overlaps with second keep region (3.5-5.0)
        # Overlap is 4.0-5.0, new start = 1.5 + (4.0 - 3.5) = 2.0

