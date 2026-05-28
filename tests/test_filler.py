"""Tests for filler word detection."""



class TestFillerDetector:
    """Tests for FillerDetector class."""

    def test_init(self, default_config):
        from opengling.core.filler import FillerDetector

        detector = FillerDetector(default_config)
        assert "um" in detector.filler_words
        assert "uh" in detector.filler_words

    def test_disabled_returns_empty(self, sample_transcript):
        from opengling.core.filler import FillerDetector
        from opengling.core.models import ProcessingConfig

        config = ProcessingConfig(remove_fillers=False)
        detector = FillerDetector(config)

        result = detector.detect_fillers(sample_transcript)
        assert result == []

    def test_detect_um(self, sample_transcript, default_config):
        from opengling.core.filler import FillerDetector

        detector = FillerDetector(default_config)
        fillers = detector.detect_fillers(sample_transcript, use_nlp=False)

        # Should find "um" in first segment
        um_fillers = [f for f in fillers if "um" in f.reason.lower()]
        assert len(um_fillers) >= 1

    def test_detect_multiword_fillers(self, sample_transcript, default_config):
        from opengling.core.filler import FillerDetector

        detector = FillerDetector(default_config)
        fillers = detector.detect_fillers(sample_transcript, use_nlp=False)

        # Should find "you know" in second segment
        you_know_fillers = [f for f in fillers if "you know" in f.reason.lower()]
        assert len(you_know_fillers) >= 1


class TestFillerStatistics:
    """Tests for filler statistics function."""

    def test_get_filler_statistics(self):
        from opengling.core.filler import get_filler_statistics
        from opengling.core.models import EditDecision, EditType

        edits = [
            EditDecision(start=0, end=0.5, edit_type=EditType.FILLER_WORD, keep=False, reason="Filler word: 'um'"),
            EditDecision(start=1, end=1.5, edit_type=EditType.FILLER_WORD, keep=False, reason="Filler word: 'um'"),
            EditDecision(start=2, end=2.5, edit_type=EditType.FILLER_WORD, keep=False, reason="Filler word: 'like'"),
            EditDecision(start=3, end=3.5, edit_type=EditType.SILENCE, keep=False, reason="Silence"),  # Not a filler
        ]

        stats = get_filler_statistics(edits)

        assert stats["um"] == 2
        assert stats["like"] == 1
        assert len(stats) == 2  # Only filler words

