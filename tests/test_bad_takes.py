"""Tests for bad takes detection."""



class TestBadTakesDetector:
    """Tests for BadTakesDetector class."""

    def test_init(self, default_config):
        from opengling.core.bad_takes import BadTakesDetector

        detector = BadTakesDetector(default_config)
        assert detector.config == default_config

    def test_disabled_returns_empty(self, sample_transcript):
        from opengling.core.bad_takes import BadTakesDetector
        from opengling.core.models import ProcessingConfig

        config = ProcessingConfig(detect_bad_takes=False)
        detector = BadTakesDetector(config)

        result = detector.detect_bad_takes(sample_transcript)
        assert result == []

    def test_detect_stutters(self, sample_transcript, default_config):
        from opengling.core.bad_takes import BadTakesDetector

        detector = BadTakesDetector(default_config)
        bad_takes = detector.detect_bad_takes(sample_transcript)

        # Should detect "This this" stutter in third segment
        stutters = [b for b in bad_takes if "stutter" in b.reason.lower() or "repetition" in b.reason.lower()]
        assert len(stutters) >= 1

    def test_detect_low_confidence(self):
        from opengling.core.bad_takes import BadTakesDetector
        from opengling.core.models import ProcessingConfig, TranscriptSegment, TranscriptWord

        config = ProcessingConfig(low_confidence_threshold=0.5)
        detector = BadTakesDetector(config)

        # Create segment with low confidence words
        segment = TranscriptSegment(
            text="mumble mumble unclear",
            start=0.0,
            end=2.0,
            words=[
                TranscriptWord(word="mumble", start=0.0, end=0.5, confidence=0.3),
                TranscriptWord(word="mumble", start=0.6, end=1.0, confidence=0.35),
                TranscriptWord(word="unclear", start=1.1, end=2.0, confidence=0.4),
            ],
            confidence=0.35,
        )

        bad_takes = detector.detect_bad_takes([segment])

        # Should detect low confidence region
        low_conf = [b for b in bad_takes if "confidence" in b.reason.lower()]
        assert len(low_conf) >= 1


class TestBadTakeStatistics:
    """Tests for bad take statistics function."""

    def test_get_bad_take_statistics(self):
        from opengling.core.bad_takes import get_bad_take_statistics
        from opengling.core.models import EditDecision, EditType

        edits = [
            EditDecision(start=0, end=0.5, edit_type=EditType.BAD_TAKE, keep=False, reason="Low confidence speech"),
            EditDecision(start=1, end=1.5, edit_type=EditType.BAD_TAKE, keep=False, reason="Sentence restart detected"),
            EditDecision(start=2, end=2.5, edit_type=EditType.BAD_TAKE, keep=False, reason="Stutter/repetition: 'the'"),
            EditDecision(start=3, end=3.5, edit_type=EditType.BAD_TAKE, keep=False, reason="Incomplete sentence"),
            EditDecision(start=4, end=4.5, edit_type=EditType.SILENCE, keep=False, reason="Silence"),  # Not a bad take
        ]

        stats = get_bad_take_statistics(edits)

        assert stats["low_confidence"] == 1
        assert stats["restarts"] == 1
        assert stats["stutters"] == 1
        assert stats["incomplete"] == 1


class TestPhraseSimilarity:
    """Tests for phrase similarity detection."""

    def test_identical_phrases(self):
        from opengling.core.bad_takes import BadTakesDetector
        from opengling.core.models import ProcessingConfig

        detector = BadTakesDetector(ProcessingConfig())

        assert detector._phrases_similar("hello world", "hello world") is True

    def test_similar_phrases(self):
        from opengling.core.bad_takes import BadTakesDetector
        from opengling.core.models import ProcessingConfig

        detector = BadTakesDetector(ProcessingConfig())

        # 2/3 = 66% overlap - under 70% threshold
        assert detector._phrases_similar("the quick fox", "the lazy fox") is False

        # 3/3 = 100% overlap
        assert detector._phrases_similar("i went to", "i went to") is True

    def test_stutter_detection(self):
        from opengling.core.bad_takes import BadTakesDetector
        from opengling.core.models import ProcessingConfig

        detector = BadTakesDetector(ProcessingConfig())

        # "th" is a stutter of "the"
        assert detector._is_stutter("th", "the") is True
        assert detector._is_stutter("the", "the") is True
        assert detector._is_stutter("hello", "world") is False

