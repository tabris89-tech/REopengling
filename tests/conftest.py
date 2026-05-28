"""Pytest configuration and fixtures for OpenGling tests."""

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_path(temp_dir: Path) -> Path:
    """Create a simple test WAV file."""
    try:
        from scipy.io import wavfile
    except ImportError:
        pytest.skip("scipy not installed")

    # Generate 5 seconds of audio with speech-like patterns
    sample_rate = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Create pattern: speech, silence, speech, silence, speech
    audio = np.zeros_like(t)

    # Speech segment 1: 0-1s
    mask1 = (t >= 0) & (t < 1)
    audio[mask1] = 0.5 * np.sin(2 * np.pi * 200 * t[mask1]) + 0.3 * np.sin(2 * np.pi * 400 * t[mask1])

    # Silence: 1-2s (already zeros)

    # Speech segment 2: 2-3.5s
    mask2 = (t >= 2) & (t < 3.5)
    audio[mask2] = 0.5 * np.sin(2 * np.pi * 250 * t[mask2]) + 0.2 * np.sin(2 * np.pi * 500 * t[mask2])

    # Silence: 3.5-4s

    # Speech segment 3: 4-5s
    mask3 = (t >= 4) & (t < 5)
    audio[mask3] = 0.4 * np.sin(2 * np.pi * 180 * t[mask3])

    # Add some noise
    audio += np.random.randn(len(audio)) * 0.01

    # Normalize to int16 range
    audio = (audio * 32767).astype(np.int16)

    audio_path = temp_dir / "test_audio.wav"
    wavfile.write(str(audio_path), sample_rate, audio)

    return audio_path


@pytest.fixture
def sample_video_path(temp_dir: Path, sample_audio_path: Path) -> Path:
    """Create a simple test video file (requires ffmpeg)."""
    try:
        import ffmpeg
    except ImportError:
        pytest.skip("ffmpeg-python not installed")

    video_path = temp_dir / "test_video.mp4"

    # Create a simple video with the audio
    try:
        (
            ffmpeg
            .input("color=c=blue:size=640x480:rate=30:duration=5", f="lavfi")
            .output(
                str(video_path),
                vcodec="libx264",
                acodec="aac",
                shortest=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error:
        pytest.skip("ffmpeg not available")

    return video_path


@pytest.fixture
def sample_transcript():
    """Create sample transcript segments for testing."""
    from opengling.core.models import TranscriptSegment, TranscriptWord

    return [
        TranscriptSegment(
            text="Hello um this is a test",
            start=0.0,
            end=2.0,
            words=[
                TranscriptWord(word="Hello", start=0.0, end=0.3, confidence=0.95),
                TranscriptWord(word="um", start=0.4, end=0.6, confidence=0.85),
                TranscriptWord(word="this", start=0.7, end=0.9, confidence=0.92),
                TranscriptWord(word="is", start=1.0, end=1.1, confidence=0.90),
                TranscriptWord(word="a", start=1.2, end=1.3, confidence=0.88),
                TranscriptWord(word="test", start=1.4, end=2.0, confidence=0.94),
            ],
            confidence=0.9,
        ),
        TranscriptSegment(
            text="You know like testing things",
            start=2.5,
            end=4.0,
            words=[
                TranscriptWord(word="You", start=2.5, end=2.6, confidence=0.91),
                TranscriptWord(word="know", start=2.65, end=2.8, confidence=0.89),
                TranscriptWord(word="like", start=2.9, end=3.1, confidence=0.87),
                TranscriptWord(word="testing", start=3.2, end=3.5, confidence=0.93),
                TranscriptWord(word="things", start=3.6, end=4.0, confidence=0.90),
            ],
            confidence=0.88,
        ),
        TranscriptSegment(
            text="This this is a restart",
            start=4.5,
            end=6.0,
            words=[
                TranscriptWord(word="This", start=4.5, end=4.7, confidence=0.80),
                TranscriptWord(word="this", start=4.8, end=5.0, confidence=0.85),
                TranscriptWord(word="is", start=5.1, end=5.2, confidence=0.90),
                TranscriptWord(word="a", start=5.3, end=5.4, confidence=0.88),
                TranscriptWord(word="restart", start=5.5, end=6.0, confidence=0.92),
            ],
            confidence=0.85,
        ),
    ]


@pytest.fixture
def default_config():
    """Create a default ProcessingConfig for testing."""
    from opengling.core.models import ProcessingConfig
    return ProcessingConfig()


@pytest.fixture
def sample_edit_decisions():
    """Create sample edit decisions for testing."""
    from opengling.core.models import EditDecision, EditType

    return [
        EditDecision(
            start=1.0,
            end=2.0,
            edit_type=EditType.SILENCE,
            keep=False,
            reason="Silence detected",
        ),
        EditDecision(
            start=3.5,
            end=4.0,
            edit_type=EditType.FILLER_WORD,
            keep=False,
            reason="Filler: um",
        ),
        EditDecision(
            start=5.0,
            end=5.5,
            edit_type=EditType.BAD_TAKE,
            keep=False,
            reason="Stutter detected",
        ),
    ]

