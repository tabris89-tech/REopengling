"""Data models for OpenGling processing pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class EditType(str, Enum):
    """Type of edit to be made."""
    SILENCE = "silence"
    FILLER_WORD = "filler_word"
    BAD_TAKE = "bad_take"
    MANUAL = "manual"


class ExportFormat(str, Enum):
    """Supported export formats."""
    MP4 = "mp4"
    FCPXML = "fcpxml"
    PREMIERE_XML = "premiere_xml"
    DAVINCI_EDL = "davinci_edl"
    SRT = "srt"
    VTT = "vtt"


@dataclass
class TranscriptWord:
    """A single word in the transcript with timing information."""
    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float  # 0.0 to 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TranscriptSegment:
    """A segment of transcript (typically a sentence or phrase)."""
    text: str
    start: float  # seconds
    end: float  # seconds
    words: list[TranscriptWord] = field(default_factory=list)
    confidence: float = 1.0
    language: str = "en"

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class EditDecision:
    """Represents a decision to cut/keep a portion of the video."""
    start: float  # seconds
    end: float  # seconds
    edit_type: EditType
    keep: bool = False  # True = keep, False = cut
    reason: str = ""
    confidence: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ZoomKeyframe:
    """A keyframe for auto-zoom effect."""
    time: float  # seconds
    zoom_level: float  # 1.0 = no zoom, 2.0 = 2x zoom
    center_x: float  # 0.0 to 1.0, relative position
    center_y: float  # 0.0 to 1.0, relative position


@dataclass
class ProcessingConfig:
    """Configuration for video processing."""
    # Silence detection
    remove_silences: bool = True  # Whether to remove silences
    silence_threshold: float = 0.5  # seconds - minimum silence to remove
    silence_padding: float = 0.1  # seconds - padding to keep around speech

    # Filler word detection
    remove_fillers: bool = True
    filler_words: list[str] = field(default_factory=lambda: [
        "um", "uh", "uhm", "uhh", "umm",
        "like", "you know", "i mean", "basically",
        "actually", "literally", "so", "well",
        "right", "okay", "ok", "er", "ah"
    ])

    # Bad takes detection
    detect_bad_takes: bool = True
    restart_detection: bool = True  # Detect when speaker restarts sentence
    low_confidence_threshold: float = 0.5  # Whisper confidence threshold

    # Noise removal
    remove_noise: bool = False  # Off by default, enable with --noise flag
    noise_reduction_strength: float = 0.5  # 0.0 to 1.0

    # Auto-zoom
    auto_zoom: bool = False
    zoom_smoothing: float = 0.3  # seconds
    max_zoom: float = 1.5

    # Transcription
    whisper_model: str = "large-v3"  # tiny, base, small, medium, large-v3
    language: Optional[str] = None  # Auto-detect if None

    # Time range (optional, None = process entire file)
    start_time: Optional[float] = None  # seconds - start of range to process
    end_time: Optional[float] = None  # seconds - end of range to process

    # YouTube generation
    generate_youtube_metadata: bool = False
    ollama_model: str = "llama3.2"

    # Output
    output_format: ExportFormat = ExportFormat.MP4
    caption_format: Optional[ExportFormat] = None  # SRT or VTT

    # Performance
    device: str = "auto"  # auto, cuda, cpu
    compute_type: str = "auto"  # auto, float16, int8


@dataclass
class YouTubeMetadata:
    """Generated YouTube metadata."""
    title: str
    description: str
    tags: list[str]
    chapters: list[tuple[float, str]]  # (timestamp, chapter_name)


@dataclass
class ProcessingResult:
    """Result of video processing."""
    input_path: Path
    output_path: Optional[Path] = None

    # Transcript
    segments: list[TranscriptSegment] = field(default_factory=list)
    full_transcript: str = ""

    # Edits
    edit_decisions: list[EditDecision] = field(default_factory=list)
    zoom_keyframes: list[ZoomKeyframe] = field(default_factory=list)

    # Stats
    original_duration: float = 0.0
    edited_duration: float = 0.0
    silences_removed: int = 0
    fillers_removed: int = 0
    bad_takes_removed: int = 0

    # YouTube
    youtube_metadata: Optional[YouTubeMetadata] = None

    # Captions
    caption_file: Optional[Path] = None

    @property
    def time_saved(self) -> float:
        return self.original_duration - self.edited_duration

    @property
    def time_saved_percentage(self) -> float:
        if self.original_duration == 0:
            return 0.0
        return (self.time_saved / self.original_duration) * 100


# --- Time parsing utilities ---

_TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2}):(\d{2})$")


def parse_time_to_seconds(time_str: str) -> float:
    """
    Parse HH:MM:SS string to seconds.

    Args:
        time_str: Time string in format HH:MM:SS or MM:SS

    Returns:
        Time in seconds

    Raises:
        ValueError: If format is invalid
    """
    time_str = time_str.strip()

    # Try HH:MM:SS
    match = _TIME_PATTERN.match(time_str)
    if match:
        hours, minutes, seconds = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return hours * 3600 + minutes * 60 + seconds

    # Try MM:SS
    parts = time_str.split(":")
    if len(parts) == 2:
        try:
            minutes, seconds = int(parts[0]), int(parts[1])
            return minutes * 60 + seconds
        except ValueError:
            pass

    # Try plain seconds
    try:
        return float(time_str)
    except ValueError:
        raise ValueError(
            f"Invalid time format: '{time_str}'. "
            f"Use HH:MM:SS, MM:SS, or seconds."
        )


def format_seconds_to_time(seconds: float) -> str:
    """Format seconds to HH:MM:SS string."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def validate_time_range(
    start_time: Optional[float],
    end_time: Optional[float],
    video_duration: float,
) -> list[str]:
    """
    Validate time range parameters.

    Returns list of error messages (empty if valid).
    """
    errors = []

    if start_time is not None and start_time < 0:
        errors.append("Start time cannot be negative.")

    if end_time is not None and end_time < 0:
        errors.append("End time cannot be negative.")

    if start_time is not None and end_time is not None:
        if start_time >= end_time:
            errors.append("Start time must be before end time.")

        if end_time - start_time < 10:
            errors.append("Time range must be at least 10 seconds.")

        if end_time - start_time > 8 * 3600:
            errors.append("Time range cannot exceed 8 hours.")

    if end_time is not None and video_duration > 0 and end_time > video_duration:
        from opengling.core.models import format_seconds_to_time
        dur_str = format_seconds_to_time(video_duration)
        errors.append(f"End time exceeds video duration ({dur_str}).")

    return errors

