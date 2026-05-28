"""Core processing modules for OpenGling."""

from opengling.core.models import (
    EditDecision,
    ProcessingConfig,
    ProcessingResult,
    TranscriptSegment,
)
from opengling.core.processor import VideoProcessor

__all__ = [
    "ProcessingConfig",
    "EditDecision",
    "TranscriptSegment",
    "ProcessingResult",
    "VideoProcessor",
]

