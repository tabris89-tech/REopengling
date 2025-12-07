"""Core processing modules for OpenGling."""

from opengling.core.models import (
    ProcessingConfig,
    EditDecision,
    TranscriptSegment,
    ProcessingResult,
)
from opengling.core.processor import VideoProcessor

__all__ = [
    "ProcessingConfig",
    "EditDecision",
    "TranscriptSegment",
    "ProcessingResult",
    "VideoProcessor",
]

