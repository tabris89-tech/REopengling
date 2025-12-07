"""
OpenGling - Open Source Gling Alternative

AI-powered video editing for content creators.
Automatically removes silences, bad takes, and filler words.
"""

__version__ = "1.0.0"
__author__ = "OpenGling Contributors"

from opengling.core.processor import VideoProcessor
from opengling.core.models import ProcessingConfig, EditDecision, TranscriptSegment

__all__ = [
    "VideoProcessor",
    "ProcessingConfig",
    "EditDecision",
    "TranscriptSegment",
]

