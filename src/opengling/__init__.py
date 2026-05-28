"""
OpenGling - Open Source Gling Alternative

AI-powered video editing for content creators.
Automatically removes silences, bad takes, and filler words.
"""

__version__ = "1.0.0"
__author__ = "OpenGling Contributors"

from opengling.core.models import EditDecision, ProcessingConfig, TranscriptSegment
from opengling.core.processor import VideoProcessor

__all__ = [
    "VideoProcessor",
    "ProcessingConfig",
    "EditDecision",
    "TranscriptSegment",
]

