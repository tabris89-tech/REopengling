"""Project save/load functionality for OpenGling."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from opengling.core.models import (
    EditDecision,
    EditType,
    ExportFormat,
    ProcessingConfig,
    ProcessingResult,
    TranscriptSegment,
    TranscriptWord,
    YouTubeMetadata,
    ZoomKeyframe,
)

logger = logging.getLogger(__name__)

PROJECT_VERSION = "1.0.0"


@dataclass
class Project:
    """
    OpenGling project file containing all processing state.

    This allows users to save their work and resume editing later.
    """
    # Metadata
    version: str = PROJECT_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Source information
    input_path: str = ""
    input_filename: str = ""
    original_duration: float = 0.0

    # Configuration
    config: dict = field(default_factory=dict)

    # Transcript
    segments: list[dict] = field(default_factory=list)
    full_transcript: str = ""

    # Edit decisions
    edit_decisions: list[dict] = field(default_factory=list)

    # Optional data
    zoom_keyframes: list[dict] = field(default_factory=list)
    youtube_metadata: Optional[dict] = None

    # Stats
    edited_duration: float = 0.0
    silences_removed: int = 0
    fillers_removed: int = 0
    bad_takes_removed: int = 0

    def save(self, path: Path | str) -> Path:
        """
        Save project to JSON file.

        Args:
            path: Output file path

        Returns:
            Path to saved file
        """
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix('.opengling')

        self.modified_at = datetime.now().isoformat()

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

        logger.info(f"Project saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | str) -> 'Project':
        """
        Load project from JSON file.

        Args:
            path: Path to project file

        Returns:
            Loaded Project instance
        """
        path = Path(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Version compatibility check
        file_version = data.get('version', '0.0.0')
        if file_version != PROJECT_VERSION:
            logger.warning(f"Project version mismatch: {file_version} vs {PROJECT_VERSION}")
            # Could add migration logic here

        project = cls(
            version=data.get('version', PROJECT_VERSION),
            created_at=data.get('created_at', ''),
            modified_at=data.get('modified_at', ''),
            input_path=data.get('input_path', ''),
            input_filename=data.get('input_filename', ''),
            original_duration=data.get('original_duration', 0.0),
            config=data.get('config', {}),
            segments=data.get('segments', []),
            full_transcript=data.get('full_transcript', ''),
            edit_decisions=data.get('edit_decisions', []),
            zoom_keyframes=data.get('zoom_keyframes', []),
            youtube_metadata=data.get('youtube_metadata'),
            edited_duration=data.get('edited_duration', 0.0),
            silences_removed=data.get('silences_removed', 0),
            fillers_removed=data.get('fillers_removed', 0),
            bad_takes_removed=data.get('bad_takes_removed', 0),
        )

        logger.info(f"Project loaded from {path}")
        return project

    @classmethod
    def from_result(
        cls,
        result: ProcessingResult,
        config: ProcessingConfig,
    ) -> 'Project':
        """
        Create a Project from a ProcessingResult.

        Args:
            result: Processing result to convert
            config: Processing configuration used

        Returns:
            New Project instance
        """
        return cls(
            input_path=str(result.input_path) if result.input_path else "",
            input_filename=result.input_path.name if result.input_path else "",
            original_duration=result.original_duration,
            config=_config_to_dict(config),
            segments=[_segment_to_dict(s) for s in result.segments],
            full_transcript=result.full_transcript,
            edit_decisions=[_edit_to_dict(e) for e in result.edit_decisions],
            zoom_keyframes=[_keyframe_to_dict(k) for k in result.zoom_keyframes],
            youtube_metadata=_youtube_to_dict(result.youtube_metadata) if result.youtube_metadata else None,
            edited_duration=result.edited_duration,
            silences_removed=result.silences_removed,
            fillers_removed=result.fillers_removed,
            bad_takes_removed=result.bad_takes_removed,
        )

    def to_result(self) -> ProcessingResult:
        """
        Convert Project back to a ProcessingResult.

        Returns:
            ProcessingResult instance
        """
        return ProcessingResult(
            input_path=Path(self.input_path) if self.input_path else Path(),
            segments=[_dict_to_segment(s) for s in self.segments],
            full_transcript=self.full_transcript,
            edit_decisions=[_dict_to_edit(e) for e in self.edit_decisions],
            zoom_keyframes=[_dict_to_keyframe(k) for k in self.zoom_keyframes],
            youtube_metadata=_dict_to_youtube(self.youtube_metadata) if self.youtube_metadata else None,
            original_duration=self.original_duration,
            edited_duration=self.edited_duration,
            silences_removed=self.silences_removed,
            fillers_removed=self.fillers_removed,
            bad_takes_removed=self.bad_takes_removed,
        )

    def to_config(self) -> ProcessingConfig:
        """
        Convert stored config dict back to ProcessingConfig.

        Returns:
            ProcessingConfig instance
        """
        return _dict_to_config(self.config)


# Serialization helpers

def _config_to_dict(config: ProcessingConfig) -> dict:
    """Convert ProcessingConfig to dictionary."""
    return {
        'remove_silences': config.remove_silences,
        'silence_threshold': config.silence_threshold,
        'silence_padding': config.silence_padding,
        'remove_fillers': config.remove_fillers,
        'filler_words': config.filler_words,
        'detect_bad_takes': config.detect_bad_takes,
        'restart_detection': config.restart_detection,
        'low_confidence_threshold': config.low_confidence_threshold,
        'remove_noise': config.remove_noise,
        'noise_reduction_strength': config.noise_reduction_strength,
        'auto_zoom': config.auto_zoom,
        'zoom_smoothing': config.zoom_smoothing,
        'max_zoom': config.max_zoom,
        'whisper_model': config.whisper_model,
        'language': config.language,
        'generate_youtube_metadata': config.generate_youtube_metadata,
        'ollama_model': config.ollama_model,
        'output_format': config.output_format.value,
        'caption_format': config.caption_format.value if config.caption_format else None,
        'device': config.device,
        'compute_type': config.compute_type,
    }


def _dict_to_config(d: dict) -> ProcessingConfig:
    """Convert dictionary to ProcessingConfig."""
    output_format = ExportFormat(d.get('output_format', 'mp4'))
    caption_format = ExportFormat(d['caption_format']) if d.get('caption_format') else None

    return ProcessingConfig(
        remove_silences=d.get('remove_silences', True),
        silence_threshold=d.get('silence_threshold', 0.5),
        silence_padding=d.get('silence_padding', 0.1),
        remove_fillers=d.get('remove_fillers', True),
        filler_words=d.get('filler_words', []),
        detect_bad_takes=d.get('detect_bad_takes', True),
        restart_detection=d.get('restart_detection', True),
        low_confidence_threshold=d.get('low_confidence_threshold', 0.5),
        remove_noise=d.get('remove_noise', False),
        noise_reduction_strength=d.get('noise_reduction_strength', 0.5),
        auto_zoom=d.get('auto_zoom', False),
        zoom_smoothing=d.get('zoom_smoothing', 0.3),
        max_zoom=d.get('max_zoom', 1.5),
        whisper_model=d.get('whisper_model', 'base'),
        language=d.get('language'),
        generate_youtube_metadata=d.get('generate_youtube_metadata', False),
        ollama_model=d.get('ollama_model', 'llama3.2'),
        output_format=output_format,
        caption_format=caption_format,
        device=d.get('device', 'auto'),
        compute_type=d.get('compute_type', 'auto'),
    )


def _segment_to_dict(segment: TranscriptSegment) -> dict:
    """Convert TranscriptSegment to dictionary."""
    return {
        'text': segment.text,
        'start': segment.start,
        'end': segment.end,
        'words': [
            {
                'word': w.word,
                'start': w.start,
                'end': w.end,
                'confidence': w.confidence,
            }
            for w in segment.words
        ],
        'confidence': segment.confidence,
        'language': segment.language,
    }


def _dict_to_segment(d: dict) -> TranscriptSegment:
    """Convert dictionary to TranscriptSegment."""
    return TranscriptSegment(
        text=d['text'],
        start=d['start'],
        end=d['end'],
        words=[
            TranscriptWord(
                word=w['word'],
                start=w['start'],
                end=w['end'],
                confidence=w['confidence'],
            )
            for w in d.get('words', [])
        ],
        confidence=d.get('confidence', 1.0),
        language=d.get('language', 'en'),
    )


def _edit_to_dict(edit: EditDecision) -> dict:
    """Convert EditDecision to dictionary."""
    return {
        'start': edit.start,
        'end': edit.end,
        'edit_type': edit.edit_type.value,
        'keep': edit.keep,
        'reason': edit.reason,
        'confidence': edit.confidence,
    }


def _dict_to_edit(d: dict) -> EditDecision:
    """Convert dictionary to EditDecision."""
    return EditDecision(
        start=d['start'],
        end=d['end'],
        edit_type=EditType(d['edit_type']),
        keep=d['keep'],
        reason=d.get('reason', ''),
        confidence=d.get('confidence', 1.0),
    )


def _keyframe_to_dict(kf: ZoomKeyframe) -> dict:
    """Convert ZoomKeyframe to dictionary."""
    return {
        'time': kf.time,
        'zoom_level': kf.zoom_level,
        'center_x': kf.center_x,
        'center_y': kf.center_y,
    }


def _dict_to_keyframe(d: dict) -> ZoomKeyframe:
    """Convert dictionary to ZoomKeyframe."""
    return ZoomKeyframe(
        time=d['time'],
        zoom_level=d['zoom_level'],
        center_x=d['center_x'],
        center_y=d['center_y'],
    )


def _youtube_to_dict(yt: YouTubeMetadata) -> dict:
    """Convert YouTubeMetadata to dictionary."""
    return {
        'title': yt.title,
        'description': yt.description,
        'tags': yt.tags,
        'chapters': yt.chapters,
    }


def _dict_to_youtube(d: dict) -> YouTubeMetadata:
    """Convert dictionary to YouTubeMetadata."""
    return YouTubeMetadata(
        title=d['title'],
        description=d['description'],
        tags=d['tags'],
        chapters=d['chapters'],
    )

