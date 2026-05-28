"""Caption and subtitle export (SRT, VTT)."""

from __future__ import annotations

import logging
from pathlib import Path

from opengling.core.models import ExportFormat, TranscriptSegment

logger = logging.getLogger(__name__)


def export_captions(
    segments: list[TranscriptSegment],
    output_path: Path | str,
    format: ExportFormat,
) -> Path:
    """
    Export transcript to caption format.

    Args:
        segments: Transcript segments
        output_path: Output file path
        format: Caption format (SRT or VTT)

    Returns:
        Path to exported file
    """
    output_path = Path(output_path)

    if format == ExportFormat.SRT:
        return export_srt(segments, output_path)
    elif format == ExportFormat.VTT:
        return export_vtt(segments, output_path)
    else:
        raise ValueError(f"Unsupported caption format: {format}")


def export_srt(
    segments: list[TranscriptSegment],
    output_path: Path,
) -> Path:
    """
    Export to SRT (SubRip) format.

    Args:
        segments: Transcript segments
        output_path: Output file path

    Returns:
        Path to exported file
    """
    logger.info(f"Exporting SRT to {output_path}")

    lines = []

    for i, segment in enumerate(segments, 1):
        # Sequence number
        lines.append(str(i))

        # Timestamps
        start_tc = seconds_to_srt_time(segment.start)
        end_tc = seconds_to_srt_time(segment.end)
        lines.append(f"{start_tc} --> {end_tc}")

        # Text (can be multiple lines)
        text = segment.text.strip()
        # Split long lines (max ~42 chars per line for readability)
        wrapped_lines = wrap_text(text, max_chars=42)
        lines.extend(wrapped_lines)

        # Blank line separator
        lines.append("")

    # Write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.srt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Exported SRT: {output_path}")
    return output_path


def export_vtt(
    segments: list[TranscriptSegment],
    output_path: Path,
) -> Path:
    """
    Export to WebVTT format.

    Args:
        segments: Transcript segments
        output_path: Output file path

    Returns:
        Path to exported file
    """
    logger.info(f"Exporting VTT to {output_path}")

    lines = []

    # VTT header
    lines.append("WEBVTT")
    lines.append("")

    for i, segment in enumerate(segments, 1):
        # Optional cue identifier
        lines.append(f"cue-{i}")

        # Timestamps (VTT uses . instead of , for milliseconds)
        start_tc = seconds_to_vtt_time(segment.start)
        end_tc = seconds_to_vtt_time(segment.end)
        lines.append(f"{start_tc} --> {end_tc}")

        # Text
        text = segment.text.strip()
        wrapped_lines = wrap_text(text, max_chars=42)
        lines.extend(wrapped_lines)

        # Blank line separator
        lines.append("")

    # Write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.vtt')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Exported VTT: {output_path}")
    return output_path


def seconds_to_srt_time(seconds: float) -> str:
    """
    Convert seconds to SRT time format (HH:MM:SS,mmm).

    Args:
        seconds: Time in seconds

    Returns:
        SRT timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def seconds_to_vtt_time(seconds: float) -> str:
    """
    Convert seconds to VTT time format (HH:MM:SS.mmm).

    Args:
        seconds: Time in seconds

    Returns:
        VTT timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)

    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def wrap_text(text: str, max_chars: int = 42) -> list[str]:
    """
    Wrap text to fit within max characters per line.

    Args:
        text: Text to wrap
        max_chars: Maximum characters per line

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_length = len(word)

        if current_length + word_length + 1 <= max_chars:
            current_line.append(word)
            current_length += word_length + 1
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length + 1

    if current_line:
        lines.append(' '.join(current_line))

    return lines if lines else [text]


def adjust_captions_for_edits(
    segments: list[TranscriptSegment],
    keep_regions: list[tuple[float, float]],
) -> list[TranscriptSegment]:
    """
    Adjust caption timing based on edit regions.

    This recalculates timestamps so captions align with the edited video.

    Args:
        segments: Original transcript segments
        keep_regions: List of (start, end) regions kept in edited video

    Returns:
        Adjusted transcript segments
    """
    adjusted = []
    timeline_offset = 0.0

    for region_start, region_end in keep_regions:
        for segment in segments:
            # Check if segment overlaps with this keep region
            if segment.end <= region_start or segment.start >= region_end:
                continue

            # Calculate overlap
            overlap_start = max(segment.start, region_start)
            overlap_end = min(segment.end, region_end)

            if overlap_end > overlap_start:
                # Calculate new timestamps
                new_start = timeline_offset + (overlap_start - region_start)
                new_end = timeline_offset + (overlap_end - region_start)

                # Create adjusted segment
                adjusted.append(TranscriptSegment(
                    text=segment.text,
                    start=new_start,
                    end=new_end,
                    words=segment.words,  # Note: word timestamps would need adjustment too
                    confidence=segment.confidence,
                    language=segment.language,
                ))

        timeline_offset += region_end - region_start

    return adjusted

