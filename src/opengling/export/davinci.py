"""DaVinci Resolve EDL export."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def export_edl(
    keep_regions: list[tuple[float, float]],
    output_path: Path,
    source_file: str,
    fps: float = 30.0,
) -> Path:
    """
    Export to EDL (Edit Decision List) format for DaVinci Resolve.

    Args:
        keep_regions: List of (start, end) tuples to keep
        output_path: Output file path
        source_file: Path to source media
        fps: Frames per second

    Returns:
        Path to exported file
    """
    logger.info(f"Exporting EDL to {output_path}")

    source_path = Path(source_file).resolve()
    source_name = source_path.name

    lines = []

    # EDL Header
    lines.append(f"TITLE: {Path(source_file).stem}_edited")
    lines.append("FCM: NON-DROP FRAME")
    lines.append("")

    # Add events for each keep region
    timeline_offset = 0.0

    for i, (start, end) in enumerate(keep_regions):
        duration = end - start

        # Event number (3 digits, zero-padded)
        event_num = f"{i + 1:03d}"

        # Reel name (source file, max 8 chars for compatibility)
        reel = source_name[:8].ljust(8)

        # Track type (AA = audio, V = video, AA/V = both)
        track = "AA/V"

        # Cut type (C = cut)
        cut_type = "C"

        # Timecodes
        src_in = seconds_to_timecode(start, fps)
        src_out = seconds_to_timecode(end, fps)
        rec_in = seconds_to_timecode(timeline_offset, fps)
        rec_out = seconds_to_timecode(timeline_offset + duration, fps)

        # EDL line format:
        # EVENT REEL TRACK CUT SRC_IN SRC_OUT REC_IN REC_OUT
        line = f"{event_num}  {reel}  {track}  {cut_type}  {src_in} {src_out} {rec_in} {rec_out}"
        lines.append(line)

        # Optional: Add source file comment
        lines.append(f"* FROM CLIP NAME: {source_name}")
        lines.append("")

        timeline_offset += duration

    # Write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.edl')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    logger.info(f"Exported EDL: {output_path}")
    return output_path


def seconds_to_timecode(seconds: float, fps: float = 30.0) -> str:
    """
    Convert seconds to SMPTE timecode format (HH:MM:SS:FF).

    Args:
        seconds: Time in seconds
        fps: Frames per second

    Returns:
        Timecode string
    """
    total_frames = int(seconds * fps)

    frames = total_frames % int(fps)
    total_seconds = total_frames // int(fps)
    secs = total_seconds % 60
    total_minutes = total_seconds // 60
    mins = total_minutes % 60
    hours = total_minutes // 60

    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def timecode_to_seconds(timecode: str, fps: float = 30.0) -> float:
    """
    Convert SMPTE timecode to seconds.

    Args:
        timecode: Timecode string (HH:MM:SS:FF)
        fps: Frames per second

    Returns:
        Time in seconds
    """
    parts = timecode.split(':')
    if len(parts) != 4:
        raise ValueError(f"Invalid timecode format: {timecode}")

    hours, mins, secs, frames = map(int, parts)

    total_seconds = hours * 3600 + mins * 60 + secs + frames / fps
    return total_seconds

