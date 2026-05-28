"""Final Cut Pro XML (FCPXML) export."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

logger = logging.getLogger(__name__)


def export_fcpxml(
    keep_regions: list[tuple[float, float]],
    output_path: Path,
    source_file: str,
    total_duration: float,
    fps: float = 30.0,
) -> Path:
    """
    Export to Final Cut Pro XML format.

    Args:
        keep_regions: List of (start, end) tuples to keep
        output_path: Output file path
        source_file: Path to source media
        total_duration: Total source duration
        fps: Frames per second

    Returns:
        Path to exported file
    """
    logger.info(f"Exporting FCPXML to {output_path}")

    # Calculate frame duration
    frame_duration = f"1/{int(fps)}s"

    # Create FCPXML structure
    fcpxml = ET.Element("fcpxml", version="1.10")

    # Resources
    resources = ET.SubElement(fcpxml, "resources")

    # Format resource
    ET.SubElement(
        resources, "format",
        id="r1",
        name="FFVideoFormat1080p30",
        frameDuration=frame_duration,
        width="1920",
        height="1080",
    )

    # Asset resource (the source video)
    source_path = Path(source_file).resolve()
    source_name = source_path.name
    source_uri = source_path.as_uri()

    asset = ET.SubElement(
        resources, "asset",
        id="r2",
        name=source_name,
        src=source_uri,
        start="0s",
        duration=f"{total_duration}s",
        hasVideo="1",
        hasAudio="1",
        format="r1",
    )

    # Media reference
    ET.SubElement(
        asset, "media-rep",
        kind="original-media",
        src=source_uri,
    )

    # Library
    library = ET.SubElement(fcpxml, "library")

    # Event
    event = ET.SubElement(
        library, "event",
        name="OpenGling Export",
    )

    # Project
    project = ET.SubElement(
        event, "project",
        name=f"{Path(source_file).stem}_edited",
    )

    # Sequence
    edited_duration = sum(end - start for start, end in keep_regions)
    sequence = ET.SubElement(
        project, "sequence",
        format="r1",
        duration=f"{edited_duration}s",
        tcStart="0s",
        tcFormat="NDF",
    )

    # Spine (main timeline)
    spine = ET.SubElement(sequence, "spine")

    # Add clips for each keep region
    timeline_offset = 0.0
    for i, (start, end) in enumerate(keep_regions):
        duration = end - start

        ET.SubElement(
            spine, "asset-clip",
            ref="r2",
            offset=f"{timeline_offset}s",
            name=f"Clip {i + 1}",
            start=f"{start}s",
            duration=f"{duration}s",
            format="r1",
            tcFormat="NDF",
        )

        timeline_offset += duration

    # Format XML nicely
    xml_string = ET.tostring(fcpxml, encoding='unicode')
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in xml_pretty.split('\n') if line.strip()]
    xml_pretty = '\n'.join(lines)

    # Write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.fcpxml')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        # Write without the XML declaration (we added it manually)
        f.write('\n'.join(xml_pretty.split('\n')[1:]))

    logger.info(f"Exported FCPXML: {output_path}")
    return output_path


def seconds_to_fcpxml_time(seconds: float, fps: float = 30.0) -> str:
    """Convert seconds to FCPXML time format."""
    frames = int(seconds * fps)
    return f"{frames}/{int(fps)}s"

