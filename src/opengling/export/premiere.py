"""Adobe Premiere Pro XML export."""

from __future__ import annotations

import logging
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

logger = logging.getLogger(__name__)


def export_premiere_xml(
    keep_regions: list[tuple[float, float]],
    output_path: Path,
    source_file: str,
    total_duration: float,
    fps: float = 30.0,
) -> Path:
    """
    Export to Adobe Premiere Pro XML format.
    
    Args:
        keep_regions: List of (start, end) tuples to keep
        output_path: Output file path
        source_file: Path to source media
        total_duration: Total source duration
        fps: Frames per second
        
    Returns:
        Path to exported file
    """
    logger.info(f"Exporting Premiere XML to {output_path}")
    
    # Calculate timebase
    timebase = int(fps)
    
    # Create XML structure
    xmeml = ET.Element("xmeml", version="5")
    
    # Sequence
    sequence = ET.SubElement(xmeml, "sequence")
    ET.SubElement(sequence, "name").text = f"{Path(source_file).stem}_edited"
    ET.SubElement(sequence, "uuid").text = str(uuid.uuid4())
    
    # Duration
    edited_duration = sum(end - start for start, end in keep_regions)
    ET.SubElement(sequence, "duration").text = str(int(edited_duration * fps))
    
    # Rate
    rate = ET.SubElement(sequence, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "FALSE"
    
    # Timecode
    timecode = ET.SubElement(sequence, "timecode")
    tc_rate = ET.SubElement(timecode, "rate")
    ET.SubElement(tc_rate, "timebase").text = str(timebase)
    ET.SubElement(tc_rate, "ntsc").text = "FALSE"
    ET.SubElement(timecode, "string").text = "00:00:00:00"
    ET.SubElement(timecode, "frame").text = "0"
    ET.SubElement(timecode, "displayformat").text = "NDF"
    
    # Media
    media = ET.SubElement(sequence, "media")
    
    # Video track
    video = ET.SubElement(media, "video")
    video_track = ET.SubElement(video, "track")
    
    # Add clips
    timeline_offset = 0
    for i, (start, end) in enumerate(keep_regions):
        duration = end - start
        
        clipitem = ET.SubElement(video_track, "clipitem", id=f"clipitem-{i+1}")
        ET.SubElement(clipitem, "name").text = f"Clip {i + 1}"
        ET.SubElement(clipitem, "duration").text = str(int(duration * fps))
        
        # Item rate
        item_rate = ET.SubElement(clipitem, "rate")
        ET.SubElement(item_rate, "timebase").text = str(timebase)
        ET.SubElement(item_rate, "ntsc").text = "FALSE"
        
        # Timeline position
        ET.SubElement(clipitem, "start").text = str(int(timeline_offset * fps))
        ET.SubElement(clipitem, "end").text = str(int((timeline_offset + duration) * fps))
        
        # Source position
        ET.SubElement(clipitem, "in").text = str(int(start * fps))
        ET.SubElement(clipitem, "out").text = str(int(end * fps))
        
        # File reference
        source_path = Path(source_file).resolve()
        file_elem = ET.SubElement(clipitem, "file", id=f"file-{i+1}")
        ET.SubElement(file_elem, "name").text = source_path.name
        ET.SubElement(file_elem, "pathurl").text = source_path.as_uri()
        
        file_rate = ET.SubElement(file_elem, "rate")
        ET.SubElement(file_rate, "timebase").text = str(timebase)
        ET.SubElement(file_rate, "ntsc").text = "FALSE"
        
        ET.SubElement(file_elem, "duration").text = str(int(total_duration * fps))
        
        # Media info
        file_media = ET.SubElement(file_elem, "media")
        file_video = ET.SubElement(file_media, "video")
        file_audio = ET.SubElement(file_media, "audio")
        
        timeline_offset += duration
    
    # Audio track (linked to video)
    audio = ET.SubElement(media, "audio")
    audio_track = ET.SubElement(audio, "track")
    
    timeline_offset = 0
    for i, (start, end) in enumerate(keep_regions):
        duration = end - start
        
        clipitem = ET.SubElement(audio_track, "clipitem", id=f"audio-clipitem-{i+1}")
        ET.SubElement(clipitem, "name").text = f"Clip {i + 1}"
        ET.SubElement(clipitem, "duration").text = str(int(duration * fps))
        
        item_rate = ET.SubElement(clipitem, "rate")
        ET.SubElement(item_rate, "timebase").text = str(timebase)
        ET.SubElement(item_rate, "ntsc").text = "FALSE"
        
        ET.SubElement(clipitem, "start").text = str(int(timeline_offset * fps))
        ET.SubElement(clipitem, "end").text = str(int((timeline_offset + duration) * fps))
        ET.SubElement(clipitem, "in").text = str(int(start * fps))
        ET.SubElement(clipitem, "out").text = str(int(end * fps))
        
        # Link to video file
        file_elem = ET.SubElement(clipitem, "file", id=f"file-{i+1}")
        
        timeline_offset += duration
    
    # Format XML nicely
    xml_string = ET.tostring(xmeml, encoding='unicode')
    xml_pretty = minidom.parseString(xml_string).toprettyxml(indent="  ")
    
    # Write file
    output_path = Path(output_path)
    if not output_path.suffix:
        output_path = output_path.with_suffix('.xml')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(xml_pretty)
    
    logger.info(f"Exported Premiere XML: {output_path}")
    return output_path

