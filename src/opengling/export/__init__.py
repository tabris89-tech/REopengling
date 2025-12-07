"""Export modules for various timeline formats."""

from pathlib import Path
from typing import Optional

from opengling.core.models import EditDecision, ExportFormat


def export_timeline(
    edit_decisions: list[EditDecision],
    output_path: Path | str,
    format: ExportFormat,
    total_duration: float,
    source_file: str,
    fps: float = 30.0,
) -> Path:
    """
    Export edit decisions to a timeline format.
    
    Args:
        edit_decisions: List of edit decisions
        output_path: Output file path
        format: Export format (FCPXML, Premiere XML, EDL)
        total_duration: Total source duration in seconds
        source_file: Path to source media file
        fps: Frames per second
        
    Returns:
        Path to exported file
    """
    output_path = Path(output_path)
    
    # Calculate keep regions
    from opengling.core.render import get_keep_regions
    keep_regions = get_keep_regions(edit_decisions, total_duration)
    
    if format == ExportFormat.FCPXML:
        from opengling.export.fcpxml import export_fcpxml
        return export_fcpxml(keep_regions, output_path, source_file, total_duration, fps)
    
    elif format == ExportFormat.PREMIERE_XML:
        from opengling.export.premiere import export_premiere_xml
        return export_premiere_xml(keep_regions, output_path, source_file, total_duration, fps)
    
    elif format == ExportFormat.DAVINCI_EDL:
        from opengling.export.davinci import export_edl
        return export_edl(keep_regions, output_path, source_file, fps)
    
    else:
        raise ValueError(f"Unsupported export format: {format}")


__all__ = ["export_timeline"]

