"""Video rendering with FFmpeg and MoviePy."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from opengling.core.models import EditDecision, ZoomKeyframe

logger = logging.getLogger(__name__)


def render_video(
    input_path: Path,
    output_path: Path,
    edit_decisions: list[EditDecision],
    zoom_keyframes: Optional[list[ZoomKeyframe]] = None,
    audio_path: Optional[Path] = None,
) -> Path:
    """
    Render the final edited video.
    
    Args:
        input_path: Path to input video
        output_path: Path for output video
        edit_decisions: List of edit decisions (cuts)
        zoom_keyframes: Optional zoom keyframes to apply
        audio_path: Optional path to processed audio (e.g., denoised)
        
    Returns:
        Path to rendered video
    """
    logger.info(f"Rendering video to {output_path.name}")
    
    # Calculate keep regions (inverse of cut regions)
    keep_regions = get_keep_regions(edit_decisions, get_duration(input_path))
    
    if not keep_regions:
        logger.warning("No regions to keep, copying original")
        import shutil
        shutil.copy(input_path, output_path)
        return output_path
    
    # Use FFmpeg for efficient rendering
    if zoom_keyframes:
        # Complex render with zoom - use MoviePy
        return _render_with_moviepy(
            input_path, output_path, keep_regions, zoom_keyframes, audio_path
        )
    else:
        # Simple cuts - use FFmpeg concat (faster)
        return _render_with_ffmpeg(
            input_path, output_path, keep_regions, audio_path
        )


def get_keep_regions(
    edit_decisions: list[EditDecision],
    total_duration: float,
) -> list[tuple[float, float]]:
    """
    Convert cut decisions to keep regions.
    
    Args:
        edit_decisions: List of regions to cut
        total_duration: Total video duration
        
    Returns:
        List of (start, end) tuples for regions to keep
    """
    if not edit_decisions:
        return [(0.0, total_duration)]
    
    # Sort cuts by start time
    cuts = sorted(
        [(e.start, e.end) for e in edit_decisions if not e.keep],
        key=lambda x: x[0]
    )
    
    # Merge overlapping cuts
    merged_cuts = []
    for start, end in cuts:
        if merged_cuts and start <= merged_cuts[-1][1]:
            merged_cuts[-1] = (merged_cuts[-1][0], max(merged_cuts[-1][1], end))
        else:
            merged_cuts.append((start, end))
    
    # Invert to get keep regions
    keep_regions = []
    current_pos = 0.0
    
    for cut_start, cut_end in merged_cuts:
        if cut_start > current_pos:
            keep_regions.append((current_pos, cut_start))
        current_pos = cut_end
    
    # Add final region
    if current_pos < total_duration:
        keep_regions.append((current_pos, total_duration))
    
    return keep_regions


def get_duration(video_path: Path) -> float:
    """Get video duration using FFprobe."""
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(video_path))
        return float(probe['format']['duration'])
    except Exception as e:
        logger.warning(f"Could not get duration: {e}")
        return 0.0


def _render_with_ffmpeg(
    input_path: Path,
    output_path: Path,
    keep_regions: list[tuple[float, float]],
    audio_path: Optional[Path] = None,
) -> Path:
    """Render using FFmpeg concat filter (fast, no re-encoding if possible)."""
    import ffmpeg
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create segment files
        segment_files = []
        for i, (start, end) in enumerate(keep_regions):
            segment_path = temp_path / f"segment_{i:04d}.mp4"
            segment_files.append(segment_path)
            
            try:
                # Extract segment
                stream = ffmpeg.input(str(input_path), ss=start, t=end - start)
                
                if audio_path:
                    # Use processed audio
                    audio_stream = ffmpeg.input(
                        str(audio_path), ss=start, t=end - start
                    )
                    stream = ffmpeg.output(
                        stream.video,
                        audio_stream.audio,
                        str(segment_path),
                        c='copy',
                        acodec='aac',
                    )
                else:
                    stream = ffmpeg.output(
                        stream,
                        str(segment_path),
                        c='copy',  # Copy without re-encoding
                    )
                
                stream.overwrite_output().run(quiet=True)
            except ffmpeg.Error as e:
                logger.error(f"Error extracting segment: {e}")
                raise
        
        # Create concat file
        concat_file = temp_path / "concat.txt"
        with open(concat_file, 'w') as f:
            for seg_path in segment_files:
                f.write(f"file '{seg_path}'\n")
        
        # Concatenate segments
        try:
            (
                ffmpeg
                .input(str(concat_file), format='concat', safe=0)
                .output(str(output_path), c='copy')
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"Error concatenating: {e}")
            # Fall back to re-encoding
            (
                ffmpeg
                .input(str(concat_file), format='concat', safe=0)
                .output(
                    str(output_path),
                    vcodec='libx264',
                    acodec='aac',
                    preset='fast',
                )
                .overwrite_output()
                .run(quiet=True)
            )
    
    logger.info(f"Rendered: {output_path}")
    return output_path


def _render_with_moviepy(
    input_path: Path,
    output_path: Path,
    keep_regions: list[tuple[float, float]],
    zoom_keyframes: list[ZoomKeyframe],
    audio_path: Optional[Path] = None,
) -> Path:
    """Render using MoviePy (supports zoom effects)."""
    try:
        from moviepy.editor import (
            VideoFileClip,
            AudioFileClip,
            concatenate_videoclips,
        )
    except ImportError:
        raise ImportError("moviepy is required for zoom effects")
    
    from opengling.core.autozoom import interpolate_keyframe, apply_zoom_to_frame
    
    logger.info("Rendering with MoviePy (zoom effects enabled)")
    
    # Load video
    video = VideoFileClip(str(input_path))
    
    # Extract clips for each keep region
    clips = []
    for start, end in keep_regions:
        clip = video.subclip(start, end)
        
        # Apply zoom if we have keyframes in this region
        if zoom_keyframes:
            relevant_keyframes = [
                kf for kf in zoom_keyframes
                if start <= kf.time <= end
            ]
            
            if relevant_keyframes:
                def apply_zoom(get_frame, t, start=start):
                    frame = get_frame(t)
                    keyframe = interpolate_keyframe(zoom_keyframes, t + start)
                    return apply_zoom_to_frame(frame, keyframe)
                
                clip = clip.fl(apply_zoom)
        
        clips.append(clip)
    
    # Concatenate
    final = concatenate_videoclips(clips, method="compose")
    
    # Replace audio if we have processed audio
    if audio_path:
        audio = AudioFileClip(str(audio_path))
        # Need to cut audio to match video cuts
        audio_clips = []
        for start, end in keep_regions:
            audio_clips.append(audio.subclip(start, end))
        
        from moviepy.editor import concatenate_audioclips
        final_audio = concatenate_audioclips(audio_clips)
        final = final.set_audio(final_audio)
    
    # Write output
    final.write_videofile(
        str(output_path),
        codec='libx264',
        audio_codec='aac',
        preset='fast',
        threads=4,
        logger=None,  # Suppress MoviePy's verbose output
    )
    
    # Cleanup
    video.close()
    final.close()
    
    logger.info(f"Rendered: {output_path}")
    return output_path


def create_preview_video(
    input_path: Path,
    output_path: Path,
    edit_decisions: list[EditDecision],
    max_duration: float = 60.0,
) -> Path:
    """
    Create a short preview video showing edits.
    
    Highlights what will be cut with visual markers.
    """
    try:
        from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
    except ImportError:
        raise ImportError("moviepy is required for preview")
    
    video = VideoFileClip(str(input_path))
    
    # Limit duration
    if video.duration > max_duration:
        video = video.subclip(0, max_duration)
    
    # Create overlay for cut regions
    def add_cut_overlay(get_frame, t):
        frame = get_frame(t)
        
        # Check if this time is in a cut region
        for edit in edit_decisions:
            if edit.start <= t <= edit.end and not edit.keep:
                # Add red tint to indicate cut
                import numpy as np
                frame = frame.copy()
                frame[:, :, 0] = np.minimum(frame[:, :, 0] + 50, 255)  # Red tint
                break
        
        return frame
    
    preview = video.fl(add_cut_overlay)
    
    preview.write_videofile(
        str(output_path),
        codec='libx264',
        preset='ultrafast',
        logger=None,
    )
    
    video.close()
    preview.close()
    
    return output_path

