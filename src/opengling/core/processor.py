"""Main video processor that orchestrates the entire pipeline."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional, Callable

from opengling.core.models import (
    EditDecision,
    EditType,
    ExportFormat,
    ProcessingConfig,
    ProcessingResult,
    TranscriptSegment,
)
from opengling.core.transcription import TranscriptionEngine, get_full_transcript
from opengling.core.silence import SilenceDetector, merge_overlapping_regions
from opengling.core.filler import FillerDetector
from opengling.core.bad_takes import BadTakesDetector
from opengling.core.noise import NoiseRemover
from opengling.core.autozoom import AutoZoomProcessor

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main processor that orchestrates the entire OpenGling pipeline.
    
    Usage:
        config = ProcessingConfig(
            remove_fillers=True,
            auto_zoom=False,
        )
        processor = VideoProcessor(config)
        result = processor.process("input.mp4", "output.mp4")
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        
        # Initialize sub-processors
        self._transcriber = TranscriptionEngine(self.config)
        self._silence_detector = SilenceDetector(self.config)
        self._filler_detector = FillerDetector(self.config)
        self._bad_takes_detector = BadTakesDetector(self.config)
        self._noise_remover = NoiseRemover(self.config)
        self._zoom_processor = AutoZoomProcessor(self.config)
        
    def process(
        self,
        input_path: Path | str,
        output_path: Optional[Path | str] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ProcessingResult:
        """
        Process a video file through the complete pipeline.
        
        Args:
            input_path: Path to input video/audio file
            output_path: Path for output file (auto-generated if None)
            progress_callback: Optional callback for progress updates (stage, percent)
            
        Returns:
            ProcessingResult with all data and paths
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Processing: {input_path.name}")
        
        # Initialize result
        result = ProcessingResult(input_path=input_path)
        
        # Generate output path if not provided
        if output_path is None:
            suffix = f".{self.config.output_format.value}"
            if self.config.output_format == ExportFormat.MP4:
                suffix = ".mp4"
            output_path = input_path.parent / f"{input_path.stem}_edited{suffix}"
        result.output_path = Path(output_path)
        
        # Create temp directory for intermediate files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Extract audio
            self._report_progress(progress_callback, "Extracting audio", 0.05)
            audio_path = self._extract_audio(input_path, temp_path)
            output_audio_path = self._extract_output_audio(input_path, temp_path)
            result.original_duration = self._get_duration(input_path)
            
            # Step 2: Apply noise removal (if enabled)
            if self.config.remove_noise:
                self._report_progress(progress_callback, "Removing noise", 0.10)
                audio_path = self._noise_remover.remove_noise(
                    audio_path,
                    temp_path / "denoised.wav",
                )
            
            # Step 3: Transcribe
            self._report_progress(progress_callback, "Transcribing", 0.20)
            result.segments = self._transcriber.transcribe(audio_path)
            result.full_transcript = get_full_transcript(result.segments)
            
            # Step 4: Detect silences
            self._report_progress(progress_callback, "Detecting silences", 0.40)
            silence_edits = self._silence_detector.detect_silences(audio_path)
            
            # Step 5: Detect filler words
            self._report_progress(progress_callback, "Detecting fillers", 0.50)
            filler_edits = self._filler_detector.detect_fillers(result.segments)
            
            # Step 6: Detect bad takes
            self._report_progress(progress_callback, "Detecting bad takes", 0.60)
            bad_take_edits = self._bad_takes_detector.detect_bad_takes(result.segments)
            
            # Step 7: Merge all edit decisions
            self._report_progress(progress_callback, "Merging edits", 0.65)
            all_edits = silence_edits + filler_edits + bad_take_edits
            result.edit_decisions = merge_overlapping_regions(all_edits)
            
            # Count statistics
            result.silences_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.SILENCE
            )
            result.fillers_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.FILLER_WORD
            )
            result.bad_takes_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.BAD_TAKE
            )
            
            # Step 8: Generate zoom keyframes (if enabled)
            if self.config.auto_zoom:
                self._report_progress(progress_callback, "Generating zoom keyframes", 0.70)
                result.zoom_keyframes = self._zoom_processor.generate_zoom_keyframes(
                    input_path
                )
            
            # Step 9: Generate YouTube metadata (if enabled)
            if self.config.generate_youtube_metadata:
                self._report_progress(progress_callback, "Generating YouTube metadata", 0.75)
                from opengling.core.youtube import YouTubeGenerator
                youtube_gen = YouTubeGenerator(self.config)
                result.youtube_metadata = youtube_gen.generate_metadata(
                    result.segments,
                    result.original_duration,
                )
            
            # Step 10: Export/Render
            self._report_progress(progress_callback, "Exporting", 0.80)
            
            if self.config.output_format == ExportFormat.MP4:
                # Render final video
                self._render_video(
                    input_path,
                    result.output_path,
                    result.edit_decisions,
                    result.zoom_keyframes,
                    audio_path if self.config.remove_noise else output_audio_path,
                )
            else:
                # Export timeline
                from opengling.export import export_timeline
                export_timeline(
                    result.edit_decisions,
                    result.output_path,
                    self.config.output_format,
                    result.original_duration,
                    str(input_path),
                )
            
            # Calculate edited duration and keep regions
            cut_duration = sum(e.duration for e in result.edit_decisions if not e.keep)
            result.edited_duration = result.original_duration - cut_duration
            
            # Calculate keep regions for caption adjustment
            from opengling.core.render import get_keep_regions
            keep_regions = get_keep_regions(result.edit_decisions, result.original_duration)
            
            # Step 11: Generate captions (if requested)
            if self.config.caption_format:
                self._report_progress(progress_callback, "Generating captions", 0.90)
                caption_path = input_path.parent / f"{input_path.stem}.{self.config.caption_format.value}"
                from opengling.export.captions import export_captions, adjust_captions_for_edits
                
                # Adjust caption timing to match edited video
                adjusted_segments = adjust_captions_for_edits(result.segments, keep_regions)
                
                result.caption_file = export_captions(
                    adjusted_segments,
                    caption_path,
                    self.config.caption_format,
                )
        
        self._report_progress(progress_callback, "Complete", 1.0)
        
        logger.info(
            f"Processing complete: {result.time_saved:.1f}s saved "
            f"({result.time_saved_percentage:.1f}%)"
        )
        
        return result
    
    def _extract_audio(self, video_path: Path, temp_dir: Path) -> Path:
        """Extract audio from video file."""
        try:
            import ffmpeg
        except ImportError:
            raise ImportError("ffmpeg-python is required")
        
        audio_path = temp_dir / "audio.wav"
        
        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec='pcm_s16le',
                    ac=1,  # mono
                    ar=16000,  # 16kHz for Whisper
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e}")
            raise
        
        return audio_path
    
    def _extract_output_audio(self, video_path: Path, temp_dir: Path) -> Path:
        """Extract audio at original quality for rendering."""
        try:
            import ffmpeg
        except ImportError:
            raise ImportError("ffmpeg-python is required")

        audio_path = temp_dir / "output_audio.wav"

        try:
            (
                ffmpeg
                .input(str(video_path))
                .output(
                    str(audio_path),
                    acodec='pcm_s16le',
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error extracting output audio: {e}")
            raise

        return audio_path

    def _get_duration(self, video_path: Path) -> float:
        """Get video duration in seconds."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(str(video_path))
            return float(probe['format']['duration'])
        except Exception:
            return 0.0
    
    def _render_video(
        self,
        input_path: Path,
        output_path: Path,
        edit_decisions: list[EditDecision],
        zoom_keyframes: list,
        denoised_audio_path: Optional[Path] = None,
    ):
        """Render the final edited video."""
        from opengling.core.render import render_video
        
        render_video(
            input_path=input_path,
            output_path=output_path,
            edit_decisions=edit_decisions,
            zoom_keyframes=zoom_keyframes,
            audio_path=denoised_audio_path,
        )
    
    def _report_progress(
        self,
        callback: Optional[Callable[[str, float], None]],
        stage: str,
        percent: float,
    ):
        """Report progress to callback if provided."""
        if callback:
            callback(stage, percent)
        logger.debug(f"Progress: {stage} ({percent*100:.0f}%)")
    
    def analyze_only(
        self,
        input_path: Path | str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> ProcessingResult:
        """
        Analyze video without rendering output.
        
        Useful for previewing what edits would be made.
        
        Args:
            input_path: Path to input video/audio file
            progress_callback: Optional callback for progress updates (stage, percent)
            
        Returns:
            ProcessingResult with analysis data (no output file)
        """
        input_path = Path(input_path)
        
        result = ProcessingResult(input_path=input_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Extract and analyze
            self._report_progress(progress_callback, "Extracting audio", 0.05)
            audio_path = self._extract_audio(input_path, temp_path)
            result.original_duration = self._get_duration(input_path)
            
            # Transcribe
            self._report_progress(progress_callback, "Transcribing", 0.25)
            result.segments = self._transcriber.transcribe(audio_path)
            result.full_transcript = get_full_transcript(result.segments)
            
            # Detect all issues
            self._report_progress(progress_callback, "Detecting silences", 0.50)
            silence_edits = self._silence_detector.detect_silences(audio_path)
            
            self._report_progress(progress_callback, "Detecting fillers", 0.65)
            filler_edits = self._filler_detector.detect_fillers(result.segments)
            
            self._report_progress(progress_callback, "Detecting bad takes", 0.80)
            bad_take_edits = self._bad_takes_detector.detect_bad_takes(result.segments)
            
            # Merge
            self._report_progress(progress_callback, "Merging edits", 0.90)
            all_edits = silence_edits + filler_edits + bad_take_edits
            result.edit_decisions = merge_overlapping_regions(all_edits)
            
            # Stats
            result.silences_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.SILENCE
            )
            result.fillers_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.FILLER_WORD
            )
            result.bad_takes_removed = sum(
                1 for e in result.edit_decisions if e.edit_type == EditType.BAD_TAKE
            )
            
            cut_duration = sum(e.duration for e in result.edit_decisions if not e.keep)
            result.edited_duration = result.original_duration - cut_duration
        
        self._report_progress(progress_callback, "Complete", 1.0)
        return result
    
    def transcribe_only(
        self,
        input_path: Path | str,
    ) -> list[TranscriptSegment]:
        """
        Only transcribe the video/audio.
        
        Args:
            input_path: Path to input file
            
        Returns:
            List of transcript segments
        """
        input_path = Path(input_path)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            audio_path = self._extract_audio(input_path, Path(temp_dir))
            return self._transcriber.transcribe(audio_path)

