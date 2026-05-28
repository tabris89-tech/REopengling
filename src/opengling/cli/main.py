"""Command-line interface for OpenGling."""

from __future__ import annotations

import logging
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from opengling.core.models import ProcessingConfig, ExportFormat, parse_time_to_seconds

app = typer.Typer(
    name="opengling",
    help="OpenGling - AI-powered video editing for content creators",
    add_completion=False,
)

console = Console()


def load_config_with_file(
    model: str = "large-v3",
    language: Optional[str] = None,
    silence_threshold: float = 0.5,
    no_silence: bool = False,
    no_fillers: bool = False,
    no_bad_takes: bool = False,
    noise_removal: bool = False,
    auto_zoom: bool = False,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> ProcessingConfig:
    """Load config from file and merge with CLI options."""
    from opengling.core.config_file import load_config
    
    # Start with file config (or defaults)
    base_config = load_config()
    
    # Override with CLI options (CLI takes precedence)
    return ProcessingConfig(
        remove_silences=base_config.remove_silences if not no_silence else False,
        silence_threshold=silence_threshold if silence_threshold != 0.5 else base_config.silence_threshold,
        silence_padding=base_config.silence_padding,
        remove_fillers=base_config.remove_fillers if not no_fillers else False,
        filler_words=base_config.filler_words,
        detect_bad_takes=base_config.detect_bad_takes if not no_bad_takes else False,
        remove_noise=noise_removal or base_config.remove_noise,
        noise_reduction_strength=base_config.noise_reduction_strength,
        auto_zoom=auto_zoom or base_config.auto_zoom,
        max_zoom=base_config.max_zoom,
        whisper_model=model if model != "large-v3" else base_config.whisper_model,
        language=language or base_config.language,
        device=base_config.device,
        compute_type=base_config.compute_type,
        start_time=start_time,
        end_time=end_time,
    )


def create_progress_callback(progress: Progress, task_id):
    """Create a progress callback for the processor."""
    def callback(stage: str, percent: float):
        progress.update(task_id, completed=int(percent * 100), description=f"[cyan]{stage}")
    return callback


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Path to input video file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("mp4", "--format", "-f", help="Output format: mp4, fcpxml, premiere_xml, davinci_edl"),
    
    # Feature toggles
    no_silence: bool = typer.Option(False, "--no-silence", help="Don't remove silences"),
    no_fillers: bool = typer.Option(False, "--no-fillers", help="Don't remove filler words"),
    no_bad_takes: bool = typer.Option(False, "--no-bad-takes", help="Don't remove bad takes"),
    noise_removal: bool = typer.Option(False, "--noise", "-n", help="Apply noise removal"),
    auto_zoom: bool = typer.Option(False, "--zoom", "-z", help="Apply auto-zoom based on face detection"),
    
    # Captions
    captions: Optional[str] = typer.Option(None, "--captions", "-c", help="Generate captions: srt or vtt"),
    
    # YouTube
    youtube: bool = typer.Option(False, "--youtube", "-y", help="Generate YouTube metadata"),
    
    # Model settings
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model: tiny, base, small, medium, large-v3"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code (auto-detect if not set)"),
    
    # Time range
    start: Optional[str] = typer.Option(None, "--start", help="Start time (HH:MM:SS or MM:SS)"),
    end: Optional[str] = typer.Option(None, "--end", help="End time (HH:MM:SS or MM:SS)"),
    
    # Thresholds
    silence_threshold: float = typer.Option(0.5, "--silence-threshold", help="Minimum silence duration to remove (seconds)"),
    
    # Verbose
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Process a video file to remove silences, filler words, and bad takes.
    
    Examples:
    
        opengling process video.mp4
        
        opengling process video.mp4 -o edited.mp4 --noise --captions srt
        
        opengling process video.mp4 --format fcpxml --youtube
    """
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
    )
    
    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    # Check dependencies
    from opengling.core.bootstrap import check_all, print_startup_info
    issues = check_all(auto_install=True)
    if any(i.critical for i in issues):
        print_startup_info(issues)
        raise typer.Exit(1)
    
    # Map format string to enum
    format_map = {
        "mp4": ExportFormat.MP4,
        "fcpxml": ExportFormat.FCPXML,
        "premiere_xml": ExportFormat.PREMIERE_XML,
        "davinci_edl": ExportFormat.DAVINCI_EDL,
    }
    
    if format not in format_map:
        console.print(f"[red]Error: Invalid format '{format}'. Use: mp4, fcpxml, premiere_xml, davinci_edl[/red]")
        raise typer.Exit(1)
    
    # Caption format
    caption_format = None
    if captions:
        caption_map = {"srt": ExportFormat.SRT, "vtt": ExportFormat.VTT}
        if captions not in caption_map:
            console.print(f"[red]Error: Invalid caption format '{captions}'. Use: srt, vtt[/red]")
            raise typer.Exit(1)
        caption_format = caption_map[captions]
    
    # Parse time range
    start_time = None
    end_time = None
    if start:
        try:
            start_time = parse_time_to_seconds(start)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    if end:
        try:
            end_time = parse_time_to_seconds(end)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    
    # Build config - load from file and merge with CLI options
    config = load_config_with_file(
        model=model,
        language=language,
        silence_threshold=silence_threshold,
        no_silence=no_silence,
        no_fillers=no_fillers,
        no_bad_takes=no_bad_takes,
        noise_removal=noise_removal,
        auto_zoom=auto_zoom,
        start_time=start_time,
        end_time=end_time,
    )
    # Override with format-specific options
    config.output_format = format_map[format]
    config.caption_format = caption_format
    config.generate_youtube_metadata = youtube
    
    # Show what we're doing
    time_range_str = ""
    if start_time is not None or end_time is not None:
        from opengling.core.models import format_seconds_to_time
        s = format_seconds_to_time(start_time or 0)
        e = format_seconds_to_time(end_time) if end_time else "end"
        time_range_str = f"\nTime range: [yellow]{s} → {e}[/yellow]"
    
    console.print(Panel.fit(
        f"[bold cyan]OpenGling[/bold cyan]\n"
        f"Processing: [yellow]{input_path.name}[/yellow]{time_range_str}",
        title="🎬 Video Editor",
    ))
    
    # Process with progress bar
    from opengling.core.processor import VideoProcessor
    processor = VideoProcessor(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Starting...", total=100)
        
        callback = create_progress_callback(progress, task)
        result = processor.process(input_path, output, progress_callback=callback)
    
    # Display results
    console.print()
    
    # Stats table
    table = Table(title="📊 Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Original Duration", f"{result.original_duration:.1f}s")
    table.add_row("Edited Duration", f"{result.edited_duration:.1f}s")
    table.add_row("Time Saved", f"{result.time_saved:.1f}s ({result.time_saved_percentage:.1f}%)")
    table.add_row("Silences Removed", str(result.silences_removed))
    table.add_row("Fillers Removed", str(result.fillers_removed))
    table.add_row("Bad Takes Removed", str(result.bad_takes_removed))
    
    console.print(table)
    
    # Output files
    console.print()
    console.print(f"[green]✓[/green] Output: [bold]{result.output_path}[/bold]")
    
    if result.caption_file:
        console.print(f"[green]✓[/green] Captions: [bold]{result.caption_file}[/bold]")
    
    if result.youtube_metadata:
        console.print()
        console.print(Panel(
            f"[bold]Title:[/bold] {result.youtube_metadata.title}\n\n"
            f"[bold]Tags:[/bold] {', '.join(result.youtube_metadata.tags[:5])}",
            title="📺 YouTube Metadata",
        ))


@app.command()
def transcribe(
    input_path: Path = typer.Argument(..., help="Path to video/audio file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output text file"),
    format: str = typer.Option("txt", "--format", "-f", help="Output format: txt, json, srt, vtt"),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model size"),
    language: Optional[str] = typer.Option(None, "--language", "-l", help="Language code"),
):
    """
    Transcribe a video or audio file to text.
    
    Examples:
    
        opengling transcribe video.mp4
        
        opengling transcribe podcast.mp3 -o transcript.txt
        
        opengling transcribe video.mp4 --format srt -o subtitles.srt
    """
    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Transcribing:[/cyan] {input_path.name}")
    
    config = ProcessingConfig(whisper_model=model, language=language)
    
    from opengling.core.processor import VideoProcessor
    processor = VideoProcessor(config)
    
    with console.status("[cyan]Transcribing..."):
        segments = processor.transcribe_only(input_path)
    
    full_text = " ".join(seg.text for seg in segments)
    
    if format == "txt":
        if output:
            output.write_text(full_text)
            console.print(f"[green]✓[/green] Saved to: {output}")
        else:
            console.print()
            console.print(full_text)
    
    elif format == "json":
        import json
        data = [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in segments
        ]
        if output:
            output.write_text(json.dumps(data, indent=2))
            console.print(f"[green]✓[/green] Saved to: {output}")
        else:
            console.print(json.dumps(data, indent=2))
    
    elif format in ("srt", "vtt"):
        from opengling.export.captions import export_srt, export_vtt
        
        out_path = output or input_path.with_suffix(f".{format}")
        if format == "srt":
            export_srt(segments, out_path)
        else:
            export_vtt(segments, out_path)
        console.print(f"[green]✓[/green] Saved to: {out_path}")


@app.command()
def analyze(
    input_path: Path = typer.Argument(..., help="Path to video file"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed edit list"),
):
    """
    Analyze a video without making edits.
    
    Shows what would be removed and potential time savings.
    """
    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Analyzing:[/cyan] {input_path.name}")
    
    config = ProcessingConfig()
    
    from opengling.core.processor import VideoProcessor
    processor = VideoProcessor(config)
    
    with console.status("[cyan]Analyzing..."):
        result = processor.analyze_only(input_path)
    
    # Display results
    console.print()
    
    table = Table(title="📊 Analysis Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Duration", f"{result.original_duration:.1f}s")
    table.add_row("Potential Edited Duration", f"{result.edited_duration:.1f}s")
    table.add_row("Potential Time Savings", f"{result.time_saved:.1f}s ({result.time_saved_percentage:.1f}%)")
    table.add_row("", "")
    table.add_row("Silences Detected", str(result.silences_removed))
    table.add_row("Filler Words Detected", str(result.fillers_removed))
    table.add_row("Bad Takes Detected", str(result.bad_takes_removed))
    
    console.print(table)
    
    if detailed and result.edit_decisions:
        console.print()
        edit_table = Table(title="Edit Decisions", show_header=True)
        edit_table.add_column("Time", style="cyan")
        edit_table.add_column("Type", style="yellow")
        edit_table.add_column("Reason", style="white")
        
        for edit in result.edit_decisions[:20]:
            edit_table.add_row(
                f"{edit.start:.1f}s - {edit.end:.1f}s",
                edit.edit_type.value,
                edit.reason[:50],
            )
        
        console.print(edit_table)
        
        if len(result.edit_decisions) > 20:
            console.print(f"[dim]... and {len(result.edit_decisions) - 20} more edits[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """
    Start the OpenGling web server for the review UI.
    """
    # Check dependencies
    from opengling.core.bootstrap import check_all, print_startup_info
    issues = check_all(auto_install=True)
    if any(i.critical for i in issues):
        print_startup_info(issues)
        raise typer.Exit(1)
    
    console.print(f"[cyan]Starting OpenGling Web Server...[/cyan]")
    console.print(f"[green]→[/green] http://{host}:{port}")
    
    from opengling.web import start_server
    start_server(host, port)


@app.command("mcp")
def run_mcp_server():
    """
    Run the MCP (Model Context Protocol) server.
    
    This allows AI assistants like Claude to use OpenGling as a tool.
    """
    console.print("[cyan]Starting OpenGling MCP Server...[/cyan]")
    
    from opengling.mcp_server import main
    main()


@app.command()
def batch(
    pattern: str = typer.Argument(..., help="Glob pattern or directory path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory"),
    parallel: int = typer.Option(1, "--parallel", "-p", help="Number of parallel jobs"),
    format: str = typer.Option("mp4", "--format", "-f", help="Output format"),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model size"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Process multiple videos in batch.
    
    Examples:
    
        opengling batch "*.mp4"
        
        opengling batch ./videos/ -o ./edited/ -p 2
        
        opengling batch "**/*.mov" --format fcpxml
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
    )
    
    # Find all matching files
    pattern_path = Path(pattern)
    
    if pattern_path.is_dir():
        # Directory - find all video files
        video_extensions = {'.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'}
        files = [f for f in pattern_path.iterdir() if f.suffix.lower() in video_extensions]
    else:
        # Glob pattern
        files = [Path(f) for f in glob_module.glob(pattern, recursive=True)]
        files = [f for f in files if f.is_file()]
    
    if not files:
        console.print(f"[red]No files found matching: {pattern}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Found {len(files)} files to process[/cyan]")
    
    # Create output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Map format
    format_map = {
        "mp4": ExportFormat.MP4,
        "fcpxml": ExportFormat.FCPXML,
        "premiere_xml": ExportFormat.PREMIERE_XML,
        "davinci_edl": ExportFormat.DAVINCI_EDL,
    }
    export_format = format_map.get(format, ExportFormat.MP4)
    
    config = ProcessingConfig(
        whisper_model=model,
        output_format=export_format,
    )
    
    from opengling.core.processor import VideoProcessor
    
    results = []
    failed = []
    
    def process_file(input_path: Path) -> tuple:
        try:
            processor = VideoProcessor(config)
            
            if output_dir:
                suffix = f".{export_format.value}" if export_format != ExportFormat.MP4 else ".mp4"
                out_path = output_dir / f"{input_path.stem}_edited{suffix}"
            else:
                out_path = None
            
            result = processor.process(input_path, out_path)
            return (input_path, result, None)
        except Exception as e:
            return (input_path, None, str(e))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing...", total=len(files))
        
        if parallel > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = {executor.submit(process_file, f): f for f in files}
                
                for future in as_completed(futures):
                    input_path, result, error = future.result()
                    if error:
                        failed.append((input_path, error))
                    else:
                        results.append((input_path, result))
                    progress.update(task, advance=1)
        else:
            # Sequential processing
            for f in files:
                input_path, result, error = process_file(f)
                if error:
                    failed.append((input_path, error))
                else:
                    results.append((input_path, result))
                progress.update(task, advance=1)
    
    # Summary
    console.print()
    console.print(Panel.fit(
        f"[green]Completed: {len(results)}[/green]\n"
        f"[red]Failed: {len(failed)}[/red]",
        title="Batch Results"
    ))
    
    if results:
        total_saved = sum(r.time_saved for _, r in results)
        console.print(f"[cyan]Total time saved: {total_saved:.1f}s[/cyan]")
    
    if failed:
        console.print("\n[red]Failed files:[/red]")
        for path, error in failed:
            console.print(f"  - {path.name}: {error}")


@app.command("save")
def save_project(
    input_path: Path = typer.Argument(..., help="Path to video file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Project file output path"),
    model: str = typer.Option("large-v3", "--model", "-m", help="Whisper model size"),
):
    """
    Analyze a video and save the project for later editing.
    
    Examples:
    
        opengling save video.mp4
        
        opengling save video.mp4 -o myproject.opengling
    """
    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Analyzing:[/cyan] {input_path.name}")
    
    config = ProcessingConfig(whisper_model=model)
    
    from opengling.core.processor import VideoProcessor
    from opengling.core.project import Project
    
    processor = VideoProcessor(config)
    
    with console.status("[cyan]Analyzing video..."):
        result = processor.analyze_only(input_path)
    
    # Create project
    project = Project.from_result(result, config)
    
    # Save
    if output is None:
        output = input_path.with_suffix('.opengling')
    
    project.save(output)
    
    console.print(f"[green]✓[/green] Project saved to: [bold]{output}[/bold]")
    console.print(f"  - Edits: {len(result.edit_decisions)}")
    console.print(f"  - Potential time savings: {result.time_saved:.1f}s")


@app.command("load")
def load_project(
    project_path: Path = typer.Argument(..., help="Path to project file"),
    format: str = typer.Option("mp4", "--format", "-f", help="Output format"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
):
    """
    Load a project and export the edited video.
    
    Examples:
    
        opengling load myproject.opengling
        
        opengling load myproject.opengling -o edited.mp4
    """
    if not project_path.exists():
        console.print(f"[red]Error: File not found: {project_path}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[cyan]Loading project:[/cyan] {project_path.name}")
    
    from opengling.core.project import Project
    from opengling.core.processor import VideoProcessor
    from opengling.export import export_timeline
    from opengling.core.render import get_keep_regions, get_duration
    
    project = Project.load(project_path)
    
    console.print(f"  - Source: {project.input_filename}")
    console.print(f"  - Edits: {len(project.edit_decisions)}")
    
    # Check if source file exists
    source_path = Path(project.input_path)
    if not source_path.exists():
        console.print(f"[yellow]Warning: Original file not found at {source_path}[/yellow]")
        new_path = typer.prompt("Enter new path to source video")
        source_path = Path(new_path)
        if not source_path.exists():
            console.print(f"[red]Error: File not found: {source_path}[/red]")
            raise typer.Exit(1)
    
    # Map format
    format_map = {
        "mp4": ExportFormat.MP4,
        "fcpxml": ExportFormat.FCPXML,
        "premiere_xml": ExportFormat.PREMIERE_XML,
        "davinci_edl": ExportFormat.DAVINCI_EDL,
    }
    export_format = format_map.get(format, ExportFormat.MP4)
    
    # Determine output path
    if output is None:
        suffix = f".{export_format.value}" if export_format != ExportFormat.MP4 else ".mp4"
        output = source_path.parent / f"{source_path.stem}_edited{suffix}"
    
    result = project.to_result()
    config = project.to_config()
    config.output_format = export_format
    
    with console.status("[cyan]Exporting..."):
        if export_format == ExportFormat.MP4:
            processor = VideoProcessor(config)
            processor.process(source_path, output)
        else:
            duration = get_duration(source_path)
            export_timeline(
                result.edit_decisions,
                output,
                export_format,
                duration,
                str(source_path),
            )
    
    console.print(f"[green]✓[/green] Exported to: [bold]{output}[/bold]")


@app.command("init-config")
def init_config(
    path: Optional[Path] = typer.Option(None, "--path", "-p", help="Output path"),
):
    """
    Create an example configuration file.
    
    Examples:
    
        opengling init-config
        
        opengling init-config -p myconfig.yaml
    """
    from opengling.core.config_file import generate_example_config
    
    if path is None:
        path = Path.cwd() / ".opengling.yaml"
    
    if path.exists():
        overwrite = typer.confirm(f"{path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit(0)
    
    config_content = generate_example_config()
    path.write_text(config_content)
    
    console.print(f"[green]✓[/green] Created config file: [bold]{path}[/bold]")
    console.print("  Edit this file to customize OpenGling behavior.")


@app.callback()
def main_callback():
    """OpenGling - AI-powered video editing for content creators."""
    pass


if __name__ == "__main__":
    app()

