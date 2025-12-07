"""MCP Server implementation for OpenGling."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    CallToolResult,
)

from opengling.core.models import ProcessingConfig, ExportFormat
from opengling.core.processor import VideoProcessor

logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("opengling")


def create_server() -> Server:
    """Create and configure the MCP server."""
    return server


# Tool definitions
TOOLS = [
    Tool(
        name="process_video",
        description="""Process a video file to automatically remove silences, filler words, and bad takes.
        
This is the main tool for editing videos. It will:
1. Transcribe the audio using AI (Whisper)
2. Detect and remove silent pauses
3. Detect and remove filler words (um, uh, like, you know, etc.)
4. Detect and remove bad takes (stutters, restarts)
5. Optionally apply noise reduction
6. Optionally apply auto-zoom based on face detection
7. Export the edited video or timeline

Returns the path to the edited video and statistics about what was removed.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the input video file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for output file (optional, auto-generated if not provided)"
                },
                "remove_silences": {
                    "type": "boolean",
                    "description": "Whether to remove silent pauses (default: true)",
                    "default": True
                },
                "remove_fillers": {
                    "type": "boolean",
                    "description": "Whether to remove filler words (default: true)",
                    "default": True
                },
                "remove_bad_takes": {
                    "type": "boolean",
                    "description": "Whether to remove bad takes/stutters (default: true)",
                    "default": True
                },
                "remove_noise": {
                    "type": "boolean",
                    "description": "Whether to apply noise reduction (default: false)",
                    "default": False
                },
                "auto_zoom": {
                    "type": "boolean",
                    "description": "Whether to apply auto-zoom based on face detection (default: false)",
                    "default": False
                },
                "output_format": {
                    "type": "string",
                    "enum": ["mp4", "fcpxml", "premiere_xml", "davinci_edl"],
                    "description": "Output format (default: mp4)",
                    "default": "mp4"
                },
                "whisper_model": {
                    "type": "string",
                    "enum": ["tiny", "base", "small", "medium", "large-v3"],
                    "description": "Whisper model size (default: base)",
                    "default": "base"
                }
            },
            "required": ["input_path"]
        }
    ),
    Tool(
        name="transcribe",
        description="""Transcribe a video or audio file to text with word-level timestamps.
        
Uses OpenAI's Whisper model for accurate speech recognition.
Returns the full transcript and individual segments with timing information.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video or audio file"
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., 'en', 'es', 'fr'). Auto-detected if not provided."
                },
                "whisper_model": {
                    "type": "string",
                    "enum": ["tiny", "base", "small", "medium", "large-v3"],
                    "description": "Whisper model size (default: base)",
                    "default": "base"
                }
            },
            "required": ["input_path"]
        }
    ),
    Tool(
        name="analyze_video",
        description="""Analyze a video without making any edits.
        
This tool will analyze the video and return information about:
- Detected silences and their durations
- Detected filler words and their locations
- Detected bad takes
- Total time that would be saved by editing
- Full transcript

Useful for previewing what edits would be made before processing.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video file to analyze"
                }
            },
            "required": ["input_path"]
        }
    ),
    Tool(
        name="generate_captions",
        description="""Generate captions/subtitles for a video file.
        
Transcribes the video and exports captions in SRT or VTT format.
These can be used for YouTube closed captions or burned into the video.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for output caption file"
                },
                "format": {
                    "type": "string",
                    "enum": ["srt", "vtt"],
                    "description": "Caption format (default: srt)",
                    "default": "srt"
                }
            },
            "required": ["input_path"]
        }
    ),
    Tool(
        name="generate_youtube_metadata",
        description="""Generate YouTube-optimized metadata from a video.
        
Uses AI to analyze the transcript and generate:
- Engaging video title
- SEO-optimized description
- Relevant tags
- Chapter markers with timestamps

Requires Ollama to be running locally with a language model.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video file"
                },
                "context": {
                    "type": "string",
                    "description": "Optional context about the video (channel name, niche, etc.)"
                },
                "ollama_model": {
                    "type": "string",
                    "description": "Ollama model to use (default: llama3.2)",
                    "default": "llama3.2"
                }
            },
            "required": ["input_path"]
        }
    ),
    Tool(
        name="export_timeline",
        description="""Export edit decisions to a professional video editor timeline format.
        
Supports:
- FCPXML for Final Cut Pro
- XML for Adobe Premiere Pro
- EDL for DaVinci Resolve

This allows you to make fine-tuned adjustments in your preferred editor.""",
        inputSchema={
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to the video file"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path for output timeline file"
                },
                "format": {
                    "type": "string",
                    "enum": ["fcpxml", "premiere_xml", "davinci_edl"],
                    "description": "Timeline format"
                }
            },
            "required": ["input_path", "format"]
        }
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "process_video":
            result = await handle_process_video(arguments)
        elif name == "transcribe":
            result = await handle_transcribe(arguments)
        elif name == "analyze_video":
            result = await handle_analyze_video(arguments)
        elif name == "generate_captions":
            result = await handle_generate_captions(arguments)
        elif name == "generate_youtube_metadata":
            result = await handle_generate_youtube_metadata(arguments)
        elif name == "export_timeline":
            result = await handle_export_timeline(arguments)
        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True,
            )
        
        return CallToolResult(
            content=[TextContent(type="text", text=result)],
            isError=False,
        )
    except Exception as e:
        logger.exception(f"Error in tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True,
        )


async def handle_process_video(args: dict[str, Any]) -> str:
    """Handle process_video tool call."""
    input_path = Path(args["input_path"])
    output_path = args.get("output_path")
    
    # Build config from arguments
    format_map = {
        "mp4": ExportFormat.MP4,
        "fcpxml": ExportFormat.FCPXML,
        "premiere_xml": ExportFormat.PREMIERE_XML,
        "davinci_edl": ExportFormat.DAVINCI_EDL,
    }
    
    config = ProcessingConfig(
        remove_silences=args.get("remove_silences", True),
        remove_fillers=args.get("remove_fillers", True),
        detect_bad_takes=args.get("remove_bad_takes", True),
        remove_noise=args.get("remove_noise", False),
        auto_zoom=args.get("auto_zoom", False),
        output_format=format_map.get(args.get("output_format", "mp4"), ExportFormat.MP4),
        whisper_model=args.get("whisper_model", "base"),
    )
    
    # Process video in thread pool (CPU-bound)
    loop = asyncio.get_event_loop()
    processor = VideoProcessor(config)
    
    result = await loop.run_in_executor(
        None,
        lambda: processor.process(input_path, output_path)
    )
    
    # Format response
    response = f"""Video processed successfully!

**Output:** {result.output_path}

**Statistics:**
- Original duration: {result.original_duration:.1f}s
- Edited duration: {result.edited_duration:.1f}s
- Time saved: {result.time_saved:.1f}s ({result.time_saved_percentage:.1f}%)

**Edits made:**
- Silences removed: {result.silences_removed}
- Filler words removed: {result.fillers_removed}
- Bad takes removed: {result.bad_takes_removed}
"""
    
    if result.youtube_metadata:
        response += f"""
**Generated YouTube Metadata:**
- Title: {result.youtube_metadata.title}
- Tags: {', '.join(result.youtube_metadata.tags[:5])}
"""
    
    return response


async def handle_transcribe(args: dict[str, Any]) -> str:
    """Handle transcribe tool call."""
    input_path = Path(args["input_path"])
    
    config = ProcessingConfig(
        whisper_model=args.get("whisper_model", "base"),
        language=args.get("language"),
    )
    
    processor = VideoProcessor(config)
    
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None,
        lambda: processor.transcribe_only(input_path)
    )
    
    # Format transcript
    full_text = " ".join(seg.text for seg in segments)
    
    response = f"""**Transcript:**

{full_text}

**Segments ({len(segments)}):**
"""
    
    for i, seg in enumerate(segments[:10]):  # Show first 10
        response += f"\n[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}"
    
    if len(segments) > 10:
        response += f"\n\n... and {len(segments) - 10} more segments"
    
    return response


async def handle_analyze_video(args: dict[str, Any]) -> str:
    """Handle analyze_video tool call."""
    input_path = Path(args["input_path"])
    
    config = ProcessingConfig()
    processor = VideoProcessor(config)
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: processor.analyze_only(input_path)
    )
    
    response = f"""**Video Analysis:**

**Duration:** {result.original_duration:.1f}s
**Estimated edited duration:** {result.edited_duration:.1f}s
**Potential time savings:** {result.time_saved:.1f}s ({result.time_saved_percentage:.1f}%)

**Detected Issues:**
- Silences: {result.silences_removed} (would be removed)
- Filler words: {result.fillers_removed} (would be removed)
- Bad takes: {result.bad_takes_removed} (would be removed)

**Edit Decisions ({len(result.edit_decisions)}):**
"""
    
    for i, edit in enumerate(result.edit_decisions[:10]):
        response += f"\n- [{edit.start:.1f}s - {edit.end:.1f}s] {edit.edit_type.value}: {edit.reason}"
    
    if len(result.edit_decisions) > 10:
        response += f"\n\n... and {len(result.edit_decisions) - 10} more edits"
    
    return response


async def handle_generate_captions(args: dict[str, Any]) -> str:
    """Handle generate_captions tool call."""
    input_path = Path(args["input_path"])
    output_path = args.get("output_path")
    format_str = args.get("format", "srt")
    
    format_map = {"srt": ExportFormat.SRT, "vtt": ExportFormat.VTT}
    caption_format = format_map.get(format_str, ExportFormat.SRT)
    
    if not output_path:
        output_path = input_path.parent / f"{input_path.stem}.{format_str}"
    
    config = ProcessingConfig()
    processor = VideoProcessor(config)
    
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None,
        lambda: processor.transcribe_only(input_path)
    )
    
    from opengling.export.captions import export_captions
    caption_path = export_captions(segments, Path(output_path), caption_format)
    
    return f"Captions generated successfully!\n\n**Output:** {caption_path}\n**Format:** {format_str.upper()}\n**Segments:** {len(segments)}"


async def handle_generate_youtube_metadata(args: dict[str, Any]) -> str:
    """Handle generate_youtube_metadata tool call."""
    input_path = Path(args["input_path"])
    context = args.get("context")
    
    config = ProcessingConfig(
        generate_youtube_metadata=True,
        ollama_model=args.get("ollama_model", "llama3.2"),
    )
    
    processor = VideoProcessor(config)
    
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None,
        lambda: processor.transcribe_only(input_path)
    )
    
    from opengling.core.youtube import YouTubeGenerator, format_chapters_for_youtube
    
    generator = YouTubeGenerator(config)
    
    # Get video duration
    from opengling.core.render import get_duration
    duration = get_duration(input_path)
    
    metadata = await loop.run_in_executor(
        None,
        lambda: generator.generate_metadata(segments, duration, context)
    )
    
    chapters_text = format_chapters_for_youtube(metadata.chapters)
    
    return f"""**Generated YouTube Metadata:**

**Title:**
{metadata.title}

**Description:**
{metadata.description}

**Tags:**
{', '.join(metadata.tags)}

**Chapters:**
{chapters_text}
"""


async def handle_export_timeline(args: dict[str, Any]) -> str:
    """Handle export_timeline tool call."""
    input_path = Path(args["input_path"])
    output_path = args.get("output_path")
    format_str = args["format"]
    
    format_map = {
        "fcpxml": ExportFormat.FCPXML,
        "premiere_xml": ExportFormat.PREMIERE_XML,
        "davinci_edl": ExportFormat.DAVINCI_EDL,
    }
    export_format = format_map[format_str]
    
    if not output_path:
        ext = {"fcpxml": ".fcpxml", "premiere_xml": ".xml", "davinci_edl": ".edl"}
        output_path = input_path.parent / f"{input_path.stem}_timeline{ext[format_str]}"
    
    config = ProcessingConfig(output_format=export_format)
    processor = VideoProcessor(config)
    
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: processor.analyze_only(input_path)
    )
    
    from opengling.export import export_timeline
    from opengling.core.render import get_duration
    
    duration = get_duration(input_path)
    
    timeline_path = export_timeline(
        result.edit_decisions,
        Path(output_path),
        export_format,
        duration,
        str(input_path),
    )
    
    format_names = {
        "fcpxml": "Final Cut Pro XML",
        "premiere_xml": "Adobe Premiere Pro XML",
        "davinci_edl": "DaVinci Resolve EDL",
    }
    
    return f"""Timeline exported successfully!

**Output:** {timeline_path}
**Format:** {format_names[format_str]}
**Edit points:** {len(result.edit_decisions)}

You can now import this file into {format_names[format_str].split()[0]} {format_names[format_str].split()[1]} for fine-tuning.
"""


def main():
    """Run the MCP server."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting OpenGling MCP Server...")
    
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(read_stream, write_stream, server.create_initialization_options())
    
    asyncio.run(run())


if __name__ == "__main__":
    main()

