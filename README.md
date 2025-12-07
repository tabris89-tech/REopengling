# OpenGling

**Open Source Gling Alternative** - AI-powered video editing for content creators.

OpenGling automatically removes silences, filler words, and bad takes from your videos, just like Gling, but completely open source!

## Features

- **Automatic Silence Removal** - Detects and removes silent pauses
- **Filler Word Detection** - Removes "um", "uh", "like", "you know", etc.
- **Bad Takes Detection** - Identifies stutters, restarts, and mistakes
- **AI Transcription** - Full speech-to-text with word-level timestamps using Whisper
- **Text-Based Editing** - Edit your video by editing the transcript
- **Auto-Generated Captions** - Export SRT or VTT subtitles
- **Auto-Zoom/Framing** - Face detection-based automatic zoom effects
- **Noise Removal** - ML-powered background noise reduction
- **YouTube Optimization** - Generate titles, descriptions, and chapters
- **Export to Pro Tools** - FCPXML (Final Cut Pro), Premiere Pro XML, DaVinci Resolve EDL
- **MCP Tool** - Use with AI assistants like Claude via Model Context Protocol

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (required for video processing)
- Ollama (optional, for YouTube metadata generation)

### Install FFmpeg

**Windows:**
```bash
winget install FFmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg
```

### Install OpenGling

```bash
# Clone or download the repository
cd opengling

# Install dependencies
pip install -e .

# Download spaCy model for filler word detection
python -m spacy download en_core_web_sm
```

## Usage

### Command Line

```bash
# Process a video (removes silences, fillers, bad takes)
opengling process video.mp4

# Process with noise removal and auto-zoom
opengling process video.mp4 --noise --zoom

# Process and generate captions
opengling process video.mp4 --captions srt

# Process with YouTube metadata generation
opengling process video.mp4 --youtube

# Export to Final Cut Pro
opengling process video.mp4 --format fcpxml

# Export to Premiere Pro
opengling process video.mp4 --format premiere_xml

# Export to DaVinci Resolve
opengling process video.mp4 --format davinci_edl

# Just transcribe (no editing)
opengling transcribe video.mp4

# Analyze without editing (preview what would be removed)
opengling analyze video.mp4 --detailed
```

### Web UI

Start the web server for a visual review interface:

```bash
opengling serve

# Or specify host and port
opengling serve --host 0.0.0.0 --port 8080
```

Then open http://localhost:8000 in your browser.

The web UI allows you to:
- Upload videos
- Review the transcript
- See what will be cut
- Toggle individual edits (keep/cut)
- Export to various formats

### MCP Tool (for AI Assistants)

OpenGling can be used as an MCP tool with AI assistants like Claude.

Add to your MCP configuration:

```json
{
  "mcpServers": {
    "opengling": {
      "command": "python",
      "args": ["-m", "opengling.mcp_server"]
    }
  }
}
```

Available MCP tools:
- `process_video` - Full video processing pipeline
- `transcribe` - Transcribe audio/video to text
- `analyze_video` - Analyze without editing
- `generate_captions` - Generate SRT/VTT subtitles
- `generate_youtube_metadata` - Create titles, descriptions, chapters
- `export_timeline` - Export to pro editing software

### Python API

```python
from opengling import VideoProcessor, ProcessingConfig

# Configure processing
config = ProcessingConfig(
    remove_fillers=True,
    detect_bad_takes=True,
    remove_noise=True,
    auto_zoom=False,
    whisper_model="base",  # tiny, base, small, medium, large-v3
    silence_threshold=0.5,  # seconds
)

# Process a video
processor = VideoProcessor(config)
result = processor.process("input.mp4", "output.mp4")

# Print stats
print(f"Time saved: {result.time_saved:.1f}s ({result.time_saved_percentage:.1f}%)")
print(f"Silences removed: {result.silences_removed}")
print(f"Fillers removed: {result.fillers_removed}")
print(f"Bad takes removed: {result.bad_takes_removed}")

# Access transcript
print(result.full_transcript)

# Just analyze (no output file)
result = processor.analyze_only("input.mp4")
for edit in result.edit_decisions:
    print(f"[{edit.start:.1f}s - {edit.end:.1f}s] {edit.edit_type}: {edit.reason}")
```

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `silence_threshold` | 0.5 | Minimum silence duration to remove (seconds) |
| `silence_padding` | 0.1 | Padding around speech regions (seconds) |
| `remove_fillers` | True | Remove filler words |
| `detect_bad_takes` | True | Detect and remove bad takes |
| `remove_noise` | False | Apply noise reduction |
| `noise_reduction_strength` | 0.5 | Noise reduction intensity (0-1) |
| `auto_zoom` | False | Enable face-tracking zoom |
| `max_zoom` | 1.5 | Maximum zoom level |
| `whisper_model` | "base" | Whisper model size |
| `language` | None | Language code (auto-detect if None) |

## Whisper Models

| Model | Speed | Accuracy | VRAM Required |
|-------|-------|----------|---------------|
| tiny | Fastest | Lower | ~1 GB |
| base | Fast | Good | ~1 GB |
| small | Medium | Better | ~2 GB |
| medium | Slower | Great | ~5 GB |
| large-v3 | Slowest | Best | ~10 GB |

## Export Formats

### MP4
Direct video export with all edits applied. The edited video is ready to upload.

### FCPXML (Final Cut Pro)
Import into Final Cut Pro for additional editing. All edit points are preserved as clips on the timeline.

### Premiere Pro XML
Import into Adobe Premiere Pro. Maintains all cut points for fine-tuning.

### DaVinci Resolve EDL
Edit Decision List format for DaVinci Resolve. Import via File > Import > EDL.

## Technology Stack

OpenGling uses state-of-the-art open source tools:

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** - GPU-accelerated speech recognition
- **[FFmpeg](https://ffmpeg.org/)** - Video/audio processing
- **[MoviePy](https://github.com/Zulko/moviepy)** - Video editing
- **[WebRTC VAD](https://github.com/wiseman/py-webrtcvad)** - Voice activity detection
- **[spaCy](https://spacy.io/)** - Natural language processing
- **[MediaPipe](https://github.com/google/mediapipe)** - Face detection
- **[noisereduce](https://github.com/timsainb/noisereduce)** - ML noise reduction
- **[Ollama](https://ollama.ai/)** - Local LLM for YouTube metadata
- **[MCP](https://modelcontextprotocol.io/)** - Model Context Protocol

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Inspired by [Gling](https://gling.ai/), the AI video editor for YouTubers.

