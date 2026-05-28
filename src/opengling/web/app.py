"""FastAPI web application for OpenGling."""

from __future__ import annotations

import asyncio
import atexit
import logging
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from opengling.core.models import ProcessingConfig, ExportFormat, EditDecision, EditType
from opengling.core.processor import VideoProcessor

logger = logging.getLogger(__name__)

# Store for processing jobs with thread safety
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()

# Undo history per job
undo_history: dict[str, list] = {}

# Create FastAPI app
app = FastAPI(
    title="OpenGling",
    description="AI-powered video editing for content creators",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProcessingRequest(BaseModel):
    """Request model for video processing."""
    remove_silences: bool = True
    remove_fillers: bool = True
    remove_bad_takes: bool = True
    remove_noise: bool = False
    auto_zoom: bool = False
    whisper_model: str = "base"
    silence_threshold: float = 0.5


class EditUpdate(BaseModel):
    """Request to update an edit decision."""
    edit_index: int
    keep: bool


class ExportRequest(BaseModel):
    """Request to export processed video."""
    job_id: str
    format: str = "mp4"


# Job cleanup thread
def cleanup_old_jobs():
    """Clean up jobs older than 1 hour."""
    while True:
        time.sleep(300)  # Check every 5 minutes
        cutoff = time.time() - 3600  # 1 hour
        with jobs_lock:
            for job_id in list(jobs.keys()):
                job = jobs[job_id]
                if job.get("created_at", 0) < cutoff:
                    # Clean up temp files
                    temp_dir = job.get("temp_dir")
                    if temp_dir and Path(temp_dir).exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    del jobs[job_id]
                    if job_id in undo_history:
                        del undo_history[job_id]
                    logger.info(f"Cleaned up old job: {job_id}")


# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
cleanup_thread.start()


def cleanup_on_shutdown():
    """Clean up all temp directories on shutdown."""
    with jobs_lock:
        for job_id, job in jobs.items():
            temp_dir = job.get("temp_dir")
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)


atexit.register(cleanup_on_shutdown)


# API Routes

@app.get("/")
async def root():
    """Serve the main UI."""
    return get_index_html()


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
):
    """Upload a video file for processing."""
    job_id = str(uuid.uuid4())
    
    # Check file size before reading into memory
    MAX_UPLOAD_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
    if file.size and file.size > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=413,
            content={"detail": f"File too large. Maximum size is 2 GB, got {file.size / (1024*1024*1024):.1f} GB"}
        )
    
    # Save uploaded file
    temp_dir = Path(tempfile.mkdtemp())
    input_path = temp_dir / file.filename
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "status": "uploaded",
            "progress": 0,
            "stage": "Uploaded",
            "input_path": str(input_path),
            "filename": file.filename,
            "temp_dir": str(temp_dir),
            "created_at": time.time(),
        }
    
    return {"job_id": job_id, "filename": file.filename}


@app.post("/api/process/{job_id}")
async def start_processing(
    job_id: str,
    request: ProcessingRequest,
    background_tasks: BackgroundTasks,
):
    """Start processing a video."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        if job["status"] not in ("uploaded", "analyzed", "complete"):
            raise HTTPException(status_code=400, detail=f"Invalid job status: {job['status']}")
        
        job["status"] = "processing"
        job["progress"] = 0
    
    background_tasks.add_task(process_video_task, job_id, request)
    
    return {"status": "processing", "job_id": job_id}


async def process_video_task(job_id: str, request: ProcessingRequest):
    """Background task to process video."""
    with jobs_lock:
        job = jobs[job_id]
    
    try:
        config = ProcessingConfig(
            remove_silences=request.remove_silences,
            remove_fillers=request.remove_fillers,
            detect_bad_takes=request.remove_bad_takes,
            remove_noise=request.remove_noise,
            auto_zoom=request.auto_zoom,
            whisper_model=request.whisper_model,
            silence_threshold=request.silence_threshold,
        )
        
        processor = VideoProcessor(config)
        
        def progress_callback(stage: str, percent: float):
            with jobs_lock:
                if job_id not in jobs:
                    return
                jobs[job_id]["stage"] = stage
                jobs[job_id]["progress"] = int(percent * 100)
        
        input_path = Path(job["input_path"])
        
        # Run in thread pool with progress callback
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: processor.analyze_only(input_path, progress_callback)
        )
        
        with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "analyzed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["stage"] = "Complete"
            jobs[job_id]["result"] = {
                "original_duration": result.original_duration,
                "edited_duration": result.edited_duration,
                "time_saved": result.time_saved,
                "time_saved_percentage": result.time_saved_percentage,
                "silences_removed": result.silences_removed,
                "fillers_removed": result.fillers_removed,
                "bad_takes_removed": result.bad_takes_removed,
                "transcript": result.full_transcript,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "confidence": seg.confidence,
                    }
                    for seg in result.segments
                ],
                "edit_decisions": [
                    {
                        "start": edit.start,
                        "end": edit.end,
                        "type": edit.edit_type.value,
                        "keep": edit.keep,
                        "reason": edit.reason,
                    }
                    for edit in result.edit_decisions
                ],
            }
        
    except Exception as e:
        logger.exception("Processing error")
        with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        return {
            "id": job["id"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "stage": job.get("stage", ""),
            "filename": job.get("filename", ""),
            "result": job.get("result"),
            "error": job.get("error"),
        }


@app.get("/api/video/{job_id}")
async def stream_video(job_id: str):
    """Stream the video file for preview."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        input_path = Path(job["input_path"])
    
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Determine content type
    suffix = input_path.suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".avi": "video/x-msvideo",
    }
    content_type = content_types.get(suffix, "video/mp4")
    
    return FileResponse(
        path=input_path,
        media_type=content_type,
        filename=input_path.name,
    )


@app.get("/api/waveform/{job_id}")
async def get_waveform(job_id: str, samples: int = 500):
    """Generate waveform data for the video's audio."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        input_path = Path(job["input_path"])
    
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    try:
        # Generate waveform data
        loop = asyncio.get_event_loop()
        waveform_data = await loop.run_in_executor(
            None,
            lambda: generate_waveform(input_path, samples)
        )
        return JSONResponse(content={"waveform": waveform_data})
    except Exception as e:
        logger.exception("Waveform generation error")
        raise HTTPException(status_code=500, detail=str(e))


def generate_waveform(video_path: Path, samples: int = 500) -> list:
    """Generate waveform peaks from video audio."""
    try:
        from pydub import AudioSegment
    except ImportError:
        return [0.5] * samples  # Return flat line if pydub not available
    
    try:
        # Extract audio
        audio = AudioSegment.from_file(str(video_path))
        audio = audio.set_channels(1)  # Mono
        
        # Get raw audio data
        raw_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize
        max_val = np.max(np.abs(raw_data))
        if max_val > 0:
            raw_data = raw_data / max_val
        
        # Downsample to desired number of samples
        chunk_size = len(raw_data) // samples
        if chunk_size < 1:
            chunk_size = 1
        
        peaks = []
        for i in range(samples):
            start = i * chunk_size
            end = start + chunk_size
            if end > len(raw_data):
                end = len(raw_data)
            if start >= len(raw_data):
                peaks.append(0)
            else:
                chunk = raw_data[start:end]
                peak = float(np.max(np.abs(chunk)))
                peaks.append(peak)
        
        return peaks
    except Exception as e:
        logger.warning(f"Waveform generation failed: {e}")
        return [0.5] * samples


@app.put("/api/edit/{job_id}")
async def update_edit(job_id: str, update: EditUpdate):
    """Update an edit decision (toggle keep/cut)."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        if "result" not in job:
            raise HTTPException(status_code=400, detail="Job not analyzed yet")
        
        edits = job["result"]["edit_decisions"]
        
        if update.edit_index < 0 or update.edit_index >= len(edits):
            raise HTTPException(status_code=400, detail="Invalid edit index")
        
        # Save to undo history
        if job_id not in undo_history:
            undo_history[job_id] = []
        undo_history[job_id].append({
            "index": update.edit_index,
            "previous_keep": edits[update.edit_index]["keep"]
        })
        # Keep only last 50 undo states
        undo_history[job_id] = undo_history[job_id][-50:]
        
        # Toggle the keep flag
        edits[update.edit_index]["keep"] = update.keep
        
        # Recalculate stats
        cut_duration = sum(
            e["end"] - e["start"]
            for e in edits
            if not e["keep"]
        )
        original_duration = job["result"]["original_duration"]
        
        job["result"]["edited_duration"] = original_duration - cut_duration
        job["result"]["time_saved"] = cut_duration
        job["result"]["time_saved_percentage"] = (cut_duration / original_duration) * 100 if original_duration > 0 else 0
    
    return {"status": "updated"}


@app.post("/api/undo/{job_id}")
async def undo_edit(job_id: str):
    """Undo the last edit toggle."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job_id not in undo_history or not undo_history[job_id]:
            raise HTTPException(status_code=400, detail="Nothing to undo")
        
        job = jobs[job_id]
        edits = job["result"]["edit_decisions"]
        
        # Pop last undo state
        undo_state = undo_history[job_id].pop()
        edits[undo_state["index"]]["keep"] = undo_state["previous_keep"]
        
        # Recalculate stats
        cut_duration = sum(
            e["end"] - e["start"]
            for e in edits
            if not e["keep"]
        )
        original_duration = job["result"]["original_duration"]
        
        job["result"]["edited_duration"] = original_duration - cut_duration
        job["result"]["time_saved"] = cut_duration
        job["result"]["time_saved_percentage"] = (cut_duration / original_duration) * 100 if original_duration > 0 else 0
    
    return {"status": "undone"}


@app.post("/api/export/{job_id}")
async def export_video(job_id: str, request: ExportRequest, background_tasks: BackgroundTasks):
    """Export the edited video."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        if "result" not in job:
            raise HTTPException(status_code=400, detail="Job not analyzed yet")
        
        if job["status"] == "exporting":
            raise HTTPException(status_code=400, detail="Export already in progress")
        
        job["status"] = "exporting"
    
    background_tasks.add_task(export_video_task, job_id, request.format)
    
    return {"status": "exporting"}


async def export_video_task(job_id: str, format: str):
    """Background task to export video."""
    with jobs_lock:
        job = jobs[job_id]
    
    try:
        format_map = {
            "mp4": ExportFormat.MP4,
            "fcpxml": ExportFormat.FCPXML,
            "premiere_xml": ExportFormat.PREMIERE_XML,
            "davinci_edl": ExportFormat.DAVINCI_EDL,
        }
        
        export_format = format_map.get(format, ExportFormat.MP4)
        
        input_path = Path(job["input_path"])
        
        # Reconstruct edit decisions
        edit_decisions = [
            EditDecision(
                start=e["start"],
                end=e["end"],
                edit_type=EditType(e["type"]),
                keep=e["keep"],
                reason=e["reason"],
            )
            for e in job["result"]["edit_decisions"]
        ]
        
        def progress_callback(stage: str, percent: float):
            with jobs_lock:
                if job_id not in jobs:
                    return
                jobs[job_id]["stage"] = stage
                jobs[job_id]["progress"] = int(percent * 100)
        
        loop = asyncio.get_running_loop()
        
        if export_format == ExportFormat.MP4:
            output_path = input_path.parent / f"{input_path.stem}_edited.mp4"
            
            def render_task():
                import ffmpeg
                import tempfile
                from opengling.core.render import render_video
                
                progress_callback("Extracting audio", 0.0)
                
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_path = Path(tmp)
                    audio_out = tmp_path / "audio.wav"
                    try:
                        (
                            ffmpeg
                            .input(str(input_path))
                            .output(str(audio_out), acodec='pcm_s16le')
                            .overwrite_output()
                            .run(quiet=True)
                        )
                    except ffmpeg.Error as e:
                        logger.warning(f"Audio extraction failed for export: {e}")
                        audio_out = None
                    
                    progress_callback("Rendering video", 0.2)
                    
                    result_path = render_video(
                        input_path=input_path,
                        output_path=output_path,
                        edit_decisions=edit_decisions,
                        audio_path=audio_out,
                    )
                    
                    progress_callback("Complete", 1.0)
                    return result_path
            
            result_path = await loop.run_in_executor(None, render_task)
            with jobs_lock:
                if job_id not in jobs:
                    return
                jobs[job_id]["output_path"] = str(result_path)
        else:
            from opengling.export import export_timeline
            from opengling.core.render import get_duration
            
            duration = get_duration(input_path)
            ext = {
                ExportFormat.FCPXML: ".fcpxml",
                ExportFormat.PREMIERE_XML: ".xml",
                ExportFormat.DAVINCI_EDL: ".edl",
            }
            
            output_path = input_path.parent / f"{input_path.stem}_edited{ext[export_format]}"
            
            await loop.run_in_executor(
                None,
                lambda: export_timeline(
                    edit_decisions,
                    output_path,
                    export_format,
                    duration,
                    str(input_path),
                )
            )
            with jobs_lock:
                if job_id not in jobs:
                    return
                jobs[job_id]["output_path"] = str(output_path)
        
        with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "complete"
        
    except Exception as e:
        logger.exception("Export error")
        with jobs_lock:
            if job_id not in jobs:
                return
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """Download the exported file."""
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job = jobs[job_id]
        
        if "output_path" not in job:
            raise HTTPException(status_code=400, detail="No output file available")
        
        output_path = Path(job["output_path"])
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        filename=output_path.name,
        media_type="application/octet-stream",
    )


def get_index_html() -> HTMLResponse:
    """Return the main HTML page with video.js and waveform."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenGling - AI Video Editor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://vjs.zencdn.net/8.6.1/video-js.css" rel="stylesheet" />
    <script src="https://vjs.zencdn.net/8.6.1/video.min.js"></script>
    <script src="https://unpkg.com/alpinejs@3.x.x/dist/cdn.min.js" defer></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Outfit:wght@400;500;600;700&display=swap');
        
        :root {
            --primary: #8b5cf6;
            --primary-dark: #7c3aed;
            --accent: #06b6d4;
            --bg-dark: #0c0a14;
            --bg-card: #18142a;
            --bg-hover: #241e3a;
            --border: rgba(139, 92, 246, 0.2);
        }
        
        * { box-sizing: border-box; }
        
        body {
            font-family: 'Outfit', sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(ellipse at top, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(ellipse at bottom right, rgba(6, 182, 212, 0.05) 0%, transparent 50%);
            min-height: 100vh;
        }
        
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border: none;
            transition: all 0.2s ease;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.4);
        }
        
        .progress-bar {
            background: linear-gradient(90deg, var(--primary), var(--accent));
        }
        
        /* Video.js custom theme */
        .video-js {
            border-radius: 12px;
            overflow: hidden;
        }
        
        .video-js .vjs-control-bar {
            background: linear-gradient(transparent, rgba(0,0,0,0.8));
            height: 4em;
        }
        
        .video-js .vjs-play-progress,
        .video-js .vjs-volume-level {
            background: var(--primary);
        }
        
        .video-js .vjs-slider {
            background: rgba(255,255,255,0.2);
        }
        
        /* Waveform container */
        .waveform-container {
            position: relative;
            height: 80px;
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            overflow: hidden;
        }
        
        .waveform-canvas {
            width: 100%;
            height: 100%;
        }
        
        .waveform-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .waveform-playhead {
            position: absolute;
            top: 0;
            width: 2px;
            height: 100%;
            background: var(--accent);
            pointer-events: none;
            z-index: 10;
        }
        
        /* Edit regions on waveform */
        .edit-region {
            position: absolute;
            top: 0;
            height: 100%;
            opacity: 0.3;
            pointer-events: none;
        }
        
        .edit-region.cut { background: #ef4444; }
        .edit-region.keep { background: #22c55e; }
        
        /* Edit list items */
        .edit-item {
            transition: all 0.15s ease;
            cursor: pointer;
        }
        
        .edit-item:hover {
            background: var(--bg-hover);
        }
        
        .edit-item.cut {
            border-left: 3px solid #ef4444;
        }
        
        .edit-item.keep {
            border-left: 3px solid #22c55e;
        }
        
        .edit-item.active {
            background: var(--bg-hover);
            box-shadow: inset 0 0 0 1px var(--primary);
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--bg-dark);
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--primary);
            border-radius: 4px;
        }
        
        /* Keyboard shortcut hints */
        .kbd {
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 11px;
            font-family: 'JetBrains Mono', monospace;
        }
        
        /* Stats animation */
        @keyframes countUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .stat-value {
            animation: countUp 0.5s ease-out;
        }
        
        .glow {
            box-shadow: 0 0 40px rgba(139, 92, 246, 0.3);
        }

        /* Toast notifications */
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            pointer-events: none;
        }
        .toast {
            pointer-events: auto;
            padding: 14px 20px;
            border-radius: 12px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            display: flex;
            align-items: center;
            gap: 12px;
            min-width: 280px;
            max-width: 420px;
            animation: toastIn 0.3s ease-out;
            backdrop-filter: blur(12px);
        }
        .toast.leaving {
            animation: toastOut 0.3s ease-in forwards;
        }
        .toast-icon {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 14px;
            font-weight: bold;
        }
        .toast-icon.success { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
        .toast-icon.info { background: rgba(6, 182, 212, 0.2); color: #06b6d4; }
        .toast-icon.error { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .toast-body { flex: 1; font-size: 14px; color: #e2e8f0; }
        .toast-close {
            flex-shrink: 0;
            width: 24px;
            height: 24px;
            border: none;
            background: rgba(255,255,255,0.1);
            border-radius: 6px;
            color: #94a3b8;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: background 0.2s;
        }
        .toast-close:hover { background: rgba(255,255,255,0.2); color: #fff; }
        @keyframes toastIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes toastOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    </style>
</head>
<body class="text-gray-100" x-data="app()">
    <!-- Header -->
    <header class="border-b border-purple-500/20 backdrop-blur-xl sticky top-0 z-50 bg-[#0c0a14]/80">
        <div class="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <div class="flex items-center gap-3">
                <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-purple-500/30">
                    <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path>
                    </svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
                        OpenGling
                    </h1>
                    <p class="text-xs text-gray-500">AI Video Editor</p>
                </div>
            </div>
            <div class="flex items-center gap-4">
                <div x-show="jobId && status === 'analyzed'" class="text-sm text-gray-400">
                    <span class="kbd">Space</span> Play
                    <span class="kbd ml-2">X</span> Toggle
                    <span class="kbd ml-2">Ctrl+Z</span> Undo
                </div>
            </div>
        </div>
    </header>

    <!-- Toast notifications -->
    <div class="toast-container">
        <template x-for="(toast, i) in notifications" :key="toast.id">
            <div class="toast" :class="{ leaving: toast.leaving }">
                <div class="toast-icon" :class="toast.type" x-text="toast.type === 'success' ? '✓' : toast.type === 'error' ? '✗' : 'ℹ'"></div>
                <div class="toast-body" x-text="toast.message"></div>
                <button class="toast-close" @click="dismissToast(i)">✕</button>
            </div>
        </template>
    </div>

    <main class="max-w-7xl mx-auto px-6 py-8" @keydown.window="handleKeydown($event)">
        <!-- Upload Section -->
        <div x-show="!jobId" class="flex flex-col items-center justify-center min-h-[70vh]">
            <div class="card p-12 text-center max-w-xl w-full glow">
                <div class="w-24 h-24 mx-auto mb-8 rounded-full bg-gradient-to-br from-purple-500/20 to-cyan-400/20 flex items-center justify-center">
                    <svg class="w-12 h-12 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
                    </svg>
                </div>
                <h2 class="text-3xl font-bold mb-3">Upload Your Video</h2>
                <p class="text-gray-400 mb-8">AI will remove silences, filler words, and bad takes</p>
                
                <label class="block">
                    <input type="file" accept="video/*,audio/*" @change="uploadFile($event)" class="hidden">
                    <div class="btn-primary text-white font-semibold py-4 px-10 rounded-xl cursor-pointer inline-flex items-center gap-2 text-lg">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path>
                        </svg>
                        Choose Video
                    </div>
                </label>
                
                <p class="mt-8 text-sm text-gray-500">Supports MP4, MOV, MKV, WEBM, MP3, WAV</p>
            </div>
        </div>

        <!-- Processing Section -->
        <div x-show="jobId && status === 'processing'" class="flex flex-col items-center justify-center min-h-[70vh]">
            <div class="card p-12 text-center max-w-xl w-full">
                <div class="w-24 h-24 mx-auto mb-8 rounded-full bg-gradient-to-br from-purple-500/20 to-cyan-400/20 flex items-center justify-center">
                    <svg class="w-12 h-12 text-purple-400 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                </div>
                <h2 class="text-2xl font-bold mb-2" x-text="stage || 'Processing...'"></h2>
                <p class="text-gray-400 mb-8 mono text-sm" x-text="filename"></p>
                
                <div class="w-full bg-gray-800 rounded-full h-3 overflow-hidden">
                    <div class="progress-bar h-full rounded-full transition-all duration-300" :style="`width: ${progress}%`"></div>
                </div>
                <p class="mt-3 text-sm text-gray-400 mono" x-text="`${progress}%`"></p>
            </div>
        </div>

        <!-- Results Section -->
        <div x-show="jobId && (status === 'analyzed' || status === 'complete' || status === 'exporting')" class="space-y-6">
            <!-- Video Player & Waveform -->
            <div class="card p-6">
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <!-- Video Player -->
                    <div class="lg:col-span-2">
                        <video 
                            id="video-player"
                            class="video-js vjs-big-play-centered w-full"
                            controls
                            preload="auto"
                            :data-setup='{"fluid": true}'
                            @timeupdate="onTimeUpdate($event)"
                            @loadedmetadata="onVideoLoaded($event)">
                            <source :src="`/api/video/${jobId}`" type="video/mp4">
                        </video>
                        
                        <!-- Waveform -->
                        <div class="waveform-container mt-4 cursor-pointer" @click="seekToPosition($event)">
                            <canvas id="waveform-canvas" class="waveform-canvas"></canvas>
                            <div class="waveform-overlay" id="waveform-overlay"></div>
                            <div class="waveform-playhead" :style="`left: ${playheadPosition}%`"></div>
                        </div>
                        
                        <div class="flex justify-between mt-2 text-xs text-gray-500 mono">
                            <span x-text="formatTime(currentTime)"></span>
                            <span x-text="formatTime(result?.original_duration)"></span>
                        </div>
                    </div>
                    
                    <!-- Stats -->
                    <div class="space-y-4">
                        <div class="grid grid-cols-2 gap-3">
                            <div class="bg-gray-800/50 rounded-xl p-4">
                                <p class="text-xs text-gray-500 mb-1">Original</p>
                                <p class="text-xl font-bold mono stat-value" x-text="formatTime(result?.original_duration)"></p>
                            </div>
                            <div class="bg-gray-800/50 rounded-xl p-4">
                                <p class="text-xs text-gray-500 mb-1">Edited</p>
                                <p class="text-xl font-bold text-green-400 mono stat-value" x-text="formatTime(result?.edited_duration)"></p>
                            </div>
                        </div>
                        
                        <div class="bg-gradient-to-br from-purple-900/30 to-cyan-900/30 rounded-xl p-4 border border-purple-500/20">
                            <p class="text-xs text-gray-400 mb-1">Time Saved</p>
                            <p class="text-2xl font-bold text-cyan-400 mono" x-text="`${formatTime(result?.time_saved)}`"></p>
                            <p class="text-sm text-gray-500" x-text="`${result?.time_saved_percentage?.toFixed(1)}% reduction`"></p>
                        </div>
                        
                        <div class="grid grid-cols-3 gap-2 text-center">
                            <div class="bg-gray-800/50 rounded-lg p-3">
                                <p class="text-lg font-bold text-purple-400" x-text="result?.silences_removed || 0"></p>
                                <p class="text-xs text-gray-500">Silences</p>
                            </div>
                            <div class="bg-gray-800/50 rounded-lg p-3">
                                <p class="text-lg font-bold text-purple-400" x-text="result?.fillers_removed || 0"></p>
                                <p class="text-xs text-gray-500">Fillers</p>
                            </div>
                            <div class="bg-gray-800/50 rounded-lg p-3">
                                <p class="text-lg font-bold text-purple-400" x-text="result?.bad_takes_removed || 0"></p>
                                <p class="text-xs text-gray-500">Bad Takes</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Transcript & Edits -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <!-- Transcript -->
                <div class="card p-6">
                    <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
                        <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                        </svg>
                        Transcript
                    </h3>
                    <div class="h-[400px] overflow-y-auto pr-2 space-y-2">
                        <template x-for="(segment, i) in result?.segments || []" :key="i">
                            <div 
                                class="p-3 rounded-lg bg-gray-800/50 hover:bg-gray-800 transition-colors cursor-pointer"
                                :class="{ 'ring-1 ring-purple-500': isSegmentActive(segment) }"
                                @click="seekTo(segment.start)">
                                <span class="text-xs text-purple-400 mono" x-text="`${formatTime(segment.start)}`"></span>
                                <p class="text-sm mt-1" x-text="segment.text"></p>
                            </div>
                        </template>
                    </div>
                </div>

                <!-- Edit Decisions -->
                <div class="card p-6">
                    <h3 class="text-lg font-semibold mb-4 flex items-center gap-2">
                        <svg class="w-5 h-5 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 9.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z"></path>
                        </svg>
                        Edit Decisions
                        <span class="text-xs text-gray-500 ml-auto">Click to toggle</span>
                    </h3>
                    <div class="h-[400px] overflow-y-auto pr-2 space-y-2">
                        <template x-for="(edit, i) in result?.edit_decisions || []" :key="i">
                            <div 
                                class="edit-item p-3 rounded-lg"
                                :class="[edit.keep ? 'keep' : 'cut', isEditActive(edit) ? 'active' : '', 'bg-gray-800/50']"
                                @click="toggleEdit(i)"
                                @dblclick="seekTo(edit.start)">
                                <div class="flex items-center justify-between">
                                    <span 
                                        class="text-xs font-bold px-2 py-0.5 rounded" 
                                        :class="edit.keep ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'"
                                        x-text="edit.keep ? 'KEEP' : 'CUT'">
                                    </span>
                                    <span class="text-xs text-gray-500 mono" x-text="edit.type"></span>
                                </div>
                                <p class="text-sm mt-2 mono text-gray-300" x-text="`${formatTime(edit.start)} → ${formatTime(edit.end)}`"></p>
                                <p class="text-xs text-gray-500 mt-1 truncate" x-text="edit.reason"></p>
                            </div>
                        </template>
                    </div>
                </div>
            </div>

            <!-- Export Section -->
            <div class="card p-6">
                <h3 class="text-lg font-semibold mb-4">Export</h3>
                <div class="flex flex-wrap gap-3">
                    <button 
                        @click="exportVideo('mp4')" 
                        class="btn-primary text-white font-medium py-3 px-8 rounded-xl flex items-center gap-2"
                        :disabled="status === 'exporting'">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                        </svg>
                        Export MP4
                    </button>
                    <button @click="exportVideo('fcpxml')" class="bg-gray-700 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-xl transition-colors">
                        Final Cut Pro
                    </button>
                    <button @click="exportVideo('premiere_xml')" class="bg-gray-700 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-xl transition-colors">
                        Premiere Pro
                    </button>
                    <button @click="exportVideo('davinci_edl')" class="bg-gray-700 hover:bg-gray-600 text-white font-medium py-3 px-6 rounded-xl transition-colors">
                        DaVinci Resolve
                    </button>
                </div>
                
                <div x-show="status === 'exporting'" class="mt-4 flex items-center gap-2 text-purple-400">
                    <svg class="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                        <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                        <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"></path>
                    </svg>
                    Exporting video...
                </div>
                
                <div x-show="status === 'complete'" class="mt-4">
                    <a :href="`/api/download/${jobId}`" class="inline-flex items-center gap-2 text-green-400 hover:text-green-300 font-medium">
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                        </svg>
                        Download Ready - Click to download
                    </a>
                </div>
            </div>
        </div>
    </main>

    <script>
        function app() {
            return {
                jobId: null,
                status: null,
                progress: 0,
                stage: '',
                filename: '',
                result: null,
                pollInterval: null,
                currentTime: 0,
                videoDuration: 0,
                playheadPosition: 0,
                waveformData: [],
                player: null,
                notifications: [],
                notificationCounter: 0,

                addNotification(type, message) {
                    const id = ++this.notificationCounter;
                    this.notifications.push({ id, type, message, leaving: false });
                    setTimeout(() => {
                        const idx = this.notifications.findIndex(n => n.id === id);
                        if (idx !== -1) {
                            this.notifications[idx].leaving = true;
                            setTimeout(() => { this.notifications.splice(idx, 1); }, 300);
                        }
                    }, 5000);
                },

                dismissToast(index) {
                    this.notifications[index].leaving = true;
                    setTimeout(() => { this.notifications.splice(index, 1); }, 300);
                },

                formatTime(seconds) {
                    if (!seconds || isNaN(seconds)) return '0:00';
                    const mins = Math.floor(seconds / 60);
                    const secs = Math.floor(seconds % 60);
                    return `${mins}:${secs.toString().padStart(2, '0')}`;
                },

                async uploadFile(event) {
                    const file = event.target.files[0];
                    if (!file) return;

                    this.filename = file.name;

                    const formData = new FormData();
                    formData.append('file', file);

                    try {
                        const response = await fetch('/api/upload', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        this.jobId = data.job_id;
                        this.addNotification('success', 'Загрузка видео завершена');
                        await this.startProcessing();
                    } catch (error) {
                        console.error('Upload error:', error);
                        alert('Upload failed: ' + error.message);
                    }
                },

                async startProcessing() {
                    try {
                        await fetch(`/api/process/${this.jobId}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                remove_silences: true,
                                remove_fillers: true,
                                remove_bad_takes: true,
                                remove_noise: false,
                                auto_zoom: false,
                                whisper_model: 'base'
                            })
                        });
                        
                        this.status = 'processing';
                        this.startPolling();
                    } catch (error) {
                        console.error('Processing error:', error);
                    }
                },

                startPolling() {
                    this.pollInterval = setInterval(async () => {
                        try {
                            const response = await fetch(`/api/status/${this.jobId}`);
                            const data = await response.json();
                            
                            this.status = data.status;
                            this.progress = data.progress;
                            this.stage = data.stage;
                            
                            if (data.result) {
                                this.result = data.result;
                            }

                            if (data.status === 'analyzed') {
                                clearInterval(this.pollInterval);
                                this.addNotification('success', 'Анализ видео завершён');
                                this.$nextTick(() => {
                                    this.initVideoPlayer();
                                    this.loadWaveform();
                                });
                            }

                            if (data.status === 'complete') {
                                clearInterval(this.pollInterval);
                                this.addNotification('success', 'Экспорт видео завершён');
                                this.$nextTick(() => {
                                    // Auto-download
                                    const a = document.createElement('a');
                                    a.href = `/api/download/${this.jobId}`;
                                    a.download = 'video_edited.mp4';
                                    document.body.appendChild(a);
                                    a.click();
                                    document.body.removeChild(a);
                                });
                            }

                            if (data.status === 'error') {
                                clearInterval(this.pollInterval);
                                this.addNotification('error', 'Ошибка: ' + data.error);
                            }
                        } catch (error) {
                            console.error('Polling error:', error);
                        }
                    }, 1000);
                },

                initVideoPlayer() {
                    const videoEl = document.getElementById('video-player');
                    if (videoEl && !this.player) {
                        this.player = videojs(videoEl, {
                            fluid: true,
                            playbackRates: [0.5, 1, 1.5, 2],
                        });
                    }
                },

                async loadWaveform() {
                    try {
                        const response = await fetch(`/api/waveform/${this.jobId}`);
                        const data = await response.json();
                        this.waveformData = data.waveform;
                        this.drawWaveform();
                    } catch (error) {
                        console.error('Waveform error:', error);
                    }
                },

                drawWaveform() {
                    const canvas = document.getElementById('waveform-canvas');
                    if (!canvas || !this.waveformData.length) return;
                    
                    const ctx = canvas.getContext('2d');
                    const dpr = window.devicePixelRatio || 1;
                    
                    canvas.width = canvas.offsetWidth * dpr;
                    canvas.height = canvas.offsetHeight * dpr;
                    ctx.scale(dpr, dpr);
                    
                    const width = canvas.offsetWidth;
                    const height = canvas.offsetHeight;
                    const barWidth = width / this.waveformData.length;
                    const centerY = height / 2;
                    
                    // Draw edit regions first
                    if (this.result?.edit_decisions) {
                        const duration = this.result.original_duration;
                        for (const edit of this.result.edit_decisions) {
                            const startX = (edit.start / duration) * width;
                            const endX = (edit.end / duration) * width;
                            ctx.fillStyle = edit.keep ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)';
                            ctx.fillRect(startX, 0, endX - startX, height);
                        }
                    }
                    
                    // Draw waveform bars
                    const gradient = ctx.createLinearGradient(0, 0, width, 0);
                    gradient.addColorStop(0, '#8b5cf6');
                    gradient.addColorStop(1, '#06b6d4');
                    
                    ctx.fillStyle = gradient;
                    
                    for (let i = 0; i < this.waveformData.length; i++) {
                        const peak = this.waveformData[i];
                        const barHeight = peak * height * 0.8;
                        const x = i * barWidth;
                        
                        ctx.fillRect(x, centerY - barHeight / 2, Math.max(1, barWidth - 1), barHeight);
                    }
                },

                onTimeUpdate(event) {
                    const video = event.target;
                    this.currentTime = video.currentTime;
                    if (this.videoDuration > 0) {
                        this.playheadPosition = (this.currentTime / this.videoDuration) * 100;
                    }
                },

                onVideoLoaded(event) {
                    this.videoDuration = event.target.duration;
                },

                seekTo(time) {
                    const video = document.getElementById('video-player');
                    if (video) {
                        video.currentTime = time;
                    }
                },

                seekToPosition(event) {
                    const rect = event.currentTarget.getBoundingClientRect();
                    const x = event.clientX - rect.left;
                    const percent = x / rect.width;
                    const time = percent * this.videoDuration;
                    this.seekTo(time);
                },

                isSegmentActive(segment) {
                    return this.currentTime >= segment.start && this.currentTime <= segment.end;
                },

                isEditActive(edit) {
                    return this.currentTime >= edit.start && this.currentTime <= edit.end;
                },

                async toggleEdit(index) {
                    const edit = this.result.edit_decisions[index];
                    const newKeep = !edit.keep;

                    try {
                        await fetch(`/api/edit/${this.jobId}`, {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                job_id: this.jobId,
                                edit_index: index,
                                keep: newKeep
                            })
                        });

                        edit.keep = newKeep;
                        
                        // Refresh stats
                        const response = await fetch(`/api/status/${this.jobId}`);
                        const data = await response.json();
                        this.result = data.result;
                        
                        // Redraw waveform
                        this.drawWaveform();
                    } catch (error) {
                        console.error('Toggle error:', error);
                    }
                },

                async undo() {
                    try {
                        await fetch(`/api/undo/${this.jobId}`, { method: 'POST' });
                        
                        const response = await fetch(`/api/status/${this.jobId}`);
                        const data = await response.json();
                        this.result = data.result;
                        this.drawWaveform();
                    } catch (error) {
                        console.error('Undo error:', error);
                    }
                },

                handleKeydown(event) {
                    if (!this.jobId || this.status !== 'analyzed') return;
                    
                    const video = document.getElementById('video-player');
                    
                    switch(event.key) {
                        case ' ':
                            event.preventDefault();
                            if (video.paused) video.play();
                            else video.pause();
                            break;
                        case 'k':
                            if (video.paused) video.play();
                            else video.pause();
                            break;
                        case 'j':
                            video.currentTime = Math.max(0, video.currentTime - 10);
                            break;
                        case 'l':
                            video.currentTime = Math.min(this.videoDuration, video.currentTime + 10);
                            break;
                        case 'ArrowLeft':
                            video.currentTime = Math.max(0, video.currentTime - 0.1);
                            break;
                        case 'ArrowRight':
                            video.currentTime = Math.min(this.videoDuration, video.currentTime + 0.1);
                            break;
                        case 'x':
                        case 'X':
                            // Toggle current edit
                            const activeEdit = this.result?.edit_decisions?.findIndex(e => 
                                this.currentTime >= e.start && this.currentTime <= e.end
                            );
                            if (activeEdit !== undefined && activeEdit >= 0) {
                                this.toggleEdit(activeEdit);
                            }
                            break;
                        case 'z':
                            if (event.ctrlKey || event.metaKey) {
                                event.preventDefault();
                                this.undo();
                            }
                            break;
                    }
                },

                async exportVideo(format) {
                    this.status = 'exporting';

                    try {
                        await fetch(`/api/export/${this.jobId}`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ job_id: this.jobId, format })
                        });

                        this.startPolling();
                    } catch (error) {
                        console.error('Export error:', error);
                        this.status = 'analyzed';
                    }
                }
            };
        }
    </script>
</body>
</html>"""
    return HTMLResponse(content=html)


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    return app


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the web server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


def main():
    """Main entry point for the web server."""
    start_server()


if __name__ == "__main__":
    main()
