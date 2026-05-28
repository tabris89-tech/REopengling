"""Download videos from public URLs using yt-dlp and direct HTTP."""

from __future__ import annotations

import json
import logging
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Maximum file size for downloads (4 GB)
MAX_DOWNLOAD_SIZE = 4 * 1024 * 1024 * 1024
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB

# Video/audio extensions for direct link detection
MEDIA_EXTENSIONS = frozenset({
    '.mp4', '.webm', '.mkv', '.avi', '.mov', '.m4v',
    '.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.opus', '.wma',
})

# Progress line regex: [download]  45.2% of ~150.00MiB at  2.50MiB/s ETA 00:33
# ETA absent on early lines: [download]   0.0% of ~150.00MiB
PROGRESS_RE = re.compile(
    r'\[download\]\s+(\d+\.?\d*)%(?:.*?ETA\s+(\S+))?'
)
DEST_RE = re.compile(r'\[download\]\s+Destination:\s+(.+)')
MERGER_RE = re.compile(r'\[Merger\]\s+Merging formats into\s+"(.+)"')
EXTRACT_RE = re.compile(r'\[ExtractAudio\]\s+Destination:\s+(.+)')


@dataclass
class VideoInfo:
    """Information about a single video."""
    url: str
    title: str
    duration: float
    ext: str = "mp4"
    filesize: Optional[float] = None
    resolution: Optional[str] = None
    webpage_url: Optional[str] = None


@dataclass
class InspectResult:
    """Result of URL inspection."""
    type: str  # "single", "playlist", "direct", "unsupported", "error"
    videos: list[VideoInfo] = field(default_factory=list)
    playlist_title: str = ""
    error: str = ""  # human-readable error message
    content_type: str = ""  # for direct links
    content_length: int = 0  # for direct links


# ── Helpers ──────────────────────────────────────────────────────────

def _format_seconds(seconds: float) -> str:
    """Convert float seconds to HH:MM:SS for yt-dlp."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _extract_filename(url: str, content_disposition: str, content_type: str) -> str:
    """Extract filename from URL or Content-Disposition."""
    if content_disposition:
        match = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^"\';\n]+)', content_disposition)
        if match:
            return urllib.parse.unquote(match.group(1).strip())
    path = urllib.parse.urlparse(url).path
    name = Path(path).name
    if name and '.' in name:
        return name
    ext_map = {
        'video/mp4': '.mp4', 'video/webm': '.webm', 'video/x-matroska': '.mkv',
        'video/quicktime': '.mov', 'video/x-msvideo': '.avi',
        'audio/mpeg': '.mp3', 'audio/mp4': '.m4a', 'audio/wav': '.wav',
        'audio/flac': '.flac', 'audio/ogg': '.ogg', 'audio/aac': '.aac',
    }
    ext = ext_map.get(content_type.split(';')[0].strip(), '.mp4')
    return f"download{ext}"


def _clean_error(stderr: str) -> str:
    """Clean up yt-dlp error message for user display."""
    lines = [line for line in stderr.split("\n") if line.strip() and "WARNING" not in line.upper()]
    return lines[-1].strip() if lines else "Unknown error"


# ── yt-dlp helpers ───────────────────────────────────────────────────

def find_ytdlp() -> str:
    """Find yt-dlp binary path."""
    ytdlp = shutil.which("yt-dlp")
    if ytdlp:
        return ytdlp
    try:
        import yt_dlp
        ytdlp_path = os.path.join(os.path.dirname(yt_dlp.__file__), "__main__.py")
        if os.path.exists(ytdlp_path):
            return f"{sys.executable} {ytdlp_path}"
    except ImportError:
        pass
    raise RuntimeError("yt-dlp не найден. Установите: pip install yt-dlp")


def _build_cmd(args: list[str]) -> list[str]:
    """Build yt-dlp command list."""
    ytdlp = find_ytdlp()
    if ytdlp.startswith(sys.executable):
        return ytdlp.split() + args
    return [ytdlp] + args


def _run_ytdlp(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run yt-dlp with given args and return result."""
    cmd = _build_cmd(args)
    logger.debug(f"Running: {' '.join(cmd)}")
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp не найден. Установите: pip install yt-dlp")


# ── Direct link detection ────────────────────────────────────────────

def _is_direct_media_url(url: str) -> bool:
    """Check if URL looks like a direct link to a media file."""
    path = urllib.parse.urlparse(url).path
    ext = Path(path).suffix.lower()
    if ext in MEDIA_EXTENSIONS:
        return True
    return False


# ── Inspection ───────────────────────────────────────────────────────

def inspect_url(url: str) -> InspectResult:
    """
    Inspect a URL and return information about its content.

    Returns InspectResult with type:
      - "single": one video found (via yt-dlp)
      - "playlist": multiple videos found (via yt-dlp)
      - "direct": direct media file link
      - "unsupported": URL/site not supported
      - "error": unexpected error
    """
    url = url.strip()

    # Step 1: Try with yt-dlp (single video)
    result = _run_ytdlp([
        "--no-playlist", "--dump-json", "--no-warnings", url,
    ])

    if result.returncode == 0 and result.stdout.strip():
        try:
            data = json.loads(result.stdout)
            return InspectResult(
                type="single",
                videos=[VideoInfo(
                    url=data.get("webpage_url", url),
                    title=data.get("title", "Unknown"),
                    duration=float(data.get("duration", 0)),
                    ext=data.get("ext", "mp4"),
                    filesize=data.get("filesize") or data.get("filesize_approx"),
                    resolution=(
                        f"{data.get('width', '?')}x{data.get('height', '?')}"
                        if data.get("width") else None
                    ),
                    webpage_url=data.get("webpage_url"),
                )],
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse single video JSON: {e}")

    # Step 2: Try as playlist
    result = _run_ytdlp([
        "--flat-playlist", "--dump-json", "--no-warnings", url,
    ])

    if result.returncode == 0 and result.stdout.strip():
        entries = []
        playlist_title = "Playlist"
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if not entries:
                    playlist_title = data.get("playlist_title", data.get("playlist", "Playlist"))
                entry_url = data.get("url") or data.get("webpage_url", url)
                entries.append(VideoInfo(
                    url=entry_url,
                    title=data.get("title", "Unknown"),
                    duration=float(data.get("duration", 0)),
                    ext=data.get("ext", "mp4"),
                    filesize=data.get("filesize") or data.get("filesize_approx"),
                    webpage_url=data.get("webpage_url"),
                ))
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to parse playlist entry JSON: {e}")

        if len(entries) > 1:
            return InspectResult(type="playlist", videos=entries, playlist_title=playlist_title)
        elif len(entries) == 1:
            return InspectResult(type="single", videos=entries)

    # Step 3: Check if it's a direct media link
    if _is_direct_media_url(url):
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=15) as resp:
                content_type = resp.headers.get('Content-Type', '')
                content_length = resp.headers.get('Content-Length', '0')
                content_disposition = resp.headers.get('Content-Disposition', '')

            filename = _extract_filename(url, content_disposition, content_type)
            filesize = int(content_length) if content_length.isdigit() else None

            return InspectResult(
                type="direct",
                content_type=content_type,
                content_length=filesize or 0,
                videos=[VideoInfo(
                    url=url,
                    title=Path(filename).stem,
                    duration=0,
                    ext=Path(filename).suffix.lstrip('.') or 'mp4',
                    filesize=filesize,
                )],
            )
        except Exception as e:
            return InspectResult(
                type="direct",
                error=str(e),
                videos=[VideoInfo(url=url, title="direct_media", duration=0)],
            )

    # Step 4: Unsupported
    error_msg = _clean_error(result.stderr) if result.returncode != 0 else "No video content found at URL"
    return InspectResult(type="unsupported", error=error_msg)


# ── Download: direct HTTP ────────────────────────────────────────────

def download_direct(
    url: str,
    output_dir: Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    max_filesize: int = MAX_DOWNLOAD_SIZE,
) -> Path:
    """
    Download a file from a direct HTTP/HTTPS link (streaming, no yt-dlp).

    Raises:
        RuntimeError: If URL is not a media file or download fails
        FileNotFoundError: If downloaded file not found
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # HEAD request to check content type and size
    req = urllib.request.Request(url, method='HEAD')
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_type = resp.headers.get('Content-Type', '')
            content_length = resp.headers.get('Content-Length')
            content_disposition = resp.headers.get('Content-Disposition', '')
    except Exception as e:
        raise RuntimeError(f"Не удалось получить информацию о файле: {e}")

    if not (content_type.startswith('video/') or content_type.startswith('audio/')):
        raise RuntimeError(
            f"Ссылка не указывает на видео/аудио файл (Content-Type: {content_type})"
        )

    file_size = int(content_length) if content_length and content_length.isdigit() else 0
    if file_size > max_filesize:
        raise RuntimeError(
            f"Файл слишком большой: {file_size / (1024*1024*1024):.1f} GB "
            f"(максимум {max_filesize / (1024*1024*1024):.1f} GB)"
        )

    filename = _extract_filename(url, content_disposition, content_type)
    output_path = output_dir / filename

    # Streaming download
    req = urllib.request.Request(url, method='GET')
    with urllib.request.urlopen(req, timeout=300) as resp:
        total_downloaded = 0
        last_report = 0.0
        with open(output_path, 'wb') as f:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                total_downloaded += len(chunk)

                if total_downloaded > max_filesize:
                    f.close()
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"Файл превысил лимит {max_filesize / (1024*1024*1024):.1f} GB"
                    )

                now = time.time()
                if progress_callback and now - last_report >= 0.2:
                    last_report = now
                    if file_size > 0:
                        percent = min(1.0, total_downloaded / file_size)
                        progress_callback(
                            percent,
                            f"Скачивание... {percent*100:.0f}% "
                            f"({total_downloaded/(1024*1024):.0f}/{file_size/(1024*1024):.0f} MB)"
                        )
                    else:
                        progress_callback(
                            0.5,
                            f"Скачано {total_downloaded/(1024*1024):.0f} MB"
                        )

    if progress_callback:
        progress_callback(1.0, "Готово")

    if not output_path.exists():
        raise FileNotFoundError(f"Файл не найден после скачивания: {output_path}")

    return output_path


# ── Download: yt-dlp ─────────────────────────────────────────────────

def download_video(
    url: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cookies_from_browser: Optional[str] = None,
    format_str: str = "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]",
    max_filesize: int = MAX_DOWNLOAD_SIZE,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Path:
    """
    Download a video from URL using yt-dlp.

    Args:
        url: Video URL to download
        output_dir: Directory to save to (default: temp dir)
        progress_callback: Optional callback (percent 0.0-1.0, status text)
        cookies_from_browser: Browser name for cookies (chrome/firefox/edge)
        format_str: yt-dlp format string
        max_filesize: Maximum file size in bytes
        start_time: Start time in seconds (uses --download-sections)
        end_time: End time in seconds (uses --download-sections)

    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "--no-playlist",
        "-f", format_str,
        "--newline",
        "--max-filesize", str(max_filesize),
        "--remux-video", "mp4",
        "--socket-timeout", "30",
        "--retries", "3",
        "--fragment-retries", "3",
        "-o", str(output_dir / "%(title)s.%(ext)s"),
    ]

    if cookies_from_browser:
        args.extend(["--cookies-from-browser", cookies_from_browser])

    if start_time is not None or end_time is not None:
        start = "00:00:00" if start_time is None else _format_seconds(start_time)
        end = "99:99:99" if end_time is None else _format_seconds(end_time)
        args.extend(["--download-sections", f"*{start}-{end}"])

    args.append(url)

    cmd = _build_cmd(args)
    logger.info(f"Downloading: {url}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp не найден. Установите: pip install yt-dlp")

    downloaded_path: Optional[Path] = None
    stderr_lines: list[str] = []

    # Track timeouts and progress
    line_timeout = 60
    extraction_timeout = 180
    process_start_time = time.time()
    last_line_time = time.time()
    last_stage_update = 0.0
    has_download_progress = False

    # Thread to read stderr lines into a queue
    stderr_q: queue.Queue[Optional[str]] = queue.Queue()

    def _reader():
        try:
            for line in iter(process.stderr.readline, ""):
                stderr_q.put(line)
        finally:
            stderr_q.put(None)  # sentinel
            process.stderr.close()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        while True:
            try:
                line = stderr_q.get(timeout=1)
            except queue.Empty:
                now = time.time()
                # Line-level timeout — yt-dlp stopped producing any output
                if now - last_line_time >= line_timeout:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"Таймаут скачивания: нет данных от yt-dlp в течение {line_timeout}с"
                    )
                # Extraction timeout — yt-dlp is outputting lines but never starts download
                if not has_download_progress and now - process_start_time >= extraction_timeout:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"Таймаут: yt-dlp не начал скачивание в течение {extraction_timeout}с"
                    )
                # Still alive — update stage with elapsed time every 5s
                if progress_callback and now - last_stage_update >= 5:
                    last_stage_update = now
                    elapsed = now - process_start_time
                    progress_callback(0, f"Подключаюсь к источнику... ({elapsed:.0f}с)")
                continue

            if line is None:
                break  # EOF

            last_line_time = time.time()
            line = line.strip()
            logger.debug(f"yt-dlp: {line}")

            match = PROGRESS_RE.search(line)
            if match and progress_callback:
                has_download_progress = True
                percent = float(match.group(1))
                eta = match.group(2)
                if eta:
                    progress_callback(percent / 100.0, f"Скачивание... {eta} осталось")
                else:
                    progress_callback(percent / 100.0, f"Скачивание... {percent:.0f}%")
                last_stage_update = time.time()
            elif progress_callback:
                # Non-progress line — show what yt-dlp is doing (throttled)
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                now = time.time()
                if clean and now - last_stage_update >= 2:
                    last_stage_update = now
                    elapsed = now - process_start_time
                    progress_callback(0, f"{clean} ({elapsed:.0f}с)")

            for regex in [DEST_RE, MERGER_RE, EXTRACT_RE]:
                m = regex.search(line)
                if m:
                    path_str = m.group(1).strip()
                    if path_str:
                        downloaded_path = Path(path_str)

            if "ERROR:" in line:
                logger.error(f"Download error: {line}")
                stderr_lines.append(line)

    except BaseException:
        process.kill()
        raise

    finally:
        reader.join(timeout=5)
        process.wait()

    if process.returncode != 0:
        stderr = "\n".join(stderr_lines) if stderr_lines else (process.stderr.read() if process.stderr else "")
        error_text = _clean_error(stderr)
        if "unsupported" in error_text.lower():
            raise RuntimeError(f"Сайт не поддерживается: {error_text}")
        raise RuntimeError(f"Ошибка скачивания: {error_text}")

    if not downloaded_path or not downloaded_path.exists():
        files = sorted(
            output_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for f in files:
            if f.suffix.lower() in MEDIA_EXTENSIONS:
                downloaded_path = f
                break

    if not downloaded_path or not downloaded_path.exists():
        raise FileNotFoundError(f"Скачанный файл не найден в {output_dir}")

    return downloaded_path


# ── Unified entry point ──────────────────────────────────────────────

def download_url(
    url: str,
    output_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cookies_from_browser: Optional[str] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Path:
    """
    Download from any URL using best available method.

    Auto-detects URL type:
      - yt-dlp supported sites (YouTube, VK, etc.) → download_video
      - Direct media file links → download_direct (HTTP streaming)
      - Google Drive / Yandex Disk → yt-dlp first, direct fallback

    Returns:
        Path to downloaded file
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading from URL: {url}")

    # Quick check for direct media links (no need to call yt-dlp)
    if _is_direct_media_url(url):
        logger.info("Detected direct media link, using HTTP streaming")
        try:
            return download_direct(
                url=url,
                output_dir=output_dir,
                progress_callback=progress_callback,
            )
        except RuntimeError as e:
            # If direct download fails, still try yt-dlp as fallback
            logger.warning(f"Direct download failed, trying yt-dlp: {e}")

    # Try yt-dlp
    return download_video(
        url=url,
        output_dir=output_dir,
        progress_callback=progress_callback,
        cookies_from_browser=cookies_from_browser,
        start_time=start_time,
        end_time=end_time,
    )


# ── Playlist download ─────────────────────────────────────

def download_playlist(
    url: str,
    output_dir: Optional[Path] = None,
    max_items: int = 0,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cookies_from_browser: Optional[str] = None,
) -> list[Path]:
    """Download all videos from a playlist via yt-dlp."""
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    format_str = "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/best[ext=mp4]"

    args = [
        "-f", format_str,
        "--newline",
        "--max-filesize", str(MAX_DOWNLOAD_SIZE),
        "--remux-video", "mp4",
        "--socket-timeout", "30",
        "--retries", "3",
        "--fragment-retries", "3",
        "-o", str(output_dir / "%(playlist_title)s/%(playlist_index)s - %(title)s.%(ext)s"),
    ]

    if max_items > 0:
        args.extend(["--playlist-end", str(max_items)])

    if cookies_from_browser:
        args.extend(["--cookies-from-browser", cookies_from_browser])

    args.append(url)

    cmd = _build_cmd(args)
    logger.info(f"Downloading playlist: {url}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
    )

    line_timeout = 60
    extraction_timeout = 180
    stderr_lines: list[str] = []
    process_start_time = time.time()
    last_line_time = time.time()
    last_stage_update = 0.0
    has_download_progress = False
    stderr_q: queue.Queue[Optional[str]] = queue.Queue()

    def _reader():
        try:
            for line in iter(process.stderr.readline, ""):
                stderr_q.put(line)
        finally:
            stderr_q.put(None)
            process.stderr.close()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    try:
        while True:
            try:
                line = stderr_q.get(timeout=1)
            except queue.Empty:
                now = time.time()
                if now - last_line_time >= line_timeout:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"Таймаут скачивания: нет данных от yt-dlp в течение {line_timeout}с"
                    )
                if not has_download_progress and now - process_start_time >= extraction_timeout:
                    process.kill()
                    process.wait()
                    raise RuntimeError(
                        f"Таймаут: yt-dlp не начал скачивание в течение {extraction_timeout}с"
                    )
                if progress_callback and now - last_stage_update >= 5:
                    last_stage_update = now
                    elapsed = now - process_start_time
                    progress_callback(0, f"Подключаюсь к источнику... ({elapsed:.0f}с)")
                continue

            if line is None:
                break

            last_line_time = time.time()
            line = line.strip()

            match = PROGRESS_RE.search(line)
            if match and progress_callback:
                has_download_progress = True
                percent = float(match.group(1))
                eta = match.group(2)
                if eta:
                    progress_callback(percent / 100.0, f"Скачивание плейлиста... {eta} осталось")
                else:
                    progress_callback(percent / 100.0, f"Скачивание плейлиста... {percent:.0f}%")
                last_stage_update = time.time()
            elif progress_callback:
                clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
                now = time.time()
                if clean and now - last_stage_update >= 2:
                    last_stage_update = now
                    elapsed = now - process_start_time
                    progress_callback(0, f"{clean} ({elapsed:.0f}с)")
            if "ERROR:" in line:
                logger.error(f"Playlist download error: {line}")
                stderr_lines.append(line)

    except BaseException:
        process.kill()
        raise
    finally:
        reader.join(timeout=5)
        process.wait()

    if process.returncode != 0:
        stderr = "\n".join(stderr_lines) if stderr_lines else (process.stderr.read() if process.stderr else "")
        raise RuntimeError(
            f"Ошибка скачивания плейлиста (код {process.returncode}):\n{_clean_error(stderr)}"
        )

    files = sorted(
        [f for f in output_dir.rglob("*") if f.suffix.lower() in MEDIA_EXTENSIONS],
        key=lambda p: p.stat().st_mtime,
    )

    if not files:
        raise FileNotFoundError(f"Скачанные файлы не найдены в {output_dir}")

    return files
