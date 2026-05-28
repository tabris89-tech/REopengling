"""Dependency checker and auto-installer for OpenGling."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from typing import Optional


class DependencyIssue:
    """Represents a missing or broken dependency."""

    def __init__(self, name: str, critical: bool, install_cmd: Optional[str] = None, message: str = ""):
        self.name = name
        self.critical = critical
        self.install_cmd = install_cmd
        self.message = message

    def __repr__(self):
        return f"DependencyIssue({self.name}, critical={self.critical})"


def _run_cmd(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0,
        )
        return result.returncode, result.stdout, result.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -2, "", "Command timed out"
    except Exception as e:
        return -3, "", str(e)


def check_ffmpeg() -> Optional[DependencyIssue]:
    """Check if FFmpeg is installed and accessible."""
    if shutil.which("ffmpeg") is not None:
        return None

    system = platform.system()
    if system == "Windows":
        install_cmd = "winget install FFmpeg"
    elif system == "Darwin":
        install_cmd = "brew install ffmpeg"
    else:
        install_cmd = "sudo apt install ffmpeg"

    return DependencyIssue(
        name="FFmpeg",
        critical=True,
        install_cmd=install_cmd,
        message=(
            "FFmpeg не найден. Без него обработка видео невозможна.\n"
            f"  Установите: {install_cmd}"
        ),
    )


def install_ffmpeg() -> bool:
    """Try to install FFmpeg automatically."""
    system = platform.system()

    if system == "Windows":
        console_print("Attempting to install FFmpeg via winget...")
        rc, stdout, stderr = _run_cmd(["winget", "install", "FFmpeg"], timeout=120)
        if rc == 0:
            console_print("FFmpeg installed successfully.")
            # Refresh PATH
            _refresh_path()
            return shutil.which("ffmpeg") is not None
        else:
            console_print(f"Auto-install failed: {stderr}")
            return False

    elif system == "Darwin":
        console_print("Attempting to install FFmpeg via brew...")
        rc, _, stderr = _run_cmd(["brew", "install", "ffmpeg"], timeout=300)
        if rc == 0:
            console_print("FFmpeg installed successfully.")
            return True
        else:
            console_print(f"Auto-install failed: {stderr}")
            return False

    elif system == "Linux":
        console_print("Attempting to install FFmpeg via apt...")
        rc, _, stderr = _run_cmd(["sudo", "apt", "install", "-y", "ffmpeg"], timeout=300)
        if rc == 0:
            console_print("FFmpeg installed successfully.")
            return True
        else:
            console_print(f"Auto-install failed: {stderr}")
            return False

    return False


def check_cuda() -> bool:
    """Check if NVIDIA CUDA GPU is available."""
    rc, stdout, _ = _run_cmd(["nvidia-smi"], timeout=10)
    return rc == 0 and "NVIDIA" in stdout


def check_spacy_model(model_name: str = "en_core_web_sm") -> Optional[DependencyIssue]:
    """Check if spaCy model is installed."""
    try:
        import spacy
        spacy.load(model_name)
        return None
    except OSError:
        return DependencyIssue(
            name=f"spaCy model: {model_name}",
            critical=False,
            install_cmd=f"python -m spacy download {model_name}",
            message=(
                f"spaCy модель '{model_name}' не найдена.\n"
                f"  Слова-паразиты не будут работать без неё.\n"
                f"  Установите: python -m spacy download {model_name}"
            ),
        )


def install_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """Try to install spaCy model automatically."""
    console_print(f"Installing spaCy model '{model_name}'...")
    rc, stdout, stderr = _run_cmd(
        [sys.executable, "-m", "spacy", "download", model_name],
        timeout=120,
    )
    if rc == 0:
        console_print(f"spaCy model '{model_name}' installed successfully.")
        return True
    else:
        console_print(f"Failed to install spaCy model: {stderr}")
        return False


def _refresh_path():
    """Refresh PATH environment variable on Windows."""
    if platform.system() == "Windows":
        import winreg
        try:
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Environment",
                0,
                winreg.KEY_READ,
            )
            path_value, _ = winreg.QueryValueEx(key, "Path")
            winreg.CloseKey(key)
            os.environ["PATH"] = path_value + ";" + os.environ.get("PATH", "")
        except Exception:
            pass


def console_print(msg: str):
    """Print to console (works with Rich if available)."""
    try:
        from rich.console import Console
        Console().print(f"[cyan]{msg}[/cyan]")
    except ImportError:
        print(msg)


def check_all(auto_install: bool = True) -> list[DependencyIssue]:
    """
    Check all dependencies. Returns list of issues found.

    If auto_install=True, attempts to install missing critical dependencies.
    """
    issues: list[DependencyIssue] = []

    # 1. FFmpeg (critical)
    ffmpeg_issue = check_ffmpeg()
    if ffmpeg_issue:
        if auto_install:
            console_print("FFmpeg not found. Attempting auto-install...")
            if install_ffmpeg():
                console_print("FFmpeg installed. Continuing...")
                ffmpeg_issue = None
            else:
                console_print("FFmpeg auto-install failed. Please install manually.")
        if ffmpeg_issue:
            issues.append(ffmpeg_issue)

    # 2. CUDA (info only)
    if check_cuda():
        console_print("NVIDIA GPU detected. Using GPU acceleration.")
    else:
        console_print("No GPU detected. Using CPU (slower but works).")

    # 3. spaCy model (non-critical)
    spacy_issue = check_spacy_model()
    if spacy_issue:
        if auto_install:
            console_print("spaCy model not found. Attempting auto-install...")
            if install_spacy_model():
                spacy_issue = None
            else:
                console_print("spaCy model auto-install failed.")
        if spacy_issue:
            issues.append(spacy_issue)

    # 4. yt-dlp (non-critical — только для скачивания по URL)
    try:
        from opengling.core.url_downloader import find_ytdlp
        find_ytdlp()
    except (ImportError, RuntimeError):
        issues.append(DependencyIssue(
            name="yt-dlp",
            critical=False,
            install_cmd="pip install yt-dlp",
            message=(
                "yt-dlp не найден. Скачивание видео по URL будет недоступно.\n"
                "  Установите: pip install yt-dlp"
            ),
        ))

    return issues


def print_startup_info(issues: list[DependencyIssue]):
    """Print startup info with issues summary."""
    if not issues:
        console_print("All dependencies OK.")
        return

    critical = [i for i in issues if i.critical]
    warnings = [i for i in issues if not i.critical]

    if critical:
        console_print("\n=== CRITICAL ISSUES ===")
        for issue in critical:
            console_print(f"  {issue.message}")

    if warnings:
        console_print("\n=== WARNINGS ===")
        for issue in warnings:
            console_print(f"  {issue.message}")
