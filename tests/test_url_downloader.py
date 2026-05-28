"""Tests for URL downloader module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

SINGLE_VIDEO_JSON = """{
    "title": "Test Video",
    "duration": 300.0,
    "ext": "mp4",
    "filesize": 52428800,
    "width": 1920,
    "height": 1080,
    "webpage_url": "https://youtube.com/watch?v=test123"
}"""

PLAYLIST_JSON = """{"title": "Video 1", "duration": 120.0, "ext": "mp4", "url": "https://youtube.com/watch?v=v1", "playlist_title": "My Playlist", "playlist": "My Playlist"}
{"title": "Video 2", "duration": 240.0, "ext": "webm", "url": "https://youtube.com/watch?v=v2"}
"""


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_creation(self):
        from opengling.core.url_downloader import VideoInfo

        info = VideoInfo(
            url="https://example.com/video",
            title="Test",
            duration=120.5,
            ext="mp4",
            filesize=1048576,
            resolution="1920x1080",
        )
        assert info.url == "https://example.com/video"
        assert info.title == "Test"
        assert info.duration == 120.5
        assert info.ext == "mp4"
        assert info.filesize == 1048576
        assert info.resolution == "1920x1080"


class TestInspectResult:
    """Tests for InspectResult dataclass."""

    def test_creation(self):
        from opengling.core.url_downloader import InspectResult, VideoInfo

        video = VideoInfo(url="https://example.com/v", title="V", duration=60, ext="mp4")
        result = InspectResult(type="single", videos=[video], error="")
        assert result.type == "single"
        assert len(result.videos) == 1
        assert result.videos[0].title == "V"
        assert result.error == ""

    def test_unsupported(self):
        from opengling.core.url_downloader import InspectResult

        result = InspectResult(type="unsupported", error="Unsupported URL")
        assert result.type == "unsupported"
        assert "Unsupported" in result.error


class TestCleanError:
    """Tests for _clean_error function."""

    def test_cleans_warnings(self):
        from opengling.core.url_downloader import _clean_error

        stderr = "WARNING: something\nERROR: Unsupported URL\nWARNING: another\n"
        result = _clean_error(stderr)
        assert result == "ERROR: Unsupported URL"
        assert "WARNING" not in result

    def test_empty_stderr(self):
        from opengling.core.url_downloader import _clean_error

        assert _clean_error("") == "Unknown error"

    def test_only_warnings(self):
        from opengling.core.url_downloader import _clean_error

        result = _clean_error("WARNING: one\nWARNING: two\n")
        assert result == "Unknown error"


class TestInspectUrl:
    """Tests for inspect_url function."""

    def test_single_video(self):
        from opengling.core.url_downloader import inspect_url

        mock_run = MagicMock()
        mock_run.returncode = 0
        mock_run.stdout = SINGLE_VIDEO_JSON
        mock_run.stderr = ""

        with patch("opengling.core.url_downloader.find_ytdlp", return_value="yt-dlp.exe"):
            with patch("opengling.core.url_downloader.subprocess.run", return_value=mock_run):
                result = inspect_url("https://youtube.com/watch?v=test123")

        assert result.type == "single"
        assert len(result.videos) == 1
        assert result.videos[0].title == "Test Video"
        assert result.videos[0].duration == 300.0
        assert result.videos[0].resolution == "1920x1080"
        assert result.videos[0].filesize == 52428800

    def test_playlist(self):
        from opengling.core.url_downloader import inspect_url

        # First call (--no-playlist) should fail or return single
        # Second call (--flat-playlist) returns playlist
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stdout = ""
        mock_fail.stderr = ""

        mock_playlist = MagicMock()
        mock_playlist.returncode = 0
        mock_playlist.stdout = PLAYLIST_JSON
        mock_playlist.stderr = ""

        with patch("opengling.core.url_downloader.find_ytdlp", return_value="yt-dlp.exe"):
            with patch("opengling.core.url_downloader.subprocess.run") as mock_run:
                mock_run.side_effect = [mock_fail, mock_playlist]
                result = inspect_url("https://youtube.com/playlist?list=abc")

        assert result.type == "playlist"
        assert result.playlist_title == "My Playlist"
        assert len(result.videos) == 2
        assert result.videos[0].title == "Video 1"
        assert result.videos[0].duration == 120.0
        assert result.videos[1].title == "Video 2"
        assert result.videos[1].duration == 240.0

    def test_unsupported_url(self):
        from opengling.core.url_downloader import inspect_url

        mock_run = MagicMock()
        mock_run.returncode = 1
        mock_run.stdout = ""
        mock_run.stderr = "ERROR: Unsupported URL"

        with patch("opengling.core.url_downloader.find_ytdlp", return_value="yt-dlp.exe"):
            with patch("opengling.core.url_downloader.subprocess.run", return_value=mock_run):
                result = inspect_url("https://unsupported-site.com/video")

        assert result.type == "unsupported"
        assert result.error != ""

    def test_single_video_fallback_from_playlist(self):
        """Single entry returned from --flat-playlist should be treated as single."""
        from opengling.core.url_downloader import inspect_url

        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stdout = ""
        mock_fail.stderr = ""

        mock_single = MagicMock()
        mock_single.returncode = 0
        mock_single.stdout = '{"title": "Only Video", "duration": 60, "ext": "mp4", "url": "https://example.com/v"}'
        mock_single.stderr = ""

        with patch("opengling.core.url_downloader.find_ytdlp", return_value="yt-dlp.exe"):
            with patch("opengling.core.url_downloader.subprocess.run") as mock_run:
                mock_run.side_effect = [mock_fail, mock_single]
                result = inspect_url("https://example.com/video")

        assert result.type == "single"
        assert len(result.videos) == 1
        assert result.videos[0].title == "Only Video"


class TestIsDirectMediaUrl:
    """Tests for _is_direct_media_url function."""

    def test_direct_mp4(self):
        from opengling.core.url_downloader import _is_direct_media_url
        assert _is_direct_media_url("https://example.com/video.mp4")
        assert _is_direct_media_url("https://example.com/video.mp4?query=1")
        assert _is_direct_media_url("http://cdn.example.com/path/to/file.MP4")

    def test_direct_audio(self):
        from opengling.core.url_downloader import _is_direct_media_url
        assert _is_direct_media_url("https://example.com/audio.mp3")
        assert _is_direct_media_url("https://example.com/song.wav")

    def test_youtube_url(self):
        from opengling.core.url_downloader import _is_direct_media_url
        assert not _is_direct_media_url("https://youtube.com/watch?v=test123")
        assert not _is_direct_media_url("https://youtu.be/test123")

    def test_no_extension(self):
        from opengling.core.url_downloader import _is_direct_media_url
        assert not _is_direct_media_url("https://example.com/video")
        assert not _is_direct_media_url("https://example.com/page.html")


class TestFormatSeconds:
    """Tests for _format_seconds function."""

    def test_zero(self):
        from opengling.core.url_downloader import _format_seconds
        assert _format_seconds(0) == "00:00:00"

    def test_simple(self):
        from opengling.core.url_downloader import _format_seconds
        assert _format_seconds(3661) == "01:01:01"

    def test_float(self):
        from opengling.core.url_downloader import _format_seconds
        assert _format_seconds(90.5) == "00:01:30"


class TestExtractFilename:
    """Tests for _extract_filename function."""

    def test_from_url(self):
        from opengling.core.url_downloader import _extract_filename
        result = _extract_filename(
            "https://example.com/video.mp4", "", "video/mp4"
        )
        assert result == "video.mp4"

    def test_from_content_disposition(self):
        from opengling.core.url_downloader import _extract_filename
        result = _extract_filename(
            "https://example.com/download",
            'attachment; filename="myvideo.mp4"',
            "video/mp4",
        )
        assert result == "myvideo.mp4"

    def test_no_filename_in_url(self):
        from opengling.core.url_downloader import _extract_filename
        result = _extract_filename("https://example.com/download", "", "video/webm")
        assert result == "download.webm"


class TestInspectUrlDirectLink:
    """Tests for inspect_url with direct media links."""

    def test_direct_mp4_link(self):
        """Direct .mp4 URL should return type='direct'."""
        from opengling.core.url_downloader import inspect_url

        # yt-dlp fails → falls through to direct link detection
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stdout = ""
        mock_fail.stderr = "ERROR: Unsupported URL"

        mock_playlist_fail = MagicMock()
        mock_playlist_fail.returncode = 1
        mock_playlist_fail.stdout = ""
        mock_playlist_fail.stderr = ""

        mock_urlopen = MagicMock()
        mock_urlopen.headers = {
            'Content-Type': 'video/mp4',
            'Content-Length': '52428800',
        }
        mock_urlopen.__enter__.return_value = mock_urlopen

        with patch("opengling.core.url_downloader.find_ytdlp", return_value="yt-dlp.exe"):
            with patch("opengling.core.url_downloader.subprocess.run", return_value=mock_fail) as mock_subprocess:
                # Also need to mock the second subprocess call for playlist
                mock_subprocess.side_effect = [mock_fail, mock_playlist_fail]
                with patch("opengling.core.url_downloader.urllib.request.urlopen", return_value=mock_urlopen):
                    result = inspect_url("https://example.com/video.mp4")

        assert result.type == "direct"
        assert len(result.videos) == 1


class TestDownloadUrl:
    """Tests for download_url auto-detection."""

    def test_ytdlp_url(self):
        """URL supported by yt-dlp should use download_video path."""
        from opengling.core.url_downloader import download_url

        mock_inspect = MagicMock()
        mock_inspect.type = "single"
        mock_inspect.videos = [MagicMock(url="https://youtube.com/watch?v=test")]

        mock_download = MagicMock(return_value=Path("/fake/output.mp4"))

        with patch("opengling.core.url_downloader.inspect_url", return_value=mock_inspect):
            with patch("opengling.core.url_downloader.download_video", mock_download):
                result = download_url("https://youtube.com/watch?v=test")

        assert result == Path("/fake/output.mp4")
        mock_download.assert_called_once()

