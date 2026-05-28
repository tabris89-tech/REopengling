"""Tests for the web API."""

import pytest


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from fastapi.testclient import TestClient

    from opengling.web.app import app

    return TestClient(app)


class TestWebAPI:
    """Tests for web API endpoints."""

    def test_root_returns_html(self, client):
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "OpenGling" in response.text

    def test_status_not_found(self, client):
        response = client.get("/api/status/nonexistent-job")

        assert response.status_code == 404

    def test_video_not_found(self, client):
        response = client.get("/api/video/nonexistent-job")

        assert response.status_code == 404

    def test_waveform_not_found(self, client):
        response = client.get("/api/waveform/nonexistent-job")

        assert response.status_code == 404

    def test_export_requires_analysis(self, client):
        # Try to export without analyzing first
        response = client.post(
            "/api/export/nonexistent-job",
            json={"job_id": "nonexistent-job", "format": "mp4"}
        )

        assert response.status_code == 404

    def test_edit_requires_analysis(self, client):
        response = client.put(
            "/api/edit/nonexistent-job",
            json={"job_id": "nonexistent-job", "edit_index": 0, "keep": True}
        )

        assert response.status_code == 404

    def test_undo_not_found(self, client):
        response = client.post("/api/undo/nonexistent-job")

        assert response.status_code == 404


class TestUploadFlow:
    """Tests for the upload and processing flow."""

    def test_upload_file(self, client, temp_dir):
        # Create a simple test file
        test_file = temp_dir / "test.mp4"
        test_file.write_bytes(b"fake video content")

        with open(test_file, "rb") as f:
            response = client.post(
                "/api/upload",
                files={"file": ("test.mp4", f, "video/mp4")}
            )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["filename"] == "test.mp4"

    def test_process_requires_upload(self, client):
        response = client.post(
            "/api/process/nonexistent-job",
            json={
                "remove_silences": True,
                "remove_fillers": True,
                "remove_bad_takes": True,
            }
        )

        assert response.status_code == 404

