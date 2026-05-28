"""Tests for project save/load functionality."""

from pathlib import Path


class TestProject:
    """Tests for Project class."""

    def test_create_project(self):
        from opengling.core.project import Project

        project = Project()

        assert project.version == "1.0.0"
        assert project.input_path == ""
        assert project.edit_decisions == []

    def test_save_and_load(self, temp_dir):
        from opengling.core.project import Project

        # Create a project with some data
        project = Project(
            input_path="/path/to/video.mp4",
            input_filename="video.mp4",
            original_duration=120.0,
            full_transcript="Hello world",
            edit_decisions=[
                {
                    "start": 1.0,
                    "end": 2.0,
                    "edit_type": "silence",
                    "keep": False,
                    "reason": "Silence detected",
                    "confidence": 0.9,
                }
            ],
        )

        # Save
        save_path = temp_dir / "test.opengling"
        project.save(save_path)

        assert save_path.exists()

        # Load
        loaded = Project.load(save_path)

        assert loaded.input_path == "/path/to/video.mp4"
        assert loaded.original_duration == 120.0
        assert loaded.full_transcript == "Hello world"
        assert len(loaded.edit_decisions) == 1
        assert loaded.edit_decisions[0]["start"] == 1.0

    def test_from_result(self):
        from opengling.core.models import (
            EditDecision,
            EditType,
            ProcessingConfig,
            ProcessingResult,
            TranscriptSegment,
        )
        from opengling.core.project import Project

        result = ProcessingResult(
            input_path=Path("test.mp4"),
            segments=[
                TranscriptSegment(text="Hello", start=0.0, end=1.0),
            ],
            edit_decisions=[
                EditDecision(start=1.0, end=2.0, edit_type=EditType.SILENCE, keep=False, reason="Silence"),
            ],
            full_transcript="Hello",
            original_duration=10.0,
            edited_duration=8.0,
        )

        config = ProcessingConfig()

        project = Project.from_result(result, config)

        assert project.input_filename == "test.mp4"
        assert project.original_duration == 10.0
        assert len(project.segments) == 1
        assert len(project.edit_decisions) == 1

    def test_to_result(self):
        from opengling.core.project import Project

        project = Project(
            input_path="/path/to/video.mp4",
            original_duration=100.0,
            edited_duration=80.0,
            segments=[
                {
                    "text": "Hello",
                    "start": 0.0,
                    "end": 1.0,
                    "words": [],
                    "confidence": 0.9,
                    "language": "en",
                }
            ],
            edit_decisions=[
                {
                    "start": 1.0,
                    "end": 2.0,
                    "edit_type": "silence",
                    "keep": False,
                    "reason": "Silence",
                    "confidence": 0.9,
                }
            ],
        )

        result = project.to_result()

        assert result.original_duration == 100.0
        assert result.edited_duration == 80.0
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello"
        assert len(result.edit_decisions) == 1


class TestProjectVersioning:
    """Tests for project version handling."""

    def test_version_in_saved_file(self, temp_dir):
        import json

        from opengling.core.project import PROJECT_VERSION, Project

        project = Project()
        save_path = temp_dir / "test.opengling"
        project.save(save_path)

        with open(save_path) as f:
            data = json.load(f)

        assert data["version"] == PROJECT_VERSION

