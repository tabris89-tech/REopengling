"""YouTube metadata generation using local LLM (Ollama)."""

from __future__ import annotations

import logging
import re
from typing import Optional

from opengling.core.models import (
    ProcessingConfig,
    TranscriptSegment,
    YouTubeMetadata,
)

logger = logging.getLogger(__name__)


class YouTubeGenerator:
    """Generates YouTube-optimized titles, descriptions, and chapters."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._client = None
        
    def _load_client(self):
        """Lazy load Ollama client."""
        if self._client is not None:
            return
            
        try:
            import ollama
            self._client = ollama
        except ImportError:
            raise ImportError(
                "ollama is required for YouTube generation. "
                "Install with: pip install opengling[youtube]\n"
                "Also ensure Ollama is running: https://ollama.ai"
            )
    
    def generate_metadata(
        self,
        segments: list[TranscriptSegment],
        video_duration: float,
        context: Optional[str] = None,
    ) -> YouTubeMetadata:
        """
        Generate YouTube metadata from transcript.
        
        Args:
            segments: Transcript segments
            video_duration: Total video duration in seconds
            context: Optional context about the video (channel name, niche, etc.)
            
        Returns:
            YouTubeMetadata with title, description, tags, and chapters
        """
        if not self.config.generate_youtube_metadata:
            return YouTubeMetadata(
                title="",
                description="",
                tags=[],
                chapters=[],
            )
            
        logger.info("Generating YouTube metadata...")
        
        self._load_client()
        
        # Combine transcript
        full_transcript = " ".join(seg.text for seg in segments)
        
        # Generate each component
        title = self._generate_title(full_transcript, context)
        description = self._generate_description(full_transcript, context)
        tags = self._generate_tags(full_transcript, title)
        chapters = self._generate_chapters(segments, video_duration)
        
        return YouTubeMetadata(
            title=title,
            description=description,
            tags=tags,
            chapters=chapters,
        )
    
    def _generate_title(
        self,
        transcript: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate an engaging YouTube title."""
        prompt = f"""Based on this video transcript, generate a single compelling YouTube title.
The title should be:
- Under 60 characters
- Attention-grabbing but not clickbait
- Include relevant keywords
- Not use all caps

{f"Context: {context}" if context else ""}

Transcript (first 2000 chars):
{transcript[:2000]}

Respond with ONLY the title, nothing else."""

        try:
            response = self._client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
            )
            title = response['response'].strip().strip('"\'')
            # Clean up any explanatory text
            if '\n' in title:
                title = title.split('\n')[0]
            return title[:100]  # Truncate if too long
        except Exception as e:
            logger.warning(f"Failed to generate title: {e}")
            return "Untitled Video"
    
    def _generate_description(
        self,
        transcript: str,
        context: Optional[str] = None,
    ) -> str:
        """Generate a YouTube description."""
        prompt = f"""Based on this video transcript, write a YouTube video description.
The description should:
- Start with a compelling 1-2 sentence summary
- Include key topics covered
- Be 150-300 words
- Include relevant keywords naturally
- End with a call to action

{f"Context: {context}" if context else ""}

Transcript (first 3000 chars):
{transcript[:3000]}

Write the description:"""

        try:
            response = self._client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
            )
            return response['response'].strip()
        except Exception as e:
            logger.warning(f"Failed to generate description: {e}")
            return "Video description"
    
    def _generate_tags(
        self,
        transcript: str,
        title: str,
    ) -> list[str]:
        """Generate relevant YouTube tags."""
        prompt = f"""Based on this video title and transcript, generate 10-15 relevant YouTube tags.
Tags should be:
- Mix of broad and specific terms
- Include variations and related terms
- Relevant to the content

Title: {title}

Transcript (first 1500 chars):
{transcript[:1500]}

Respond with comma-separated tags only, no explanations:"""

        try:
            response = self._client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
            )
            tags_text = response['response'].strip()
            # Parse comma-separated tags
            tags = [tag.strip().strip('"\'') for tag in tags_text.split(',')]
            # Clean and dedupe
            tags = list(dict.fromkeys(tag for tag in tags if tag and len(tag) < 50))
            return tags[:20]  # YouTube limit is around 500 chars total
        except Exception as e:
            logger.warning(f"Failed to generate tags: {e}")
            return []
    
    def _generate_chapters(
        self,
        segments: list[TranscriptSegment],
        video_duration: float,
    ) -> list[tuple[float, str]]:
        """Generate YouTube chapters from transcript segments."""
        if not segments:
            return []
        
        # Group segments into logical sections (roughly 2-5 min each)
        min_chapter_duration = 60  # 1 minute minimum
        max_chapter_duration = 300  # 5 minutes maximum
        
        chapters = []
        current_chapter_start = 0.0
        current_chapter_text = []
        
        for segment in segments:
            current_chapter_text.append(segment.text)
            chapter_duration = segment.end - current_chapter_start
            
            # Check if we should end this chapter
            should_end = (
                chapter_duration >= max_chapter_duration or
                (chapter_duration >= min_chapter_duration and 
                 self._is_topic_break(segment.text))
            )
            
            if should_end and segment.end < video_duration - 30:
                # Generate chapter title
                chapter_text = " ".join(current_chapter_text)
                chapter_title = self._generate_chapter_title(chapter_text)
                
                chapters.append((current_chapter_start, chapter_title))
                
                # Start new chapter
                current_chapter_start = segment.end
                current_chapter_text = []
        
        # Add first chapter at 0:00 if not already there
        if not chapters or chapters[0][0] > 0:
            first_text = " ".join(seg.text for seg in segments[:5])
            chapters.insert(0, (0.0, self._generate_chapter_title(first_text)))
        
        return chapters
    
    def _is_topic_break(self, text: str) -> bool:
        """Detect if this segment might be a topic break."""
        # Simple heuristics for topic breaks
        break_indicators = [
            "now let's", "moving on", "next", "another thing",
            "the next", "let's talk about", "speaking of",
            "alright so", "okay so", "now,", "finally,"
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in break_indicators)
    
    def _generate_chapter_title(self, text: str) -> str:
        """Generate a short chapter title from text."""
        prompt = f"""Based on this transcript section, write a short YouTube chapter title (2-5 words).
Keep it descriptive and concise.

Text: {text[:500]}

Chapter title (2-5 words only):"""

        try:
            response = self._client.generate(
                model=self.config.ollama_model,
                prompt=prompt,
            )
            title = response['response'].strip().strip('"\'')
            # Clean up
            title = title.split('\n')[0]
            # Truncate if needed
            words = title.split()[:6]
            return " ".join(words)
        except Exception as e:
            logger.warning(f"Failed to generate chapter title: {e}")
            return "Section"


def format_chapters_for_youtube(
    chapters: list[tuple[float, str]],
) -> str:
    """
    Format chapters for YouTube description.
    
    Args:
        chapters: List of (timestamp, title) tuples
        
    Returns:
        Formatted chapter string for YouTube description
    """
    lines = []
    for timestamp, title in chapters:
        minutes = int(timestamp // 60)
        seconds = int(timestamp % 60)
        lines.append(f"{minutes}:{seconds:02d} {title}")
    
    return "\n".join(lines)


def generate_hashtags(tags: list[str], limit: int = 3) -> str:
    """
    Generate hashtags from tags for YouTube description.
    
    Args:
        tags: List of tags
        limit: Maximum number of hashtags
        
    Returns:
        Formatted hashtag string
    """
    hashtags = []
    for tag in tags[:limit]:
        # Remove spaces and special chars
        hashtag = re.sub(r'[^a-zA-Z0-9]', '', tag)
        if hashtag:
            hashtags.append(f"#{hashtag}")
    
    return " ".join(hashtags)

