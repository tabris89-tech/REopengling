"""Bad takes detection using confidence scores and pattern analysis."""

from __future__ import annotations

import logging
import re
from typing import Optional

from opengling.core.models import (
    EditDecision,
    EditType,
    ProcessingConfig,
    TranscriptSegment,
    TranscriptWord,
)

logger = logging.getLogger(__name__)


class BadTakesDetector:
    """Detects bad takes, restarts, and stutters in transcripts."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
    def detect_bad_takes(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EditDecision]:
        """
        Detect bad takes in transcript segments.
        
        Bad takes include:
        - Low confidence speech (mumbling, unclear)
        - Restarts (speaker starts sentence over)
        - Stutters and repeated words
        - Incomplete sentences followed by restarts
        
        Args:
            segments: Transcript segments with word-level timing
            
        Returns:
            List of EditDecisions marking bad takes for removal
        """
        if not self.config.detect_bad_takes:
            return []
            
        logger.info("Detecting bad takes...")
        
        bad_takes = []
        
        # Detect low confidence regions
        bad_takes.extend(self._detect_low_confidence(segments))
        
        # Detect restarts
        if self.config.restart_detection:
            bad_takes.extend(self._detect_restarts(segments))
        
        # Detect stutters
        bad_takes.extend(self._detect_stutters(segments))
        
        # Detect incomplete sentences
        bad_takes.extend(self._detect_incomplete_sentences(segments))
        
        logger.info(f"Found {len(bad_takes)} bad takes")
        return bad_takes
    
    def _detect_low_confidence(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EditDecision]:
        """Detect regions with low transcription confidence."""
        bad_takes = []
        threshold = self.config.low_confidence_threshold
        
        for segment in segments:
            # Check for consecutive low-confidence words
            low_conf_start = None
            low_conf_words = []
            
            for word in segment.words:
                if word.confidence < threshold:
                    if low_conf_start is None:
                        low_conf_start = word.start
                    low_conf_words.append(word)
                else:
                    # End of low confidence region
                    if low_conf_start is not None and len(low_conf_words) >= 2:
                        # Only flag if multiple consecutive low-conf words
                        bad_takes.append(EditDecision(
                            start=low_conf_start,
                            end=low_conf_words[-1].end,
                            edit_type=EditType.BAD_TAKE,
                            keep=False,
                            reason=f"Low confidence speech ({len(low_conf_words)} words)",
                            confidence=0.7,
                        ))
                    low_conf_start = None
                    low_conf_words = []
            
            # Handle end of segment
            if low_conf_start is not None and len(low_conf_words) >= 2:
                bad_takes.append(EditDecision(
                    start=low_conf_start,
                    end=low_conf_words[-1].end,
                    edit_type=EditType.BAD_TAKE,
                    keep=False,
                    reason=f"Low confidence speech ({len(low_conf_words)} words)",
                    confidence=0.7,
                ))
        
        return bad_takes
    
    def _detect_restarts(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EditDecision]:
        """
        Detect sentence restarts where speaker starts over.
        
        Example: "I went to the... I went to the store yesterday"
        The first "I went to the..." should be removed.
        """
        bad_takes = []
        
        # Analyze text for restart patterns
        all_words = []
        for segment in segments:
            all_words.extend(segment.words)
        
        # Look for repeated phrase patterns
        i = 0
        while i < len(all_words) - 3:
            # Get a window of words
            window_size = 5
            window = all_words[i:i + window_size]
            
            if len(window) < 3:
                i += 1
                continue
            
            # Look for this sequence repeated later
            window_text = " ".join(w.word.lower().strip() for w in window[:3])
            
            # Search ahead for similar start
            for j in range(i + 3, min(i + 20, len(all_words) - 2)):
                future_text = " ".join(
                    w.word.lower().strip() for w in all_words[j:j + 3]
                )
                
                # Check similarity (allow for minor differences)
                if self._phrases_similar(window_text, future_text):
                    # Found a restart - mark the first occurrence as bad take
                    # Find where the first attempt ends (pause or interruption)
                    end_idx = j - 1
                    
                    # Look for natural break point
                    for k in range(i + 3, j):
                        gap = all_words[k].start - all_words[k - 1].end
                        if gap > 0.3:  # 300ms pause
                            end_idx = k - 1
                            break
                    
                    if end_idx > i:
                        bad_takes.append(EditDecision(
                            start=all_words[i].start,
                            end=all_words[end_idx].end,
                            edit_type=EditType.BAD_TAKE,
                            keep=False,
                            reason="Sentence restart detected",
                            confidence=0.75,
                        ))
                        i = j  # Skip to the restart
                        break
            
            i += 1
        
        return bad_takes
    
    def _detect_stutters(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EditDecision]:
        """Detect stutters and repeated words."""
        bad_takes = []
        
        for segment in segments:
            words = segment.words
            i = 0
            
            while i < len(words) - 1:
                word1 = words[i].word.lower().strip()
                word1_clean = re.sub(r'[^\w]', '', word1)
                
                # Check for immediate repetition
                j = i + 1
                repetitions = [words[i]]
                
                while j < len(words):
                    word2 = words[j].word.lower().strip()
                    word2_clean = re.sub(r'[^\w]', '', word2)
                    
                    if word1_clean == word2_clean or self._is_stutter(word1_clean, word2_clean):
                        repetitions.append(words[j])
                        j += 1
                    else:
                        break
                
                # If we found repetitions, mark all but the last for removal
                if len(repetitions) > 1:
                    for rep in repetitions[:-1]:  # Keep the last one
                        bad_takes.append(EditDecision(
                            start=rep.start,
                            end=rep.end,
                            edit_type=EditType.BAD_TAKE,
                            keep=False,
                            reason=f"Stutter/repetition: '{rep.word}'",
                            confidence=0.85,
                        ))
                    i = j
                else:
                    i += 1
        
        return bad_takes
    
    def _detect_incomplete_sentences(
        self,
        segments: list[TranscriptSegment],
    ) -> list[EditDecision]:
        """Detect incomplete sentences that are followed by a restart."""
        bad_takes = []
        
        for i, segment in enumerate(segments):
            # Check if segment ends abruptly (no punctuation, short)
            text = segment.text.strip()
            
            # Skip if it has ending punctuation
            if text and text[-1] in '.!?':
                continue
            
            # Check if it's short and followed by another segment starting similarly
            if len(segment.words) < 8 and i + 1 < len(segments):
                next_segment = segments[i + 1]
                
                # Check if next segment starts similarly
                if len(segment.words) >= 2 and len(next_segment.words) >= 2:
                    first_words = " ".join(w.word.lower() for w in segment.words[:2])
                    next_first = " ".join(w.word.lower() for w in next_segment.words[:2])
                    
                    if self._phrases_similar(first_words, next_first):
                        bad_takes.append(EditDecision(
                            start=segment.start,
                            end=segment.end,
                            edit_type=EditType.BAD_TAKE,
                            keep=False,
                            reason="Incomplete sentence (restarted)",
                            confidence=0.7,
                        ))
        
        return bad_takes
    
    def _phrases_similar(self, phrase1: str, phrase2: str) -> bool:
        """Check if two phrases are similar enough to be a restart."""
        # Simple word overlap check
        words1 = set(phrase1.split())
        words2 = set(phrase2.split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1 & words2)
        min_len = min(len(words1), len(words2))
        
        return overlap >= min_len * 0.7
    
    def _is_stutter(self, word1: str, word2: str) -> bool:
        """Check if word2 is a stutter of word1 (partial word)."""
        if not word1 or not word2:
            return False
        
        # Check if one is prefix of the other
        if word1.startswith(word2) or word2.startswith(word1):
            return True
        
        # Check for common stutter patterns
        # e.g., "th-the", "I-I", "wh-what"
        if len(word1) <= 3 and word2.startswith(word1):
            return True
        if len(word2) <= 3 and word1.startswith(word2):
            return True
        
        return False


def get_bad_take_statistics(
    edits: list[EditDecision],
) -> dict[str, int]:
    """Get statistics about detected bad takes."""
    stats = {
        "low_confidence": 0,
        "restarts": 0,
        "stutters": 0,
        "incomplete": 0,
    }
    
    for edit in edits:
        if edit.edit_type == EditType.BAD_TAKE:
            reason = edit.reason.lower()
            if "confidence" in reason:
                stats["low_confidence"] += 1
            elif "restart" in reason:
                stats["restarts"] += 1
            elif "stutter" in reason or "repetition" in reason:
                stats["stutters"] += 1
            elif "incomplete" in reason:
                stats["incomplete"] += 1
    
    return stats

