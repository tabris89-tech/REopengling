"""Filler word detection using NLP and transcript analysis."""

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

# Default filler words and patterns
DEFAULT_FILLERS = {
    # Pure fillers
    "um", "uh", "uhm", "uhh", "umm", "er", "ah", "eh",
    "hmm", "hm", "mm", "mhm",
    
    # Common verbal crutches
    "like",  # when used as filler, not comparison
    "basically",
    "actually",
    "literally",
    "honestly",
    "obviously",
    "essentially",
    "definitely",
    
    # Discourse markers (often fillers)
    "you know",
    "i mean",
    "kind of",
    "sort of",
    "right",
    "so",
    "well",
    "anyway",
    "anyways",
}

# Patterns that indicate filler usage
FILLER_PATTERNS = [
    r"\b(um+|uh+|er+|ah+)\b",
    r"\byou know\b",
    r"\bi mean\b",
    r"\bkind of\b",
    r"\bsort of\b",
    r"\blike,?\s+like\b",  # repeated "like"
]


class FillerDetector:
    """Detects filler words in transcripts using NLP."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.filler_words = set(config.filler_words) if config.filler_words else DEFAULT_FILLERS
        self._nlp = None
        
    def _load_nlp(self):
        """Lazy load spaCy model."""
        if self._nlp is not None:
            return
            
        try:
            import spacy
        except ImportError:
            raise ImportError(
                "spaCy is required for filler detection. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model not found. Downloading en_core_web_sm..."
            )
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self._nlp = spacy.load("en_core_web_sm")
    
    def detect_fillers(
        self,
        segments: list[TranscriptSegment],
        use_nlp: bool = True,
    ) -> list[EditDecision]:
        """
        Detect filler words in transcript segments.
        
        Args:
            segments: Transcript segments with word-level timing
            use_nlp: Whether to use NLP for context-aware detection
            
        Returns:
            List of EditDecisions marking filler regions for removal
        """
        if not self.config.remove_fillers:
            return []
            
        logger.info("Detecting filler words...")
        
        fillers = []
        
        for segment in segments:
            segment_fillers = self._detect_in_segment(segment, use_nlp)
            fillers.extend(segment_fillers)
        
        logger.info(f"Found {len(fillers)} filler words/phrases")
        return fillers
    
    def _detect_in_segment(
        self,
        segment: TranscriptSegment,
        use_nlp: bool,
    ) -> list[EditDecision]:
        """Detect fillers in a single segment."""
        fillers = []
        
        # Check each word
        for i, word in enumerate(segment.words):
            word_lower = word.word.lower().strip()
            
            # Remove punctuation for matching
            word_clean = re.sub(r'[^\w\s]', '', word_lower)
            
            # Check if it's a simple filler word
            if word_clean in self.filler_words:
                # Check if it's used as actual filler (not meaningful word)
                is_filler = self._is_filler_usage(segment, i, word_clean, use_nlp)
                
                if is_filler:
                    fillers.append(EditDecision(
                        start=word.start,
                        end=word.end,
                        edit_type=EditType.FILLER_WORD,
                        keep=False,
                        reason=f"Filler word: '{word.word}'",
                        confidence=0.85 if use_nlp else 0.7,
                    ))
        
        # Check for multi-word fillers
        fillers.extend(self._detect_multiword_fillers(segment))
        
        return fillers
    
    def _is_filler_usage(
        self,
        segment: TranscriptSegment,
        word_index: int,
        word: str,
        use_nlp: bool,
    ) -> bool:
        """
        Determine if a word is used as a filler vs. meaningful content.
        
        For example, "like" can be:
        - Filler: "I, like, went to the store"
        - Meaningful: "I like pizza"
        """
        # Pure fillers are always fillers
        pure_fillers = {"um", "uh", "uhm", "uhh", "umm", "er", "ah", "eh", "hmm", "hm", "mm"}
        if word in pure_fillers:
            return True
        
        if not use_nlp:
            # Without NLP, be conservative - only flag pure fillers
            return word in pure_fillers
        
        # Use NLP for context-aware detection
        self._load_nlp()
        
        # Analyze the sentence
        doc = self._nlp(segment.text)
        
        # Find the token corresponding to this word
        words = segment.words
        if word_index >= len(words):
            return False
            
        target_word = words[word_index].word.lower().strip()
        
        for token in doc:
            if token.text.lower() == target_word:
                # Check grammatical role
                
                # "like" as filler is usually:
                # - Not a verb (not "I like pizza")
                # - Not a preposition with object (not "looks like rain")
                # - Often has pauses around it
                if word == "like":
                    if token.pos_ == "VERB":
                        return False  # "I like pizza"
                    if token.pos_ == "ADP" and token.head.pos_ in ("NOUN", "VERB"):
                        return False  # "looks like rain"
                    return True
                
                # "so" as filler is usually at sentence start with no dependent
                if word == "so":
                    if token.i == 0 and not list(token.children):
                        return True
                    return False
                
                # "right" as filler is usually:
                # - At sentence end
                # - Not an adjective modifying noun
                if word == "right":
                    if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
                        return False
                    if token.i == len(doc) - 1:
                        return True
                    return False
                
                # "well" at start of sentence is often filler
                if word == "well":
                    if token.i == 0:
                        return True
                    return False
                
                # "basically", "actually", "literally", "honestly" are often fillers
                if word in {"basically", "actually", "literally", "honestly", "obviously"}:
                    # If they don't modify a specific word meaningfully
                    if token.pos_ == "ADV" and token.head.pos_ not in ("ADJ", "VERB"):
                        return True
                    return True  # Often safe to remove
                
                break
        
        return False
    
    def _detect_multiword_fillers(
        self,
        segment: TranscriptSegment,
    ) -> list[EditDecision]:
        """Detect multi-word filler phrases like 'you know', 'I mean'."""
        fillers = []
        text_lower = segment.text.lower()
        words = segment.words
        
        multiword_fillers = [
            ("you know", 2),
            ("i mean", 2),
            ("kind of", 2),
            ("sort of", 2),
        ]
        
        for phrase, word_count in multiword_fillers:
            if phrase in text_lower:
                # Find the words that make up this phrase
                for i in range(len(words) - word_count + 1):
                    phrase_words = " ".join(
                        w.word.lower().strip() for w in words[i:i + word_count]
                    )
                    
                    # Remove punctuation for comparison
                    phrase_words_clean = re.sub(r'[^\w\s]', '', phrase_words)
                    
                    if phrase in phrase_words_clean:
                        start = words[i].start
                        end = words[i + word_count - 1].end
                        
                        fillers.append(EditDecision(
                            start=start,
                            end=end,
                            edit_type=EditType.FILLER_WORD,
                            keep=False,
                            reason=f"Filler phrase: '{phrase}'",
                            confidence=0.8,
                        ))
        
        return fillers


def get_filler_statistics(
    edits: list[EditDecision],
) -> dict[str, int]:
    """Get statistics about detected fillers."""
    stats = {}
    
    for edit in edits:
        if edit.edit_type == EditType.FILLER_WORD:
            # Extract filler word from reason
            match = re.search(r"'([^']+)'", edit.reason)
            if match:
                filler = match.group(1).lower()
                stats[filler] = stats.get(filler, 0) + 1
    
    return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

