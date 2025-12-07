"""Configuration file support for OpenGling.

Supports loading configuration from:
- .opengling.yaml / .opengling.yml
- opengling.yaml / opengling.yml
- opengling.toml
- pyproject.toml (under [tool.opengling] section)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Any

from opengling.core.models import ProcessingConfig, ExportFormat

logger = logging.getLogger(__name__)

# Config file names to search for (in order of priority)
CONFIG_FILES = [
    '.opengling.yaml',
    '.opengling.yml',
    'opengling.yaml',
    'opengling.yml',
    'opengling.toml',
    'pyproject.toml',
]


def find_config_file(start_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find configuration file by walking up the directory tree.
    
    Args:
        start_dir: Directory to start searching from (defaults to cwd)
        
    Returns:
        Path to config file if found, None otherwise
    """
    if start_dir is None:
        start_dir = Path.cwd()
    
    current = start_dir.resolve()
    
    # Walk up to root
    while current != current.parent:
        for config_name in CONFIG_FILES:
            config_path = current / config_name
            if config_path.exists():
                logger.debug(f"Found config file: {config_path}")
                return config_path
        current = current.parent
    
    # Check root
    for config_name in CONFIG_FILES:
        config_path = current / config_name
        if config_path.exists():
            return config_path
    
    return None


def load_config_file(path: Path) -> dict[str, Any]:
    """
    Load configuration from file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    suffix = path.suffix.lower()
    
    if suffix in ('.yaml', '.yml'):
        return _load_yaml(path)
    elif suffix == '.toml':
        return _load_toml(path)
    else:
        logger.warning(f"Unknown config file format: {path}")
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed, skipping YAML config")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as e:
        logger.warning(f"Failed to load YAML config: {e}")
        return {}


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            logger.warning("tomllib/tomli not installed, skipping TOML config")
            return {}
    
    try:
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        
        # For pyproject.toml, look in [tool.opengling] section
        if path.name == 'pyproject.toml':
            return data.get('tool', {}).get('opengling', {})
        
        return data
    except Exception as e:
        logger.warning(f"Failed to load TOML config: {e}")
        return {}


def merge_config(
    base: ProcessingConfig,
    file_config: dict[str, Any],
) -> ProcessingConfig:
    """
    Merge file configuration into base configuration.
    
    File config values override base values.
    
    Args:
        base: Base ProcessingConfig
        file_config: Configuration dictionary from file
        
    Returns:
        Merged ProcessingConfig
    """
    # Map config file keys to ProcessingConfig attributes
    key_mapping = {
        'silence_threshold': 'silence_threshold',
        'silence_padding': 'silence_padding',
        'remove_silences': 'remove_silences',
        'remove_fillers': 'remove_fillers',
        'detect_bad_takes': 'detect_bad_takes',
        'remove_noise': 'remove_noise',
        'noise_reduction_strength': 'noise_reduction_strength',
        'auto_zoom': 'auto_zoom',
        'max_zoom': 'max_zoom',
        'zoom_smoothing': 'zoom_smoothing',
        'whisper_model': 'whisper_model',
        'language': 'language',
        'device': 'device',
        'compute_type': 'compute_type',
        'ollama_model': 'ollama_model',
        # Aliases
        'model': 'whisper_model',
        'noise': 'remove_noise',
        'zoom': 'auto_zoom',
    }
    
    # Create a copy of base config as dict
    config_dict = {
        'remove_silences': base.remove_silences,
        'silence_threshold': base.silence_threshold,
        'silence_padding': base.silence_padding,
        'remove_fillers': base.remove_fillers,
        'filler_words': list(base.filler_words),
        'detect_bad_takes': base.detect_bad_takes,
        'restart_detection': base.restart_detection,
        'low_confidence_threshold': base.low_confidence_threshold,
        'remove_noise': base.remove_noise,
        'noise_reduction_strength': base.noise_reduction_strength,
        'auto_zoom': base.auto_zoom,
        'zoom_smoothing': base.zoom_smoothing,
        'max_zoom': base.max_zoom,
        'whisper_model': base.whisper_model,
        'language': base.language,
        'generate_youtube_metadata': base.generate_youtube_metadata,
        'ollama_model': base.ollama_model,
        'output_format': base.output_format,
        'caption_format': base.caption_format,
        'device': base.device,
        'compute_type': base.compute_type,
    }
    
    # Apply file config values
    for file_key, value in file_config.items():
        config_key = key_mapping.get(file_key, file_key)
        
        if config_key in config_dict:
            # Handle special cases
            if config_key == 'output_format' and isinstance(value, str):
                try:
                    value = ExportFormat(value)
                except ValueError:
                    logger.warning(f"Invalid output_format: {value}")
                    continue
            
            if config_key == 'caption_format' and isinstance(value, str):
                try:
                    value = ExportFormat(value)
                except ValueError:
                    logger.warning(f"Invalid caption_format: {value}")
                    continue
            
            config_dict[config_key] = value
            logger.debug(f"Config: {config_key} = {value}")
        
        # Handle custom filler words
        elif file_key == 'custom_fillers' and isinstance(value, list):
            config_dict['filler_words'] = list(set(config_dict['filler_words'] + value))
    
    return ProcessingConfig(**config_dict)


def load_config(
    base: Optional[ProcessingConfig] = None,
    config_path: Optional[Path] = None,
) -> ProcessingConfig:
    """
    Load configuration, merging with any config file found.
    
    Args:
        base: Base configuration (defaults to ProcessingConfig defaults)
        config_path: Explicit path to config file (auto-detect if None)
        
    Returns:
        Merged ProcessingConfig
    """
    if base is None:
        base = ProcessingConfig()
    
    # Find config file
    if config_path is None:
        config_path = find_config_file()
    
    if config_path is None:
        logger.debug("No config file found, using defaults")
        return base
    
    # Load and merge
    logger.info(f"Loading config from {config_path}")
    file_config = load_config_file(config_path)
    
    if not file_config:
        return base
    
    return merge_config(base, file_config)


def generate_example_config() -> str:
    """
    Generate an example YAML configuration file.
    
    Returns:
        Example configuration as YAML string
    """
    return """# OpenGling Configuration
# Place this file as .opengling.yaml in your project directory

# Silence detection
remove_silences: true
silence_threshold: 0.5  # seconds - minimum silence duration to remove
silence_padding: 0.1    # seconds - padding to keep around speech

# Filler word detection
remove_fillers: true
custom_fillers:        # Add custom filler words to detect
  - "basically"
  - "sort of"

# Bad takes detection
detect_bad_takes: true

# Noise removal (off by default)
remove_noise: false
noise_reduction_strength: 0.5  # 0.0 to 1.0

# Auto-zoom (off by default)
auto_zoom: false
max_zoom: 1.5

# Whisper transcription
whisper_model: base  # tiny, base, small, medium, large-v3
language: null       # Auto-detect if null

# Device settings
device: auto         # auto, cuda, cpu
compute_type: auto   # auto, float16, int8

# YouTube metadata generation (requires Ollama)
generate_youtube_metadata: false
ollama_model: llama3.2
"""

