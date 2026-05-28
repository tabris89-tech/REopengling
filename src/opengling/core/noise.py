"""Noise removal using spectral gating and ML-based denoising."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from opengling.core.models import ProcessingConfig

logger = logging.getLogger(__name__)


class NoiseRemover:
    """Removes background noise from audio using spectral gating."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def remove_noise(
        self,
        audio_path: Path | str,
        output_path: Optional[Path | str] = None,
        strength: Optional[float] = None,
    ) -> Path:
        """
        Remove background noise from audio file.

        Args:
            audio_path: Path to input audio file
            output_path: Path for output file (auto-generated if None)
            strength: Noise reduction strength 0.0-1.0 (uses config if None)

        Returns:
            Path to denoised audio file
        """
        if not self.config.remove_noise:
            return Path(audio_path)

        audio_path = Path(audio_path)
        strength = strength if strength is not None else self.config.noise_reduction_strength

        if output_path is None:
            output_path = audio_path.parent / f"{audio_path.stem}_denoised{audio_path.suffix}"
        output_path = Path(output_path)

        logger.info(f"Removing noise from {audio_path.name} (strength: {strength})")

        try:
            return self._remove_with_noisereduce(audio_path, output_path, strength)
        except Exception as e:
            logger.warning(f"noisereduce failed: {e}, trying FFmpeg fallback")
            return self._remove_with_ffmpeg(audio_path, output_path, strength)

    def _remove_with_noisereduce(
        self,
        audio_path: Path,
        output_path: Path,
        strength: float,
    ) -> Path:
        """Remove noise using noisereduce library (spectral gating)."""
        try:
            import noisereduce as nr
            from pydub import AudioSegment
            from scipy.io import wavfile
        except ImportError:
            raise ImportError(
                "noisereduce and scipy are required. "
                "Install with: pip install noisereduce scipy"
            )

        # Load audio
        audio = AudioSegment.from_file(str(audio_path))
        sample_rate = audio.frame_rate

        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Handle stereo
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
            # Process each channel
            denoised_channels = []
            for channel in range(2):
                channel_data = samples[:, channel]
                # Normalize
                channel_data = channel_data / np.max(np.abs(channel_data))
                # Apply noise reduction
                denoised = nr.reduce_noise(
                    y=channel_data,
                    sr=sample_rate,
                    prop_decrease=strength,
                    stationary=False,  # Handles non-stationary noise better
                    n_fft=2048,
                    hop_length=512,
                )
                denoised_channels.append(denoised)
            denoised = np.column_stack(denoised_channels)
        else:
            # Mono
            samples = samples / np.max(np.abs(samples))
            denoised = nr.reduce_noise(
                y=samples,
                sr=sample_rate,
                prop_decrease=strength,
                stationary=False,
                n_fft=2048,
                hop_length=512,
            )

        # Convert back to int16
        denoised = (denoised * 32767).astype(np.int16)

        # Save with scipy, then convert to match original format
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        if len(denoised.shape) == 1:
            wavfile.write(str(tmp_path), sample_rate, denoised)
        else:
            wavfile.write(str(tmp_path), sample_rate, denoised)

        # Convert to original format if needed
        if output_path.suffix.lower() != ".wav":
            denoised_audio = AudioSegment.from_wav(str(tmp_path))
            denoised_audio.export(str(output_path), format=output_path.suffix[1:])
            tmp_path.unlink()
        else:
            tmp_path.rename(output_path)

        logger.info(f"Saved denoised audio to {output_path}")
        return output_path

    def _remove_with_ffmpeg(
        self,
        audio_path: Path,
        output_path: Path,
        strength: float,
    ) -> Path:
        """Remove noise using FFmpeg's afftdn filter."""
        try:
            import ffmpeg
        except ImportError:
            raise ImportError(
                "ffmpeg-python is required. Install with: pip install ffmpeg-python"
            )

        # Map strength (0-1) to FFmpeg's noise reduction (0-100)
        nr_level = int(strength * 50)  # Keep it moderate

        try:
            (
                ffmpeg
                .input(str(audio_path))
                .output(
                    str(output_path),
                    af=f"afftdn=nr={nr_level}:nf=-20",  # Noise reduction filter
                )
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr}")
            raise

        logger.info(f"Saved denoised audio to {output_path}")
        return output_path


class NoiseProfiler:
    """Analyzes audio to create a noise profile for better removal."""

    def analyze_noise_profile(
        self,
        audio_path: Path | str,
        noise_sample_duration: float = 1.0,
    ) -> np.ndarray:
        """
        Analyze the beginning of audio to create a noise profile.

        Assumes first N seconds contain representative background noise.

        Args:
            audio_path: Path to audio file
            noise_sample_duration: Duration of noise sample in seconds

        Returns:
            Noise profile as numpy array
        """
        try:
            import numpy as np
            from pydub import AudioSegment
        except ImportError:
            raise ImportError("pydub is required")

        audio = AudioSegment.from_file(str(audio_path))

        # Get first N seconds
        sample_ms = int(noise_sample_duration * 1000)
        noise_sample = audio[:sample_ms]

        # Convert to numpy
        samples = np.array(noise_sample.get_array_of_samples(), dtype=np.float32)

        # Compute spectral profile
        from scipy.fft import rfft

        # Use windowed FFT
        window_size = 2048
        hop_size = 512

        spectra = []
        for i in range(0, len(samples) - window_size, hop_size):
            window = samples[i:i + window_size]
            spectrum = np.abs(rfft(window))
            spectra.append(spectrum)

        # Average spectrum = noise profile
        noise_profile = np.mean(spectra, axis=0)

        return noise_profile


def estimate_snr(audio_path: Path | str) -> float:
    """
    Estimate Signal-to-Noise Ratio of audio file.

    Returns:
        Estimated SNR in dB
    """
    try:
        import numpy as np
        from pydub import AudioSegment
    except ImportError:
        return 0.0

    audio = AudioSegment.from_file(str(audio_path))
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)

    # Simple SNR estimation using signal variance
    # Assumes noise is in quiet parts

    # Find quiet parts (below threshold)
    threshold = np.percentile(np.abs(samples), 10)
    quiet_samples = samples[np.abs(samples) < threshold]
    loud_samples = samples[np.abs(samples) >= threshold]

    if len(quiet_samples) < 100 or len(loud_samples) < 100:
        return 20.0  # Assume good SNR if can't estimate

    noise_power = np.var(quiet_samples)
    signal_power = np.var(loud_samples)

    if noise_power == 0:
        return 40.0  # Very clean audio

    snr = 10 * np.log10(signal_power / noise_power)
    return float(snr)

