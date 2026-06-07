from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from analyzeAudio import Audio, Spectrogram, SpectrogramMagnitude, SpectrogramPower
	from pathlib import Path

class SpectrogramMagnitudeSampleRate(NamedTuple):
	pathFilename: Path
	spectrogramMagnitude: SpectrogramMagnitude
	sampleRate: int

class SpectrogramPowerSampleRate(NamedTuple):
	pathFilename: Path
	spectrogramPower: SpectrogramPower
	sampleRate: int

class SpectrogramSampleRate(NamedTuple):
	pathFilename: Path
	spectrogram: Spectrogram
	sampleRate: int

class WaveformSampleRate(NamedTuple):
	pathFilename: Path
	waveform: Audio
	sampleRate: int
