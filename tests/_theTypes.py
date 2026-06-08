from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy import Spectrogram, Waveform
	from pathlib import Path
	from torch import Tensor

class AspectSpectrogram(NamedTuple):
	pathFilename: Path
	spectrogram: Spectrogram
	sampleRate: int

class AspectSpectrogramMagnitude(NamedTuple):
	pathFilename: Path
	spectrogramMagnitude: SpectrogramMagnitude
	sampleRate: int

class AspectSpectrogramPower(NamedTuple):
	pathFilename: Path
	spectrogramPower: SpectrogramPower
	sampleRate: int

class AspectTensor(NamedTuple):
	pathFilename: Path
	tensorAudio: Tensor
	sampleRate: int

class AspectWaveform(NamedTuple):
	pathFilename: Path
	waveform: Audio
	sampleRate: int

class ContestFilename(NamedTuple):
	pathFilenameAlfa: Path
	pathFilenameBeta: Path

class ContestSpectrogram(NamedTuple):
	paths: ContestFilename
	spectrogramAlfa: Spectrogram
	sampleRateAlfa: int
	spectrogramBeta: Spectrogram
	sampleRateBeta: int

class ContestSpectrogramMagnitude(NamedTuple):
	paths: ContestFilename
	spectrogramMagnitudeAlfa: SpectrogramMagnitude
	sampleRateAlfa: int
	spectrogramMagnitudeBeta: SpectrogramMagnitude
	sampleRateBeta: int

class ContestTensor(NamedTuple):
	paths: ContestFilename
	tensorAlfa: Tensor
	sampleRateAlfa: int
	tensorBeta: Tensor
	sampleRateBeta: int

class ContestWaveform(NamedTuple):
	paths: ContestFilename
	waveformAlfa: Waveform
	sampleRateAlfa: int
	waveformBeta: Waveform
	sampleRateBeta: int
