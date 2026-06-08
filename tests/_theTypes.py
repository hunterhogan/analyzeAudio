from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy import Spectrogram, Waveform
	from pathlib import Path
	from torch import Tensor

class ContestFilenames(NamedTuple):
	pathFilenameAlfa: Path
	pathFilenameBeta: Path

class ContestSpectrogramMagnitudesSampleRates(NamedTuple):
	pathFilenamesContest: ContestFilenames
	spectrogramMagnitudeAlfa: SpectrogramMagnitude
	sampleRateAlfa: int
	spectrogramMagnitudeBeta: SpectrogramMagnitude
	sampleRateBeta: int

class ContestSpectrogramsSampleRates(NamedTuple):
	pathFilenamesContest: ContestFilenames
	spectrogramAlfa: Spectrogram
	sampleRateAlfa: int
	spectrogramBeta: Spectrogram
	sampleRateBeta: int

class ContestWaveformsSampleRates(NamedTuple):
	pathFilenamesContest: ContestFilenames
	waveformAlfa: Waveform
	sampleRateAlfa: int
	waveformBeta: Waveform
	sampleRateBeta: int

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

class TensorSampleRate(NamedTuple):
	pathFilename: Path
	tensorAudio: Tensor
	sampleRate: int

class WaveformSampleRate(NamedTuple):
	pathFilename: Path
	waveform: Audio
	sampleRate: int
