from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy.theTypes import Spectrogram, Waveform
	from pathlib import Path
	from torch import Tensor

class SpectrogramAndData(NamedTuple):
	pathFilename: Path
	spectrogram: Spectrogram
	sampleRate: int

class SpectrogramMagnitudeAndData(NamedTuple):
	pathFilename: Path
	spectrogramMagnitude: SpectrogramMagnitude
	sampleRate: int

class SpectrogramPowerAndData(NamedTuple):
	pathFilename: Path
	spectrogramPower: SpectrogramPower
	sampleRate: int

class TensorAndData(NamedTuple):
	pathFilename: Path
	tensorAudio: Tensor
	sampleRate: int

class WaveformAndData(NamedTuple):
	pathFilename: Path
	waveform: Audio
	sampleRate: int

class ContestPathFilenames(NamedTuple):
	alfa: Path
	beta: Path

class ContestSpectrograms(NamedTuple):
	alfa: SpectrogramAndData
	beta: SpectrogramAndData

class ContestSpectrogramsMagnitude(NamedTuple):
	alfa: SpectrogramMagnitudeAndData
	beta: SpectrogramMagnitudeAndData

class ContestTensorSpectrograms(NamedTuple):
	alfa: TensorAndData
	beta: TensorAndData

class ContestTensorSpectrogramsMagnitude(NamedTuple):
	alfa: TensorAndData
	beta: TensorAndData

class ContestTensors(NamedTuple):
	alfa: TensorAndData
	beta: TensorAndData

class ContestWaveforms(NamedTuple):
	alfa: WaveformAndData
	beta: WaveformAndData

# old system
class ContestSpectrogram(NamedTuple):
	paths: ContestPathFilenames
	spectrogramAlfa: Spectrogram
	sampleRateAlfa: int
	spectrogramBeta: Spectrogram
	sampleRateBeta: int
class ContestSpectrogramMagnitude(NamedTuple):
	paths: ContestPathFilenames
	spectrogramMagnitudeAlfa: SpectrogramMagnitude
	sampleRateAlfa: int
	spectrogramMagnitudeBeta: SpectrogramMagnitude
	sampleRateBeta: int
class ContestTensorSpectrogram(NamedTuple):
	paths: ContestPathFilenames
	tensorSpectrogramAlfa: Tensor
	sampleRateAlfa: int
	tensorSpectrogramBeta: Tensor
	sampleRateBeta: int
class ContestTensorSpectrogramMagnitude(NamedTuple):
	paths: ContestPathFilenames
	tensorSpectrogramMagnitudeAlfa: Tensor
	sampleRateAlfa: int
	tensorSpectrogramMagnitudeBeta: Tensor
	sampleRateBeta: int
class ContestTensor(NamedTuple):
	paths: ContestPathFilenames
	tensorAlfa: Tensor
	sampleRateAlfa: int
	tensorBeta: Tensor
	sampleRateBeta: int

class ContestWaveform(NamedTuple):
	paths: ContestPathFilenames
	waveformAlfa: Waveform
	sampleRateAlfa: int
	waveformBeta: Waveform
	sampleRateBeta: int
