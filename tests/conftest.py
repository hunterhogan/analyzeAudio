# ruff: noqa: DOC201
from __future__ import annotations

from hunterHearsPy import parametersDEFAULT, readAudioFile, stft
from tests import (
	AspectSpectrogram, AspectSpectrogramMagnitude, AspectSpectrogramPower, AspectTensor, AspectWaveform, ContestFilename, ContestSpectrogram,
	ContestSpectrogramMagnitude, ContestTensor, ContestTensorSpectrogram, ContestTensorSpectrogramMagnitude, ContestWaveform,
	listPathFilenamesContests, listPathFilenamesDataSamples, pathFilenameMixture)
from tests.dataSamples.SpeakSoftly_BrokenMan60sec import expected
from typing import TYPE_CHECKING
import librosa
import numpy
import pytest
import soundfile
import torch

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy import Spectrogram, Waveform
	from pathlib import Path
	from torch import Tensor

# TODO abstract the expected values dictionaries into a fixture. This `from tests.dataSamples.expected
# import expectedFilename` is not scalable. Right now, the expected values are effectively indexed by
# keys: tests.dataSamples.expected, expectedFilename, analyzeAbs_Peak_countTotal, and pathFilename.
# Therefore, abstract the keys to those that actually matter--analyzeAbs_Peak_countTotal and
# pathFilename, the function being tested and the test parameters--and have the test function get the
# value from a fixture. The fixture will track the physical location of the expected values.

# ================== Audio Aspects =================================================================

@pytest.fixture(params=listPathFilenamesDataSamples, ids=lambda pathFilename: pathFilename.name, scope='session')
def pathFilename(request: pytest.FixtureRequest) -> Path:
	"""Return each static audio data-sample path."""
	pathFilename: Path = request.param
	return pathFilename

@pytest.fixture(scope='session')
def aspectWaveform(pathFilename: Path) -> AspectWaveform:
	"""Return each static waveform with its source path and sample rate."""
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform: Audio = readSoundFile.read(dtype='float32').astype(numpy.float32)
	waveform = waveform.T
	return AspectWaveform(pathFilename, waveform, sampleRate)

@pytest.fixture(scope='session')
def aspectSpectrogram(aspectWaveform: AspectWaveform) -> AspectSpectrogram:
	"""Return each static spectrogram with its source path and sample rate."""
	spectrogram: Spectrogram = librosa.stft(aspectWaveform.waveform)
	return AspectSpectrogram(aspectWaveform.pathFilename, spectrogram, aspectWaveform.sampleRate)

@pytest.fixture(scope='session')
def aspectSpectrogramMagnitude(aspectSpectrogram: AspectSpectrogram) -> AspectSpectrogramMagnitude:
	"""Return each static spectrogram magnitude with its source path and sample rate."""
	spectrogramMagnitude: SpectrogramMagnitude = numpy.absolute(aspectSpectrogram.spectrogram)
	return AspectSpectrogramMagnitude(aspectSpectrogram.pathFilename, spectrogramMagnitude, aspectSpectrogram.sampleRate)

@pytest.fixture(scope='session')
def aspectSpectrogramPower(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude) -> AspectSpectrogramPower:
	"""Return each static spectrogram power with its source path and sample rate."""
	spectrogramPower: SpectrogramPower = aspectSpectrogramMagnitude.spectrogramMagnitude**2
	return AspectSpectrogramPower(aspectSpectrogramMagnitude.pathFilename, spectrogramPower, aspectSpectrogramMagnitude.sampleRate)

@pytest.fixture(scope='session')
def aspectTensor(aspectWaveform: AspectWaveform) -> AspectTensor:
	"""Return each static audio tensor with its source path and sample rate."""
	return AspectTensor(aspectWaveform.pathFilename, torch.from_numpy(aspectWaveform.waveform), aspectWaveform.sampleRate)  # pyright: ignore[reportUnknownMemberType]

# ================== Contests ======================================================================

def _idContestFilename(paths: ContestFilename) -> str:
	return f'{paths.pathFilenameAlfa.stem}--{paths.pathFilenameBeta.stem}'

@pytest.fixture(params=listPathFilenamesContests, ids=_idContestFilename, scope='session')
def pathFilenamesContest(request: pytest.FixtureRequest) -> ContestFilename:
	"""Return each matching reference and comparand audio-file pair."""
	pathFilenamesContest: ContestFilename = request.param
	return pathFilenamesContest

@pytest.fixture(scope='session')
def contestWaveform(pathFilenamesContest: ContestFilename) -> ContestWaveform:
	"""Return each contest waveform pair with its sample rates."""
	# TODO Think about this.
	sampleRate: int = int(parametersDEFAULT['sampleRate'])
	waveformAlfa: Waveform = readAudioFile(pathFilenamesContest.pathFilenameAlfa, sampleRate=sampleRate)
	waveformBeta: Waveform = readAudioFile(pathFilenamesContest.pathFilenameBeta, sampleRate=sampleRate)
	return ContestWaveform(pathFilenamesContest, waveformAlfa, sampleRate, waveformBeta, sampleRate)

@pytest.fixture(scope='session')
def contestSpectrogram(contestWaveform: ContestWaveform) -> ContestSpectrogram:
	"""Return each contest complex-valued spectrogram pair with its sample rates."""
	return ContestSpectrogram(
		contestWaveform.paths
		, stft(contestWaveform.waveformAlfa, sampleRate=contestWaveform.sampleRateAlfa)
		, contestWaveform.sampleRateAlfa
		, stft(contestWaveform.waveformBeta, sampleRate=contestWaveform.sampleRateBeta)
		, contestWaveform.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestSpectrogramMagnitude(contestSpectrogram: ContestSpectrogram) -> ContestSpectrogramMagnitude:
	"""Return each contest magnitude spectrogram pair with its sample rates."""
	return ContestSpectrogramMagnitude(
		contestSpectrogram.paths
		, numpy.absolute(contestSpectrogram.spectrogramAlfa)
		, contestSpectrogram.sampleRateAlfa
		, numpy.absolute(contestSpectrogram.spectrogramBeta)
		, contestSpectrogram.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestTensorSpectrogram(contestSpectrogram: ContestSpectrogram) -> ContestTensorSpectrogram:
	"""Return each contest complex-valued spectrogram tensor pair with its sample rates."""
	return ContestTensorSpectrogram(
		contestSpectrogram.paths
		, torch.view_as_real(torch.from_numpy(contestSpectrogram.spectrogramAlfa))  # pyright: ignore[reportUnknownMemberType]
		, contestSpectrogram.sampleRateAlfa
		, torch.view_as_real(torch.from_numpy(contestSpectrogram.spectrogramBeta))  # pyright: ignore[reportUnknownMemberType]
		, contestSpectrogram.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestTensorSpectrogramMagnitude(contestSpectrogramMagnitude: ContestSpectrogramMagnitude) -> ContestTensorSpectrogramMagnitude:
	"""Return each contest magnitude spectrogram tensor pair with its sample rates."""
	return ContestTensorSpectrogramMagnitude(
		contestSpectrogramMagnitude.paths
		, torch.from_numpy(contestSpectrogramMagnitude.spectrogramMagnitudeAlfa)  # pyright: ignore[reportUnknownMemberType]
		, contestSpectrogramMagnitude.sampleRateAlfa
		, torch.from_numpy(contestSpectrogramMagnitude.spectrogramMagnitudeBeta)  # pyright: ignore[reportUnknownMemberType]
		, contestSpectrogramMagnitude.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestTensor(contestWaveform: ContestWaveform) -> ContestTensor:
	"""Return each contest complex-valued spectrogram pair with its sample rates."""
	return ContestTensor(
		contestWaveform.paths
		, torch.from_numpy(contestWaveform.waveformAlfa)  # pyright: ignore[reportUnknownMemberType]
		, contestWaveform.sampleRateAlfa
		, torch.from_numpy(contestWaveform.waveformBeta)  # pyright: ignore[reportUnknownMemberType]
		, contestWaveform.sampleRateBeta
	)

@pytest.fixture(scope='session')
def tensorAudioMixture(aPathFilename: Path = pathFilenameMixture) -> Tensor:
	"""Return the audio mixture tensor with its sample rate."""
	return torch.from_numpy(readAudioFile(aPathFilename))  # pyright: ignore[reportUnknownMemberType]

#------------------ Expected values ---------------------------------------------------------------

@pytest.fixture(scope='session')
def expectedContestFilename(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilename) -> float:
	"""Return the stored expected value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return expected.expectedFilename[analyzer][pairFilenames]

@pytest.fixture(scope='session')
def expectedContestSpectrogram(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilename) -> float:
	"""Return the stored expected spectrogram value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return expected.expectedSpectrogram[analyzer][pairFilenames]

@pytest.fixture(scope='session')
def expectedContestTensorSpectrogram(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilename) -> float:
	"""Return the stored expected tensor-spectrogram value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return expected.expectedTensorSpectrogram[analyzer][pairFilenames]

@pytest.fixture(scope='session')
def expectedContestTensor(request: pytest.FixtureRequest, contestTensor: ContestTensor) -> float:
	"""Return the stored expected tensor value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (contestTensor.paths.pathFilenameAlfa.name, contestTensor.paths.pathFilenameBeta.name)
	return expected.expectedTensor[analyzer][pairFilenames]
