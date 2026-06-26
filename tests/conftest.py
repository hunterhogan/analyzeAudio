# pyright: reportUnknownMemberType=false
from __future__ import annotations

from collections import ChainMap
from hunterHearsPy import readAudioFile, stft
from tests import (
	ContestPathFilenames, ContestSpectrogram, ContestSpectrogramMagnitude, ContestSpectrograms, ContestTensor, ContestTensorSpectrogram,
	ContestTensorSpectrogramMagnitude, ContestWaveform, ContestWaveforms, listPathFilenamesContests, listPathFilenamesDataSamples,
	messageTestFailure, pathFilenameMixture, SpectrogramAndData, SpectrogramMagnitudeAndData, SpectrogramPowerAndData, TensorAndData,
	WaveformAndData)
from tests.dataSamples import expected
from tests.dataSamples.SpeakSoftly_BrokenMan60sec import expected as contestExpected
from typing import TYPE_CHECKING
import numpy
import pytest
import soundfile
import torch

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy.theTypes import Spectrogram, Waveform
	from pathlib import Path
	from torch import Tensor

#================== Settings =====================================================================

# TODO	"""Install FFmpeg before tests start when GitHub Actions Linux needs it."""

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
	"""Sort collected tests by node id for stable fixture-cache access patterns."""
	items.sort(key=lambda item: item.nodeid)

@pytest.fixture()
def approx_abs(request: pytest.FixtureRequest) -> float:
	"""Return the absolute tolerance for approximate comparisons."""
	return 1e-12

@pytest.fixture()
def approx_rel(request: pytest.FixtureRequest) -> float:
	"""Return the relative tolerance for approximate comparisons."""
	return 1e-6

@pytest.fixture()
def atol(request: pytest.FixtureRequest) -> float:
	"""Return the absolute tolerance for `numpy.allclose` comparisons."""
	return 1e-08

@pytest.fixture()
def rtol(request: pytest.FixtureRequest) -> float:
	"""Return the relative tolerance for `numpy.allclose` comparisons."""
	return 1e-05

#================== Assert =======================================================================

def assert_approx(actual: float | None, expected: float | None, pytest_rel: float, pytest_abs: float, analyzer: str, pathFilename: Path, pytorchOnCPU: bool | None = None) -> None:
	assert actual == pytest.approx(expected, rel=pytest_rel, abs=pytest_abs, nan_ok=True), messageTestFailure(
		actual, expected, analyzer, pathFilename=pathFilename.name, **({} if pytorchOnCPU is None else {'pytorchOnCPU': pytorchOnCPU})
	)

def assert_contest(actual: float | None, expected: float | None, pytest_rel: float, pytest_abs: float, analyzer: str, contestPathFilenames: ContestPathFilenames, sampleRate: int) -> None:
	assert actual == pytest.approx(expected, rel=pytest_rel, abs=pytest_abs, nan_ok=True), messageTestFailure(
		actual, expected, analyzer, pathFilenameAlfa=contestPathFilenames.alfa.name, pathFilenameBeta=contestPathFilenames.beta.name, sampleRate=sampleRate
	)

#================== Audio and data =================================================================

@pytest.fixture(params=listPathFilenamesDataSamples, ids=lambda pathFilename: pathFilename.name, scope='session')
def pathFilename(request: pytest.FixtureRequest) -> Path:
	"""Return each static audio data-sample path."""
	pathFilename: Path = request.param
	return pathFilename

@pytest.fixture(scope='session')
def waveformAndData(pathFilename: Path) -> WaveformAndData:
	"""Return each static waveform with its source path and sample rate."""
	# TODO Use the sampleRate encoded in pathFilename.stem and `readAudioFile`.
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform: Audio = readSoundFile.read(dtype='float32').astype(numpy.float32)
	waveform = waveform.T
	return WaveformAndData(pathFilename, waveform, sampleRate)

@pytest.fixture(scope='session')
def spectrogramAndData(waveformAndData: WaveformAndData) -> SpectrogramAndData:
	"""Return each static spectrogram with its source path and sample rate."""
	spectrogram: Spectrogram = stft(numpy.atleast_2d(waveformAndData.waveform), sampleRate=waveformAndData.sampleRate)
	return SpectrogramAndData(waveformAndData.pathFilename, spectrogram, waveformAndData.sampleRate)

@pytest.fixture(scope='session')
def spectrogramMagnitudeAndData(spectrogramAndData: SpectrogramAndData) -> SpectrogramMagnitudeAndData:
	"""Return each static spectrogram magnitude with its source path and sample rate."""
	spectrogramMagnitude: SpectrogramMagnitude = numpy.absolute(spectrogramAndData.spectrogram)
	return SpectrogramMagnitudeAndData(spectrogramAndData.pathFilename, spectrogramMagnitude, spectrogramAndData.sampleRate)

@pytest.fixture(scope='session')
def spectrogramPowerAndData(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData) -> SpectrogramPowerAndData:
	"""Return each static spectrogram power with its source path and sample rate."""
	spectrogramPower: SpectrogramPower = spectrogramMagnitudeAndData.spectrogramMagnitude**2
	return SpectrogramPowerAndData(spectrogramMagnitudeAndData.pathFilename, spectrogramPower, spectrogramMagnitudeAndData.sampleRate)

@pytest.fixture(scope='session')
def tensorAndData(waveformAndData: WaveformAndData) -> TensorAndData:
	"""Return each static audio tensor with its source path and sample rate."""
	return TensorAndData(waveformAndData.pathFilename, torch.from_numpy(waveformAndData.waveform), waveformAndData.sampleRate)

@pytest.fixture(scope='session')
def expectedAspect(request: pytest.FixtureRequest, pathFilename: Path) -> float | None:
	"""Return the stored expected aspect value for the current aspect fixture, function, and path."""
	analyzer: str = request.param
	dictionaryExpected: ChainMap[str, dict[str, float | None]] = ChainMap(expected.expectedFilename, expected.expectedSpectrogram, expected.expectedTensor, expected.expectedWaveform)  # pyright: ignore[reportArgumentType] # ty:ignore[invalid-assignment, invalid-argument-type]
	return dictionaryExpected[analyzer][pathFilename.name]  # pyright: ignore[reportReturnType]

#================== Contests ======================================================================

def _idContestFilename(contestPathFilenames: ContestPathFilenames) -> str:
	return f'{contestPathFilenames.alfa.stem}--{contestPathFilenames.beta.stem}'

@pytest.fixture(params=listPathFilenamesContests, ids=_idContestFilename, scope='session')
def pathFilenamesContest(request: pytest.FixtureRequest) -> ContestPathFilenames:
	"""Return each matching reference and comparand audio-file pair."""
	pathFilenamesContest: ContestPathFilenames = request.param
	return pathFilenamesContest

@pytest.fixture(scope='session')
def contestWaveform(pathFilenamesContest: ContestPathFilenames) -> ContestWaveform:
	"""Return each contest waveform pair with its sample rates."""
	# be DRY: one function loads files: `waveformAndData`.
	sampleRate = 44100
	waveformAlfa: Waveform = readAudioFile(pathFilenamesContest.alfa, sampleRateDesired=sampleRate)
	waveformBeta: Waveform = readAudioFile(pathFilenamesContest.beta, sampleRateDesired=sampleRate)
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
def Z0Z_contestSpectrogram(contestWaveform: ContestWaveforms) -> ContestSpectrograms:
	"""Return each contest complex-valued spectrogram pair with its sample rates."""
	return ContestSpectrograms(
		alfa=spectrogramAndData(contestWaveform.alfa)
		, beta=spectrogramAndData(contestWaveform.beta)
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
		, torch.view_as_real(torch.from_numpy(contestSpectrogram.spectrogramAlfa))
		, contestSpectrogram.sampleRateAlfa
		, torch.view_as_real(torch.from_numpy(contestSpectrogram.spectrogramBeta))
		, contestSpectrogram.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestTensorSpectrogramMagnitude(contestSpectrogramMagnitude: ContestSpectrogramMagnitude) -> ContestTensorSpectrogramMagnitude:
	"""Return each contest magnitude spectrogram tensor pair with its sample rates."""
	return ContestTensorSpectrogramMagnitude(
		contestSpectrogramMagnitude.paths
		, torch.from_numpy(contestSpectrogramMagnitude.spectrogramMagnitudeAlfa)
		, contestSpectrogramMagnitude.sampleRateAlfa
		, torch.from_numpy(contestSpectrogramMagnitude.spectrogramMagnitudeBeta)
		, contestSpectrogramMagnitude.sampleRateBeta
	)

@pytest.fixture(scope='session')
def contestTensor(contestWaveform: ContestWaveform) -> ContestTensor:
	"""Return each contest complex-valued spectrogram pair with its sample rates."""
	return ContestTensor(
		contestWaveform.paths
		, torch.from_numpy(contestWaveform.waveformAlfa)
		, contestWaveform.sampleRateAlfa
		, torch.from_numpy(contestWaveform.waveformBeta)
		, contestWaveform.sampleRateBeta
	)

@pytest.fixture(scope='session')
def expectedContest(request: pytest.FixtureRequest, pathFilenamesContest: ContestPathFilenames) -> float:
	"""Return the stored expected contest value for the current contest fixture, function, and path pair."""
	analyzer: str = request.param
	dictionaryExpectedContest = ChainMap(contestExpected.expectedTensorSpectrogram, contestExpected.expectedSpectrogram, contestExpected.expectedTensor)
	keyName = (pathFilenamesContest.alfa.name, pathFilenamesContest.beta.name)
	return dictionaryExpectedContest[analyzer][keyName]

@pytest.fixture(scope='session')
def tensorAudioMixture(aPathFilename: Path = pathFilenameMixture) -> Tensor:
	"""Return the audio mixture tensor with its sample rate."""
	return torch.from_numpy(readAudioFile(aPathFilename))
