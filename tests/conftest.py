# pyright: reportUnknownMemberType=false
from __future__ import annotations

from collections import ChainMap
from hunterHearsPy import readAudioFile, stft
from tests import (
	AspectSpectrogram, AspectSpectrogramMagnitude, AspectSpectrogramPower, AspectTensor, AspectWaveform, ContestFilename, ContestSpectrogram,
	ContestSpectrogramMagnitude, ContestTensor, ContestTensorSpectrogram, ContestTensorSpectrogramMagnitude, ContestWaveform,
	listPathFilenamesContests, listPathFilenamesDataSamples, messageTestFailure, pathFilenameMixture)
from tests.dataSamples import expected
from tests.dataSamples.SpeakSoftly_BrokenMan60sec import expected as contestExpected
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

def assert_contest(actual: float | None, expected: float | None, pytest_rel: float, pytest_abs: float, analyzer: str, paths: ContestFilename, sampleRate: int) -> None:
	assert actual == pytest.approx(expected, rel=pytest_rel, abs=pytest_abs, nan_ok=True), messageTestFailure(
		actual, expected, analyzer, pathFilenameAlfa=paths.pathFilenameAlfa.name, pathFilenameBeta=paths.pathFilenameBeta.name, sampleRate=sampleRate
	)

#================== Audio Aspects =================================================================

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
	return AspectTensor(aspectWaveform.pathFilename, torch.from_numpy(aspectWaveform.waveform), aspectWaveform.sampleRate)

@pytest.fixture(scope='session')
def expectedAspect(request: pytest.FixtureRequest, pathFilename: Path) -> float | None:
	"""Return the stored expected aspect value for the current aspect fixture, function, and path."""
	analyzer: str = request.param
	dictionaryExpected: ChainMap[str, dict[str, float | None]] = ChainMap(expected.expectedFilename, expected.expectedSpectrogram, expected.expectedTensor, expected.expectedWaveform)  # pyright: ignore[reportArgumentType] # ty:ignore[invalid-assignment, invalid-argument-type]
	return dictionaryExpected[analyzer][pathFilename.name]  # pyright: ignore[reportReturnType]

#================== Contests ======================================================================

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
	waveformAlfa: Waveform = readAudioFile(pathFilenamesContest.pathFilenameAlfa)
	waveformBeta: Waveform = readAudioFile(pathFilenamesContest.pathFilenameBeta)
	sampleRate = 44100
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
def tensorAudioMixture(aPathFilename: Path = pathFilenameMixture) -> Tensor:
	"""Return the audio mixture tensor with its sample rate."""
	return torch.from_numpy(readAudioFile(aPathFilename))

@pytest.fixture(scope='session')
def expectedContest(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilename) -> float:
	"""Return the stored expected contest value for the current contest fixture, function, and path pair."""
	analyzer: str = request.param
	dictionaryExpectedContest = ChainMap(contestExpected.expectedTensorSpectrogram, contestExpected.expectedSpectrogram, contestExpected.expectedTensor)
	keyName = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return dictionaryExpectedContest[analyzer][keyName]
