# pyright: reportUnknownMemberType=false
from __future__ import annotations

from analyzeAudio import settingsPackage
from analyzeAudio.ffmpeg import FFmpegGitHub
from collections import ChainMap
from hunterHearsPy import readAudioFile, stft
from operator import attrgetter
from tests.dataSamples import expected
from tests.dataSamples.audioFiles import (
	lookupPathFilenamesAudio, lookupPathFilenamesContest, pathFilenameMixture, sampleRateSpeakSoftly)
from tests.dataSamples.SpeakSoftly_BrokenMan60sec import expected as expectedSpeakSoftly
from typing import TYPE_CHECKING
import numpy
import pytest
import soundfile
import torch

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from collections.abc import Iterator
	from hunterHearsPy.theTypes import Spectrogram, Waveform
	from hunterMakesPy import CallableFunction
	from pathlib import Path
	from torch import Tensor

def pytest_configure(config: pytest.Config) -> None:
	"""Install FFmpeg when a GitHub Actions Linux runner lacks FFmpeg 7 or later."""
	FFmpegGitHub()

@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
	"""Sort collected tests by node id for stable fixture-cache access patterns."""
	items.sort(key=attrgetter('nodeid'))

@pytest.fixture(scope='session')
def pathDataSamples() -> Path:
	"""Return the directory that holds the audio samples."""
	return settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'

@pytest.fixture(params=tuple(lookupPathFilenamesAudio.values()), ids=tuple(lookupPathFilenamesAudio.keys()), scope='session')
def pathFilename(request: pytest.FixtureRequest, pathDataSamples: Path) -> Path:
	"""Return the path of each audio file in `lookupPathFilenamesAudio`."""
	return pathDataSamples / request.param

@pytest.fixture(scope='session')
def sampleRate(pathFilename: Path) -> int:
	"""Return the sample rate stored in the audio file at `pathFilename`."""
	return soundfile.info(pathFilename).samplerate

@pytest.fixture(scope='session')
def waveform(pathFilename: Path) -> Audio:
	"""Return the float32 waveform of the audio file at `pathFilename`, with channels on the first axis."""
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		return readSoundFile.read(dtype='float32').astype(numpy.float32, copy=False).T

@pytest.fixture(scope='session')
def spectrogramMagnitude(waveform: Audio, sampleRate: int) -> SpectrogramMagnitude:
	"""Return the magnitude of the spectrogram of `waveform`, with channels on the first axis."""
	return numpy.absolute(stft(numpy.atleast_2d(waveform), sampleRate=sampleRate))

@pytest.fixture(scope='session')
def spectrogramPower(spectrogramMagnitude: SpectrogramMagnitude) -> SpectrogramPower:
	"""Return the power of the spectrogram of `waveform`."""
	return spectrogramMagnitude**2

@pytest.fixture(scope='session')
def tensorAudio(waveform: Audio) -> Tensor:
	"""Return a tensor that shares memory with `waveform`."""
	return torch.from_numpy(waveform)

@pytest.fixture
def expectedAspect(analyzer: CallableFunction[..., object], pathFilename: Path) -> float | None:
	"""Return the accepted recording of `analyzer` for the audio file at `pathFilename`."""
	lookupExpected: ChainMap[str, dict[str, float | None]] = ChainMap(
		expected.expectedFilename, expected.expectedSpectrogram, expected.expectedTensor, expected.expectedWaveform)
	return lookupExpected[analyzer.__name__][pathFilename.name]

@pytest.fixture(params=tuple(lookupPathFilenamesContest.values()), ids=tuple(lookupPathFilenamesContest.keys()), scope='session')
def pathFilenamesContest(request: pytest.FixtureRequest, pathDataSamples: Path) -> tuple[Path, Path]:
	"""Return the paths of the reference and the comparand of each contest in `lookupPathFilenamesContest`."""
	pathFilenameAlfa, pathFilenameBeta = request.param
	return pathDataSamples / pathFilenameAlfa, pathDataSamples / pathFilenameBeta

@pytest.fixture(scope='session')
def waveformAlfa(pathFilenamesContest: tuple[Path, Path]) -> Waveform:
	"""Return the waveform of the reference."""
	return readAudioFile(pathFilenamesContest[0], sampleRateSpeakSoftly)

@pytest.fixture(scope='session')
def waveformBeta(pathFilenamesContest: tuple[Path, Path]) -> Waveform:
	"""Return the waveform of the comparand."""
	return readAudioFile(pathFilenamesContest[1], sampleRateSpeakSoftly)

@pytest.fixture(scope='session')
def spectrogramAlfa(waveformAlfa: Waveform) -> Spectrogram:
	"""Return the complex-valued spectrogram of the reference."""
	return stft(waveformAlfa, sampleRate=sampleRateSpeakSoftly)

@pytest.fixture(scope='session')
def spectrogramBeta(waveformBeta: Waveform) -> Spectrogram:
	"""Return the complex-valued spectrogram of the comparand."""
	return stft(waveformBeta, sampleRate=sampleRateSpeakSoftly)

@pytest.fixture(scope='session')
def spectrogramMagnitudeAlfa(spectrogramAlfa: Spectrogram) -> SpectrogramMagnitude:
	"""Return the magnitude of the spectrogram of the reference."""
	return numpy.absolute(spectrogramAlfa)

@pytest.fixture(scope='session')
def spectrogramMagnitudeBeta(spectrogramBeta: Spectrogram) -> SpectrogramMagnitude:
	"""Return the magnitude of the spectrogram of the comparand."""
	return numpy.absolute(spectrogramBeta)

@pytest.fixture(scope='session')
def tensorAudioAlfa(waveformAlfa: Waveform) -> Tensor:
	"""Return a tensor that shares memory with the waveform of the reference."""
	return torch.from_numpy(waveformAlfa)

@pytest.fixture(scope='session')
def tensorAudioBeta(waveformBeta: Waveform) -> Tensor:
	"""Return a tensor that shares memory with the waveform of the comparand."""
	return torch.from_numpy(waveformBeta)

@pytest.fixture(scope='session')
def tensorAudioMixture(pathDataSamples: Path) -> Tensor:
	"""Return a tensor of the waveform of the mixture from which each comparand was separated."""
	return torch.from_numpy(readAudioFile(pathDataSamples / pathFilenameMixture))

@pytest.fixture(scope='session')
def tensorSpectrogramAlfa(spectrogramAlfa: Spectrogram) -> Tensor:
	"""Return the complex-valued spectrogram of the reference as a real tensor with a last axis of length 2."""
	return torch.view_as_real(torch.from_numpy(spectrogramAlfa))

@pytest.fixture(scope='session')
def tensorSpectrogramBeta(spectrogramBeta: Spectrogram) -> Tensor:
	"""Return the complex-valued spectrogram of the comparand as a real tensor with a last axis of length 2."""
	return torch.view_as_real(torch.from_numpy(spectrogramBeta))

@pytest.fixture(scope='session')
def tensorSpectrogramMagnitudeAlfa(spectrogramMagnitudeAlfa: SpectrogramMagnitude) -> Tensor:
	"""Return a tensor that shares memory with the magnitude of the spectrogram of the reference."""
	return torch.from_numpy(spectrogramMagnitudeAlfa)

@pytest.fixture(scope='session')
def tensorSpectrogramMagnitudeBeta(spectrogramMagnitudeBeta: SpectrogramMagnitude) -> Tensor:
	"""Return a tensor that shares memory with the magnitude of the spectrogram of the comparand."""
	return torch.from_numpy(spectrogramMagnitudeBeta)

@pytest.fixture
def expectedContest(analyzer: CallableFunction[..., object], pathFilenamesContest: tuple[Path, Path]) -> float:
	"""Return the accepted recording of `analyzer` for the reference and the comparand at `pathFilenamesContest`."""
	lookupExpected: ChainMap[str, dict[tuple[str, str], float]] = ChainMap(
		expectedSpeakSoftly.expectedSpectrogram, expectedSpeakSoftly.expectedTensor, expectedSpeakSoftly.expectedTensorSpectrogram)
	return lookupExpected[analyzer.__name__][pathFilenamesContest[0].name, pathFilenamesContest[1].name]

@pytest.fixture
def seedRandomNumpy(request: pytest.FixtureRequest) -> Iterator[int]:
	"""Seed the global NumPy random generator with `request.param`, yield the seed, and then restore the generator's state."""
	stateRandom: dict[str, object] = numpy.random.get_state(legacy=False)  # ruff: ignore[numpy-legacy-random]
	numpy.random.seed(request.param)  # ruff: ignore[numpy-legacy-random]
	yield request.param
	numpy.random.set_state(stateRandom)  # ruff: ignore[numpy-legacy-random]
