# ruff: noqa: DOC201
from __future__ import annotations

from hunterHearsPy import parametersDEFAULT, readAudioFile, stft
from tests import (
	ContestFilenames, ContestSpectrogramMagnitudesSampleRates, ContestSpectrogramsSampleRates, ContestWaveformsSampleRates,
	listPathFilenamesContests, listPathFilenamesDataSamples, randomSeed, SpectrogramMagnitudeSampleRate, SpectrogramPowerSampleRate,
	SpectrogramSampleRate, TensorSampleRate, WaveformSampleRate)
from tests.dataSamples.SpeakSoftly_BrokenMan60sec.expected import (
	expectedFilename as dictionaryExpectedContestFilename, expectedSpectrogram as dictionaryExpectedContestSpectrogram)
from typing import TYPE_CHECKING
import librosa
import numpy
import pytest
import soundfile

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from hunterHearsPy import Spectrogram, Waveform
	from pathlib import Path

# TODO abstract the expected values dictionaries into a fixture.
# This `from tests.dataSamples.expected import expectedFilename` is not scalable.
# Right now, the expected values are effectively indexed by keys: tests.dataSamples.expected,
# expectedFilename, analyzeAbs_Peak_countTotal, pathFilename.
# Therefore, abstract the keys to those that actually matter--analyzeAbs_Peak_countTotal and
# pathFilename, the function being tested and the test parameters--and have the test function get
# the value from a fixture.
# The fixture will track the physical location of the expected values.

@pytest.fixture(params=listPathFilenamesDataSamples, ids=lambda pathFilename: pathFilename.name, scope='session')
def pathFilename(request: pytest.FixtureRequest) -> Path:
	"""Return each static audio data-sample path."""
	pathFilename: Path = request.param
	return pathFilename

@pytest.fixture(scope='session')
def waveform_sampleRate(pathFilename: Path) -> WaveformSampleRate:
	"""Return each static waveform with its source path and sample rate."""
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform: Audio = readSoundFile.read(dtype='float32').astype(numpy.float32)
	waveform = waveform.T
	return WaveformSampleRate(pathFilename, waveform, sampleRate)

@pytest.fixture(scope='session')
def spectrogram_sampleRate(waveform_sampleRate: WaveformSampleRate) -> SpectrogramSampleRate:
	"""Return each static spectrogram with its source path and sample rate."""
	spectrogram: Spectrogram = librosa.stft(waveform_sampleRate.waveform)
	return SpectrogramSampleRate(waveform_sampleRate.pathFilename, spectrogram, waveform_sampleRate.sampleRate)

@pytest.fixture(scope='session')
def spectrogramMagnitude_sampleRate(spectrogram_sampleRate: SpectrogramSampleRate) -> SpectrogramMagnitudeSampleRate:
	"""Return each static spectrogram magnitude with its source path and sample rate."""
	spectrogramMagnitude: SpectrogramMagnitude = numpy.absolute(spectrogram_sampleRate.spectrogram)
	return SpectrogramMagnitudeSampleRate(spectrogram_sampleRate.pathFilename, spectrogramMagnitude, spectrogram_sampleRate.sampleRate)

@pytest.fixture(scope='session')
def spectrogramPower_sampleRate(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate) -> SpectrogramPowerSampleRate:
	"""Return each static spectrogram power with its source path and sample rate."""
	spectrogramPower: SpectrogramPower = spectrogramMagnitude_sampleRate.spectrogramMagnitude**2
	return SpectrogramPowerSampleRate(
		spectrogramMagnitude_sampleRate.pathFilename, spectrogramPower, spectrogramMagnitude_sampleRate.sampleRate
	)

def _idContestFilenames(pathFilenamesContest: ContestFilenames) -> str:
	return f'{pathFilenamesContest.pathFilenameAlfa.stem}--{pathFilenamesContest.pathFilenameBeta.stem}'

@pytest.fixture(params=listPathFilenamesContests, ids=_idContestFilenames, scope='session')
def pathFilenamesContest(request: pytest.FixtureRequest) -> ContestFilenames:
	"""Return each matching reference and comparand audio-file pair."""
	pathFilenamesContest: ContestFilenames = request.param
	return pathFilenamesContest

@pytest.fixture(scope='session')
def waveformsContestSampleRates(pathFilenamesContest: ContestFilenames) -> ContestWaveformsSampleRates:
	"""Return each contest waveform pair with its sample rates."""
	sampleRate: int = int(parametersDEFAULT['sampleRate'])
	waveformAlfa: Waveform = readAudioFile(pathFilenamesContest.pathFilenameAlfa, sampleRate=sampleRate)
	waveformBeta: Waveform = readAudioFile(pathFilenamesContest.pathFilenameBeta, sampleRate=sampleRate)
	return ContestWaveformsSampleRates(pathFilenamesContest, waveformAlfa, sampleRate, waveformBeta, sampleRate)

@pytest.fixture(scope='session')
def spectrogramsContestSampleRates(waveformsContestSampleRates: ContestWaveformsSampleRates) -> ContestSpectrogramsSampleRates:
	"""Return each contest complex-valued spectrogram pair with its sample rates."""
	spectrogramAlfa: Spectrogram = stft(waveformsContestSampleRates.waveformAlfa, sampleRate=waveformsContestSampleRates.sampleRateAlfa)
	spectrogramBeta: Spectrogram = stft(waveformsContestSampleRates.waveformBeta, sampleRate=waveformsContestSampleRates.sampleRateBeta)
	return ContestSpectrogramsSampleRates(
		waveformsContestSampleRates.pathFilenamesContest
		, spectrogramAlfa
		, waveformsContestSampleRates.sampleRateAlfa
		, spectrogramBeta
		, waveformsContestSampleRates.sampleRateBeta
	)

@pytest.fixture(scope='session')
def spectrogramMagnitudesContestSampleRates(
	spectrogramsContestSampleRates: ContestSpectrogramsSampleRates,
) -> ContestSpectrogramMagnitudesSampleRates:
	"""Return each contest magnitude spectrogram pair with its sample rates."""
	spectrogramMagnitudeAlfa: SpectrogramMagnitude = numpy.absolute(spectrogramsContestSampleRates.spectrogramAlfa)
	spectrogramMagnitudeBeta: SpectrogramMagnitude = numpy.absolute(spectrogramsContestSampleRates.spectrogramBeta)
	return ContestSpectrogramMagnitudesSampleRates(
		spectrogramsContestSampleRates.pathFilenamesContest
		, spectrogramMagnitudeAlfa
		, spectrogramsContestSampleRates.sampleRateAlfa
		, spectrogramMagnitudeBeta
		, spectrogramsContestSampleRates.sampleRateBeta
	)

@pytest.fixture(scope='session')
def expectedContestFilename(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilenames) -> float:
	"""Return the stored expected value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return dictionaryExpectedContestFilename[analyzer][pairFilenames]

@pytest.fixture(scope='session')
def expectedContestSpectrogram(request: pytest.FixtureRequest, pathFilenamesContest: ContestFilenames) -> float:
	"""Return the stored expected spectrogram value for the current contest function and path pair."""
	analyzer: str = request.param
	pairFilenames = (pathFilenamesContest.pathFilenameAlfa.name, pathFilenamesContest.pathFilenameBeta.name)
	return dictionaryExpectedContestSpectrogram[analyzer][pairFilenames]
