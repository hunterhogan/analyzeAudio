# ruff: noqa: DOC201
from __future__ import annotations

from analyzeAudio import Audio, settingsPackage, Spectrogram, SpectrogramMagnitude, SpectrogramPower
from tests import SpectrogramMagnitudeSampleRate, SpectrogramPowerSampleRate, SpectrogramSampleRate, WaveformSampleRate
from typing import TYPE_CHECKING
import librosa
import numpy
import pytest
import soundfile

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'

listDataSamplesPathFilenames: tuple[Path, ...] = tuple(sorted(pathDataSamples.glob('*.wav')))

@pytest.fixture(params=listDataSamplesPathFilenames, ids=lambda pathFilename: pathFilename.name, scope='session')
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
