# ruff: noqa: DOC201
from __future__ import annotations

from analyzeAudio import Audio, settingsPackage
from hunterHearsPy import stft
from typing import NamedTuple, TYPE_CHECKING
import numpy
import pytest
import soundfile

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'

listDataSamplesPathFilenames: tuple[Path, ...] = tuple(sorted(pathDataSamples.glob('*.wav')))

class WaveformSampleRate(NamedTuple):
	pathFilename: Path
	waveform: Audio
	sampleRate: int

@pytest.fixture(params=listDataSamplesPathFilenames, ids=lambda pathFilename: pathFilename.name)
def pathFilename(request: pytest.FixtureRequest) -> Path:
	"""Return each static audio data-sample path."""
	pathFilename: Path = request.param
	return pathFilename

@pytest.fixture
def waveform_sampleRate(pathFilename: Path) -> WaveformSampleRate:
	"""Return each static waveform with its source path and sample rate."""
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform: Audio = readSoundFile.read(dtype='float32').astype(numpy.float32)
	waveform = waveform.T
	return WaveformSampleRate(pathFilename, waveform, sampleRate)
