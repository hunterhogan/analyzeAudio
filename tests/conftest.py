# ruff: noqa: DOC201
from __future__ import annotations

from analyzeAudio import settingsPackage
from hunterHearsPy import readAudioFile, stft
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'

listDataSamplesPathFilenames: tuple[Path, ...] = tuple(sorted(pathDataSamples.glob('*.wav')))

@pytest.fixture(params=listDataSamplesPathFilenames, ids=lambda pathFilename: pathFilename.name)
def pathFilename(request: pytest.FixtureRequest) -> Path:
	"""Return each static audio data-sample path."""
	pathFilename: Path = request.param
	return pathFilename
