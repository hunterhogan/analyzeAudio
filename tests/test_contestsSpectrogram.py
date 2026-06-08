from __future__ import annotations

from analyzeAudio.contestsSpectrogram import analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestSpectrogramMagnitude

def _standardizedEqualScalars(
	analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int
) -> None:
	parameters = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, '
		f'pathFilenameBeta={paths.pathFilenameBeta.name!r}, '
		f'sampleRate={sampleRate!r}'
	)
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8), (  # pyright: ignore[reportUnknownMemberType]
		f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	)

@pytest.mark.parametrize('expectedContestSpectrogram', ['analyzeBleedlessMelDBMean'], indirect=True)
def test_analyzeBleedlessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContestSpectrogram: float) -> None:
	actual = analyzeBleedlessMelDBMean(
		contestSpectrogramMagnitude.spectrogramMagnitudeAlfa
		, contestSpectrogramMagnitude.spectrogramMagnitudeBeta
		, sr=contestSpectrogramMagnitude.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeBleedlessMelDBMean'
		, contestSpectrogramMagnitude.paths
		, actual
		, expectedContestSpectrogram
		, contestSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestSpectrogram', ['analyzeFullnessMelDBMean'], indirect=True)
def test_analyzeFullnessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContestSpectrogram: float) -> None:
	actual = analyzeFullnessMelDBMean(
		contestSpectrogramMagnitude.spectrogramMagnitudeAlfa
		, contestSpectrogramMagnitude.spectrogramMagnitudeBeta
		, sr=contestSpectrogramMagnitude.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeFullnessMelDBMean'
		, contestSpectrogramMagnitude.paths
		, actual
		, expectedContestSpectrogram
		, contestSpectrogramMagnitude.sampleRateAlfa
	)
