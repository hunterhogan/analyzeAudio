from __future__ import annotations

from analyzeAudio.contestsSpectrogram import analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestSpectrogramMagnitude

def _standardizedEqualScalars(analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int) -> None:
	parameters: str = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), (  # pyright: ignore[reportUnknownMemberType]
		f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	)

@pytest.mark.parametrize('expectedContest', ['analyzeBleedlessMelDBMean'], indirect=True)
def test_analyzeBleedlessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContest: float) -> None:
	actual = analyzeBleedlessMelDBMean(
		contestSpectrogramMagnitude.spectrogramMagnitudeAlfa
		, contestSpectrogramMagnitude.spectrogramMagnitudeBeta
		, sr=contestSpectrogramMagnitude.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeBleedlessMelDBMean', contestSpectrogramMagnitude.paths, actual, expectedContest, contestSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeFullnessMelDBMean'], indirect=True)
def test_analyzeFullnessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContest: float) -> None:
	actual = analyzeFullnessMelDBMean(
		contestSpectrogramMagnitude.spectrogramMagnitudeAlfa
		, contestSpectrogramMagnitude.spectrogramMagnitudeBeta
		, sr=contestSpectrogramMagnitude.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeFullnessMelDBMean', contestSpectrogramMagnitude.paths, actual, expectedContest, contestSpectrogramMagnitude.sampleRateAlfa
	)
