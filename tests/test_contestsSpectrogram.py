from __future__ import annotations

from analyzeAudio.contestsSpectrogram import analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilenames, ContestSpectrogramMagnitudesSampleRates

def _standardizedEqualScalars(
	analyzer: str, pathFilenamesContest: ContestFilenames, actual: float, expected: float, sampleRate: int
) -> None:
	parameters = (
		f'pathFilenameAlfa={pathFilenamesContest.pathFilenameAlfa.name!r}, '
		f'pathFilenameBeta={pathFilenamesContest.pathFilenameBeta.name!r}, '
		f'sampleRate={sampleRate!r}'
	)
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8), (  # pyright: ignore[reportUnknownMemberType]
		f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	)

@pytest.mark.parametrize('expectedContestSpectrogram', ['analyzeBleedlessMelDBMean'], indirect=True)
def test_analyzeBleedlessMelDBMean(
	spectrogramMagnitudesContestSampleRates: ContestSpectrogramMagnitudesSampleRates, expectedContestSpectrogram: float
) -> None:
	actual = analyzeBleedlessMelDBMean(
		spectrogramMagnitudesContestSampleRates.spectrogramMagnitudeAlfa
		, spectrogramMagnitudesContestSampleRates.spectrogramMagnitudeBeta
		, sr=spectrogramMagnitudesContestSampleRates.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeBleedlessMelDBMean'
		, spectrogramMagnitudesContestSampleRates.pathFilenamesContest
		, actual
		, expectedContestSpectrogram
		, spectrogramMagnitudesContestSampleRates.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestSpectrogram', ['analyzeFullnessMelDBMean'], indirect=True)
def test_analyzeFullnessMelDBMean(
	spectrogramMagnitudesContestSampleRates: ContestSpectrogramMagnitudesSampleRates, expectedContestSpectrogram: float
) -> None:
	actual = analyzeFullnessMelDBMean(
		spectrogramMagnitudesContestSampleRates.spectrogramMagnitudeAlfa
		, spectrogramMagnitudesContestSampleRates.spectrogramMagnitudeBeta
		, sr=spectrogramMagnitudesContestSampleRates.sampleRateAlfa
	)
	_standardizedEqualScalars(
		'analyzeFullnessMelDBMean'
		, spectrogramMagnitudesContestSampleRates.pathFilenamesContest
		, actual
		, expectedContestSpectrogram
		, spectrogramMagnitudesContestSampleRates.sampleRateAlfa
	)
