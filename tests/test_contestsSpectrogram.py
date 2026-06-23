from __future__ import annotations

from analyzeAudio.contestsSpectrogram import analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean
from tests.conftest import assert_contest
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestSpectrogramMagnitude

@pytest.mark.parametrize('expectedContest', ['analyzeBleedlessMelDBMean'], indirect=True)
def test_analyzeBleedlessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeBleedlessMelDBMean(contestSpectrogramMagnitude.spectrogramMagnitudeAlfa, contestSpectrogramMagnitude.spectrogramMagnitudeBeta, sr=contestSpectrogramMagnitude.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeBleedlessMelDBMean', contestSpectrogramMagnitude.paths, contestSpectrogramMagnitude.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeFullnessMelDBMean'], indirect=True)
def test_analyzeFullnessMelDBMean(contestSpectrogramMagnitude: ContestSpectrogramMagnitude, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeFullnessMelDBMean(contestSpectrogramMagnitude.spectrogramMagnitudeAlfa, contestSpectrogramMagnitude.spectrogramMagnitudeBeta, sr=contestSpectrogramMagnitude.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeFullnessMelDBMean', contestSpectrogramMagnitude.paths, contestSpectrogramMagnitude.sampleRateAlfa)
