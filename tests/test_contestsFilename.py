from __future__ import annotations

from analyzeAudio.contestsFilename import (
	analyzePSNRmean, analyzePSNRmeanK, analyzeSDRmean, analyzeSDRmeanK, analyzeSI_SDRmean, analyzeSI_SDRmeanK)
from typing import TYPE_CHECKING
import os
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename

pytestmark: pytest.MarkDecorator = pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Skipped in GitHub Actions')

def _standardizedEqualScalars(
	analyzer: str, pathFilenamesContest: ContestFilename, actual: float, expected: float, K: float | None = None
) -> None:
	parameters = (
		f'pathFilenameAlfa={pathFilenamesContest.pathFilenameAlfa.name!r}, pathFilenameBeta={pathFilenamesContest.pathFilenameBeta.name!r}'
	)
	if K is not None:
		parameters = f'{parameters}, K={K!r}'
	assert actual == pytest.approx(expected), (  # pyright: ignore[reportUnknownMemberType]
		f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	)

@pytest.mark.parametrize('expectedContestFilename', ['analyzePSNRmean'], indirect=True)
def test_analyzePSNRmean(pathFilenamesContest: ContestFilename, expectedContestFilename: float) -> None:
	actual = analyzePSNRmean(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta)
	_standardizedEqualScalars('analyzePSNRmean', pathFilenamesContest, actual, expectedContestFilename)

@pytest.mark.parametrize('expectedContestFilename', ['analyzeSDRmean'], indirect=True)
def test_analyzeSDRmean(pathFilenamesContest: ContestFilename, expectedContestFilename: float) -> None:
	actual = analyzeSDRmean(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta)
	_standardizedEqualScalars('analyzeSDRmean', pathFilenamesContest, actual, expectedContestFilename)

@pytest.mark.parametrize('expectedContestFilename', ['analyzeSI_SDRmean'], indirect=True)
def test_analyzeSI_SDRmean(pathFilenamesContest: ContestFilename, expectedContestFilename: float) -> None:
	actual = analyzeSI_SDRmean(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta)
	_standardizedEqualScalars('analyzeSI_SDRmean', pathFilenamesContest, actual, expectedContestFilename)

@pytest.mark.parametrize('K', [10.0])
@pytest.mark.parametrize('expectedContestFilename', ['analyzePSNRmeanK'], indirect=True)
def test_analyzePSNRmeanK(pathFilenamesContest: ContestFilename, K: float, expectedContestFilename: float) -> None:
	actual = analyzePSNRmeanK(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta, K)
	_standardizedEqualScalars('analyzePSNRmeanK', pathFilenamesContest, actual, expectedContestFilename, K)

@pytest.mark.parametrize('K', [10.0])
@pytest.mark.parametrize('expectedContestFilename', ['analyzeSDRmeanK'], indirect=True)
def test_analyzeSDRmeanK(pathFilenamesContest: ContestFilename, K: float, expectedContestFilename: float) -> None:
	actual = analyzeSDRmeanK(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta, K)
	_standardizedEqualScalars('analyzeSDRmeanK', pathFilenamesContest, actual, expectedContestFilename, K)

@pytest.mark.parametrize('K', [10.0])
@pytest.mark.parametrize('expectedContestFilename', ['analyzeSI_SDRmeanK'], indirect=True)
def test_analyzeSI_SDRmeanK(pathFilenamesContest: ContestFilename, K: float, expectedContestFilename: float) -> None:
	actual = analyzeSI_SDRmeanK(pathFilenamesContest.pathFilenameAlfa, pathFilenamesContest.pathFilenameBeta, K)
	_standardizedEqualScalars('analyzeSI_SDRmeanK', pathFilenamesContest, actual, expectedContestFilename, K)
