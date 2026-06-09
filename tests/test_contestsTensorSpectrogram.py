from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import analyzeL1FrequencyLoss, analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestTensorSpectrogram

def _standardizedEqualScalars(analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int) -> None:
	parameters = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	message = f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8, nan_ok=True), message  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSpectralConvergenceLossMean'], indirect=True)
def test_analyzeSpectralConvergenceLossMean(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeSpectralConvergenceLossMean(
		contestTensorSpectrogram.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogram.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSpectralConvergenceLossMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSTFTMagnitudeLossMean'], indirect=True)
def test_analyzeSTFTMagnitudeLossMean(contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float) -> None:
	actual = analyzeSTFTMagnitudeLossMean(
		contestTensorSpectrogram.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogram.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSTFTMagnitudeLossMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeL1FrequencyLoss'], indirect=True)
def test_analyzeL1FrequencyLoss(contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float) -> None:
	actual = analyzeL1FrequencyLoss(
		contestTensorSpectrogram.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogram.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeL1FrequencyLoss'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)
