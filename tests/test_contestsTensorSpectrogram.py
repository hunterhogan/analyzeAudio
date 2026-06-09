from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import analyzeL1FrequencyLoss, analyzeSpectralConvergenceLoss, analyzeSTFTMagnitudeLoss
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

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSpectralConvergenceLoss'], indirect=True)
def test_analyzeSpectralConvergenceLoss(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeSpectralConvergenceLoss(
		contestTensorSpectrogram.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogram.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSpectralConvergenceLoss'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSTFTMagnitudeLoss'], indirect=True)
def test_analyzeSTFTMagnitudeLoss(contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float) -> None:
	actual = analyzeSTFTMagnitudeLoss(
		contestTensorSpectrogram.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogram.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSTFTMagnitudeLoss'
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
