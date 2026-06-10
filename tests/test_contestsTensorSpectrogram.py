from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import (
	analyzeComplexScaleInvariantSignalNoiseRatioLossMean, analyzeComplexScaleInvariantSignalNoiseRatioMean, analyzeL1FrequencyLoss,
	analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestTensorSpectrogram, ContestTensorSpectrogramMagnitude

def _standardizedEqualScalars(analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int) -> None:
	parameters = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	message = f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8, nan_ok=True), message  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeComplexScaleInvariantSignalNoiseRatioMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioMean(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioMean(
		contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta
	)
	_standardizedEqualScalars(
		'analyzeComplexScaleInvariantSignalNoiseRatioMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeComplexScaleInvariantSignalNoiseRatioLossMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioLossMean(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioLossMean(
		contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta
	)
	_standardizedEqualScalars(
		'analyzeComplexScaleInvariantSignalNoiseRatioLossMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSpectralConvergenceLossMean'], indirect=True)
def test_analyzeSpectralConvergenceLossMean(
	contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeSpectralConvergenceLossMean(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSpectralConvergenceLossMean'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeSTFTMagnitudeLossMean'], indirect=True)
def test_analyzeSTFTMagnitudeLossMean(
	contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeSTFTMagnitudeLossMean(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSTFTMagnitudeLossMean'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensorSpectrogram', ['analyzeL1FrequencyLoss'], indirect=True)
def test_analyzeL1FrequencyLoss(
	contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContestTensorSpectrogram: float
) -> None:
	actual = analyzeL1FrequencyLoss(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeL1FrequencyLoss'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContestTensorSpectrogram
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)
