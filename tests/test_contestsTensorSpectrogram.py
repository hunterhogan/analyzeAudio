from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import (
	analyzeComplexScaleInvariantSignalNoiseRatioLossMean, analyzeComplexScaleInvariantSignalNoiseRatioMean, analyzeL1FrequencyLoss,
	analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestTensorSpectrogram, ContestTensorSpectrogramMagnitude

def _standardizedEqualScalars(analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int) -> None:
	parameters: str = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	message: str = f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6, nan_ok=True), message  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedContest', ['analyzeComplexScaleInvariantSignalNoiseRatioMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioMean(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContest: float
) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioMean(
		contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta
	)
	_standardizedEqualScalars(
		'analyzeComplexScaleInvariantSignalNoiseRatioMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContest
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeComplexScaleInvariantSignalNoiseRatioLossMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioLossMean(
	contestTensorSpectrogram: ContestTensorSpectrogram, expectedContest: float
) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioLossMean(
		contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta
	)
	_standardizedEqualScalars(
		'analyzeComplexScaleInvariantSignalNoiseRatioLossMean'
		, contestTensorSpectrogram.paths
		, actual
		, expectedContest
		, contestTensorSpectrogram.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeSpectralConvergenceLossMean'], indirect=True)
def test_analyzeSpectralConvergenceLossMean(
	contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float
) -> None:
	actual = analyzeSpectralConvergenceLossMean(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSpectralConvergenceLossMean'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContest
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTMagnitudeLossMean'], indirect=True)
def test_analyzeSTFTMagnitudeLossMean(contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float) -> None:
	actual = analyzeSTFTMagnitudeLossMean(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeSTFTMagnitudeLossMean'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContest
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeL1FrequencyLoss'], indirect=True)
def test_analyzeL1FrequencyLoss(contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float) -> None:
	actual = analyzeL1FrequencyLoss(
		contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta
	)
	_standardizedEqualScalars(
		'analyzeL1FrequencyLoss'
		, contestTensorSpectrogramMagnitude.paths
		, actual
		, expectedContest
		, contestTensorSpectrogramMagnitude.sampleRateAlfa
	)
