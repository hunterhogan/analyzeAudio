from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import (
	analyzeComplexScaleInvariantSignalNoiseRatioLossMean, analyzeComplexScaleInvariantSignalNoiseRatioMean, analyzeL1FrequencyLoss,
	analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean)
from tests.conftest import assert_contest
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import ContestTensorSpectrogram, ContestTensorSpectrogramMagnitude

@pytest.mark.parametrize('expectedContest', ['analyzeComplexScaleInvariantSignalNoiseRatioMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioMean(contestTensorSpectrogram: ContestTensorSpectrogram, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioMean(contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeComplexScaleInvariantSignalNoiseRatioMean', contestTensorSpectrogram.paths, contestTensorSpectrogram.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeComplexScaleInvariantSignalNoiseRatioLossMean'], indirect=True)
def test_analyzeComplexScaleInvariantSignalNoiseRatioLossMean(contestTensorSpectrogram: ContestTensorSpectrogram, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeComplexScaleInvariantSignalNoiseRatioLossMean(contestTensorSpectrogram.tensorSpectrogramAlfa, contestTensorSpectrogram.tensorSpectrogramBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeComplexScaleInvariantSignalNoiseRatioLossMean', contestTensorSpectrogram.paths, contestTensorSpectrogram.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSpectralConvergenceLossMean'], indirect=True)
def test_analyzeSpectralConvergenceLossMean(contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectralConvergenceLossMean(contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSpectralConvergenceLossMean', contestTensorSpectrogramMagnitude.paths, contestTensorSpectrogramMagnitude.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTMagnitudeLossMean'], indirect=True)
def test_analyzeSTFTMagnitudeLossMean(contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSTFTMagnitudeLossMean(contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSTFTMagnitudeLossMean', contestTensorSpectrogramMagnitude.paths, contestTensorSpectrogramMagnitude.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeL1FrequencyLoss'], indirect=True)
def test_analyzeL1FrequencyLoss(contestTensorSpectrogramMagnitude: ContestTensorSpectrogramMagnitude, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeL1FrequencyLoss(contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeAlfa, contestTensorSpectrogramMagnitude.tensorSpectrogramMagnitudeBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeL1FrequencyLoss', contestTensorSpectrogramMagnitude.paths, contestTensorSpectrogramMagnitude.sampleRateAlfa)
