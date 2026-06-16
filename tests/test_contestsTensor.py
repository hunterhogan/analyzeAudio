from __future__ import annotations

from analyzeAudio.contestsTensor import (
	analyzeChromaSTFTLossMean, analyzeDCLossMean, analyzeESRLossMean, analyzeL1SNRDBMean, analyzeL1SNRMean, analyzeLogCoshLossMean,
	analyzeLogWMSEMean, analyzeMelSTFTLossMean, analyzeMultiL1SNRDBMean, analyzeMultiResolutionSTFTLossMean,
	analyzePerceptualEvaluationSpeechQualityMean, analyzePermutationInvariantTrainingMean, analyzeRandomResolutionSTFTLossMean,
	analyzeScaleInvariantSignalDistortionRatioMean, analyzeScaleInvariantSignalNoiseRatioMean, analyzeSDSDRLossMean,
	analyzeShortTimeObjectiveIntelligibilityMean, analyzeSignalDistortionRatioMean, analyzeSignalNoiseRatioMean, analyzeSISDRLossMean,
	analyzeSNRLossMean, analyzeSourceAggregatedSignalDistortionRatioMean, analyzeSTFTL1SNRDBMean, analyzeSTFTLossMean,
	analyzeSumAndDifferenceSTFTLossMean)
from tests.conftest import assert_contest
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from tests import ContestTensor
	from torch import Tensor

@pytest.mark.parametrize('expectedContest', ['analyzeLogWMSEMean'], indirect=True)
def test_analyzeLogWMSEMean(contestTensor: ContestTensor, tensorAudioMixture: Tensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLogWMSEMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, tensorAudioMixture, contestTensor.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeLogWMSEMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeL1SNRMean'], indirect=True)
def test_analyzeL1SNRMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeL1SNRMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeL1SNRMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeL1SNRDBMean'], indirect=True)
def test_analyzeL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	actual = analyzeL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeL1SNRDBMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMultiL1SNRDBMean'], indirect=True)
def test_analyzeMultiL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	actual = analyzeMultiL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeMultiL1SNRDBMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTL1SNRDBMean'], indirect=True)
def test_analyzeSTFTL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	actual = analyzeSTFTL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSTFTL1SNRDBMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzePerceptualEvaluationSpeechQualityMean'], indirect=True)
def test_analyzePerceptualEvaluationSpeechQualityMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzePerceptualEvaluationSpeechQualityMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzePerceptualEvaluationSpeechQualityMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeShortTimeObjectiveIntelligibilityMean'], indirect=True)
def test_analyzeShortTimeObjectiveIntelligibilityMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeShortTimeObjectiveIntelligibilityMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeShortTimeObjectiveIntelligibilityMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSignalNoiseRatioMean'], indirect=True)
def test_analyzeSignalNoiseRatioMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSignalNoiseRatioMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSignalNoiseRatioMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeScaleInvariantSignalNoiseRatioMean'], indirect=True)
def test_analyzeScaleInvariantSignalNoiseRatioMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeScaleInvariantSignalNoiseRatioMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeScaleInvariantSignalNoiseRatioMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeScaleInvariantSignalDistortionRatioMean'], indirect=True)
def test_analyzeScaleInvariantSignalDistortionRatioMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeScaleInvariantSignalDistortionRatioMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeScaleInvariantSignalDistortionRatioMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSignalDistortionRatioMean'], indirect=True)
def test_analyzeSignalDistortionRatioMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSignalDistortionRatioMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSignalDistortionRatioMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSourceAggregatedSignalDistortionRatioMean'], indirect=True)
def test_analyzeSourceAggregatedSignalDistortionRatioMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSourceAggregatedSignalDistortionRatioMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSourceAggregatedSignalDistortionRatioMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzePermutationInvariantTrainingMean'], indirect=True)
def test_analyzePermutationInvariantTrainingMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzePermutationInvariantTrainingMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzePermutationInvariantTrainingMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeDCLossMean'], indirect=True)
def test_analyzeDCLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeDCLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeDCLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeESRLossMean'], indirect=True)
def test_analyzeESRLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeESRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeESRLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeLogCoshLossMean'], indirect=True)
def test_analyzeLogCoshLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLogCoshLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeLogCoshLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSNRLossMean'], indirect=True)
def test_analyzeSNRLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSNRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSNRLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSISDRLossMean'], indirect=True)
def test_analyzeSISDRLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSISDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSISDRLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSDSDRLossMean'], indirect=True)
def test_analyzeSDSDRLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSDSDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSDSDRLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTLossMean'], indirect=True)
def test_analyzeSTFTLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMelSTFTLossMean'], indirect=True)
def test_analyzeMelSTFTLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMelSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeMelSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeChromaSTFTLossMean'], indirect=True)
def test_analyzeChromaSTFTLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeChromaSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeChromaSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMultiResolutionSTFTLossMean'], indirect=True)
def test_analyzeMultiResolutionSTFTLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMultiResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeMultiResolutionSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('randomSeed', [1597])
@pytest.mark.parametrize('expectedContest', ['analyzeRandomResolutionSTFTLossMean'], indirect=True)
def test_analyzeRandomResolutionSTFTLossMean(contestTensor: ContestTensor, randomSeed: int, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	numpy.random.seed(randomSeed)  # noqa: NPY002
	actual = analyzeRandomResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeRandomResolutionSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSumAndDifferenceSTFTLossMean'], indirect=True)
def test_analyzeSumAndDifferenceSTFTLossMean(contestTensor: ContestTensor, expectedContest: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSumAndDifferenceSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	assert_contest(actual, expectedContest, approx_rel, approx_abs, 'analyzeSumAndDifferenceSTFTLossMean', contestTensor.paths, contestTensor.sampleRateAlfa)
