"""Expected values are the accepted recordings in `tests.dataSamples.SpeakSoftly_BrokenMan60sec.expected.expectedTensor`."""
from __future__ import annotations

from analyzeAudio.contestsTensor import (
	analyzeChromaSTFTLossMean, analyzeDCLossMean, analyzeESRLossMean, analyzeL1SNRDBMean, analyzeL1SNRMean, analyzeLogCoshLossMean,
	analyzeLogWMSEMean, analyzeMelSTFTLossMean, analyzeMultiL1SNRDBMean, analyzeMultiResolutionSTFTLossMean,
	analyzePerceptualEvaluationSpeechQualityMean, analyzePermutationInvariantTrainingMean, analyzeRandomResolutionSTFTLossMean,
	analyzeScaleInvariantSignalDistortionRatioMean, analyzeScaleInvariantSignalNoiseRatioMean, analyzeSDSDRLossMean,
	analyzeShortTimeObjectiveIntelligibilityMean, analyzeSignalDistortionRatioMean, analyzeSignalNoiseRatioMean, analyzeSISDRLossMean,
	analyzeSNRLossMean, analyzeSourceAggregatedSignalDistortionRatioMean, analyzeSTFTL1SNRDBMean, analyzeSTFTLossMean,
	analyzeSumAndDifferenceSTFTLossMean)
from tests import assert_approx
from tests.dataSamples.audioFiles import sampleRateSpeakSoftly
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from pathlib import Path
	from torch import Tensor

@pytest.mark.parametrize('analyzer, pytest_rel', (
	pytest.param(analyzeDCLossMean, 1e-6, id='analyzeDCLossMean'),
	pytest.param(analyzeESRLossMean, 1e-6, id='analyzeESRLossMean'),
	pytest.param(analyzeL1SNRDBMean, 1e-5, id='analyzeL1SNRDBMean'),
	pytest.param(analyzeL1SNRMean, 1e-6, id='analyzeL1SNRMean'),
	pytest.param(analyzeLogCoshLossMean, 1e-6, id='analyzeLogCoshLossMean'),
	pytest.param(analyzeMultiL1SNRDBMean, 1e-5, id='analyzeMultiL1SNRDBMean'),
	pytest.param(analyzeMultiResolutionSTFTLossMean, 1e-6, id='analyzeMultiResolutionSTFTLossMean'),
	pytest.param(analyzePermutationInvariantTrainingMean, 1e-6, id='analyzePermutationInvariantTrainingMean'),
	pytest.param(analyzeScaleInvariantSignalDistortionRatioMean, 1e-6, id='analyzeScaleInvariantSignalDistortionRatioMean'),
	pytest.param(analyzeScaleInvariantSignalNoiseRatioMean, 1e-6, id='analyzeScaleInvariantSignalNoiseRatioMean'),
	pytest.param(analyzeSDSDRLossMean, 1e-6, id='analyzeSDSDRLossMean'),
	pytest.param(analyzeSignalDistortionRatioMean, 1e-6, id='analyzeSignalDistortionRatioMean'),
	pytest.param(analyzeSignalNoiseRatioMean, 1e-6, id='analyzeSignalNoiseRatioMean'),
	pytest.param(analyzeSISDRLossMean, 1e-6, id='analyzeSISDRLossMean'),
	pytest.param(analyzeSNRLossMean, 1e-6, id='analyzeSNRLossMean'),
	pytest.param(analyzeSourceAggregatedSignalDistortionRatioMean, 1e-6, id='analyzeSourceAggregatedSignalDistortionRatioMean'),
	pytest.param(analyzeSTFTL1SNRDBMean, 1e-5, id='analyzeSTFTL1SNRDBMean'),
	pytest.param(analyzeSTFTLossMean, 1e-6, id='analyzeSTFTLossMean'),
	pytest.param(analyzeSumAndDifferenceSTFTLossMean, 1e-6, id='analyzeSumAndDifferenceSTFTLossMean'),
))
def test_contestTensor(analyzer: CallableFunction[[Tensor, Tensor], float], pytest_rel: float, pathFilenamesContest: tuple[Path, Path], tensorAudioAlfa: Tensor
	, tensorAudioBeta: Tensor, expectedContest: float) -> None:
	"""The three L1 SNR dB analyzers keep the relative tolerance 1e-5 with which the maintainer accepted their recordings. On 2026-11-24,
	Linux on x86-64 differed from those recordings by up to 1.7e-6.
	"""
	assert_approx(analyzer(tensorAudioAlfa, tensorAudioBeta), expectedContest, pytest_rel, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name)

@pytest.mark.parametrize('analyzer, pytest_rel', (
	pytest.param(analyzeChromaSTFTLossMean, 1e-6, id='analyzeChromaSTFTLossMean'),
	pytest.param(analyzeMelSTFTLossMean, 1e-6, id='analyzeMelSTFTLossMean'),
	pytest.param(analyzePerceptualEvaluationSpeechQualityMean, 1e-2, id='analyzePerceptualEvaluationSpeechQualityMean'),
	pytest.param(analyzeShortTimeObjectiveIntelligibilityMean, 1e-6, id='analyzeShortTimeObjectiveIntelligibilityMean'),
))
def test_contestTensorSampleRate(analyzer: CallableFunction[[Tensor, Tensor, int], float], pytest_rel: float, pathFilenamesContest: tuple[Path, Path], tensorAudioAlfa: Tensor
	, tensorAudioBeta: Tensor, expectedContest: float) -> None:
	"""`analyzePerceptualEvaluationSpeechQualityMean` has the relative tolerance 1e-2. The `pesq` package compiles its C source on each
	machine. On 2026-11-24, the GitHub Actions Ubuntu runners differed from the recordings by up to 6.1e-3, at `drumsBad`, and a Linux
	container on x86-64 reproduced every recording exactly.
	"""
	assert_approx(analyzer(tensorAudioAlfa, tensorAudioBeta, sampleRateSpeakSoftly), expectedContest, pytest_rel, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name, sampleRateSpeakSoftly)

@pytest.mark.parametrize('analyzer', (analyzeLogWMSEMean,))
def test_contestTensorMixture(analyzer: CallableFunction[[Tensor, Tensor, Tensor, int], float], pathFilenamesContest: tuple[Path, Path], tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor
	, tensorAudioMixture: Tensor, expectedContest: float) -> None:
	"""The recordings were accepted with torch_log_wmse 0.3, which filtered with a transform of a fixed number of whole seconds; from
	torch_log_wmse 1.0, the length of each call sets the transform size. On 2026-11-24, torch_log_wmse 1.1.0 reproduced each recording
	within a relative difference of 7.3e-6.
	"""
	assert_approx(analyzer(tensorAudioAlfa, tensorAudioBeta, tensorAudioMixture, sampleRateSpeakSoftly), expectedContest, 1e-5, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name, 'mixture', sampleRateSpeakSoftly)

@pytest.mark.parametrize('seedRandomNumpy', (1597,), indirect=True)
@pytest.mark.parametrize('analyzer', (analyzeRandomResolutionSTFTLossMean,))
def test_contestTensorRandom(analyzer: CallableFunction[[Tensor, Tensor], float], seedRandomNumpy: int, pathFilenamesContest: tuple[Path, Path], tensorAudioAlfa: Tensor
	, tensorAudioBeta: Tensor, expectedContest: float) -> None:
	"""The global NumPy random generator chooses the resolutions, so each recording depends on the seed."""
	assert_approx(analyzer(tensorAudioAlfa, tensorAudioBeta), expectedContest, 1e-6, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name, seedRandomNumpy=seedRandomNumpy)
