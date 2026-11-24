"""Expected values are the accepted recordings in `tests.dataSamples.SpeakSoftly_BrokenMan60sec.expected.expectedTensorSpectrogram`."""
from __future__ import annotations

from analyzeAudio.contestsTensorSpectrogram import (
	analyzeComplexScaleInvariantSignalNoiseRatioLossMean, analyzeComplexScaleInvariantSignalNoiseRatioMean, analyzeL1FrequencyLoss,
	analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean)
from tests import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from pathlib import Path
	from torch import Tensor

@pytest.mark.parametrize('analyzer', (analyzeComplexScaleInvariantSignalNoiseRatioLossMean, analyzeComplexScaleInvariantSignalNoiseRatioMean))
def test_contestTensorSpectrogram(analyzer: CallableFunction[[Tensor, Tensor], float], pathFilenamesContest: tuple[Path, Path], tensorSpectrogramAlfa: Tensor
	, tensorSpectrogramBeta: Tensor, expectedContest: float) -> None:
	assert_approx(analyzer(tensorSpectrogramAlfa, tensorSpectrogramBeta), expectedContest, 1e-6, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name)

@pytest.mark.parametrize('analyzer', (analyzeL1FrequencyLoss, analyzeSpectralConvergenceLossMean, analyzeSTFTMagnitudeLossMean))
def test_contestTensorSpectrogramMagnitude(analyzer: CallableFunction[[Tensor, Tensor], float], pathFilenamesContest: tuple[Path, Path], tensorSpectrogramMagnitudeAlfa: Tensor
	, tensorSpectrogramMagnitudeBeta: Tensor, expectedContest: float) -> None:
	assert_approx(analyzer(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta), expectedContest, 1e-6, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name)
