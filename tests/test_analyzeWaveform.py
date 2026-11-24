"""Expected values are the accepted recordings in `tests.dataSamples.expected.expectedWaveform`."""
from __future__ import annotations

from analyzeAudio.analyzeWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean,
	analyzeZeroCrossingsTotal)
from tests import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from analyzeAudio import Audio
	from hunterMakesPy import CallableFunction
	from pathlib import Path

@pytest.mark.parametrize('analyzer', (analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeZeroCrossingRateMean, analyzeZeroCrossingsTotal))
def test_aspectWaveform(analyzer: CallableFunction[[Audio], float], pathFilename: Path, waveform: Audio, expectedAspect: float) -> None:
	assert_approx(analyzer(waveform), expectedAspect, 1e-6, 1e-12, analyzer.__name__, pathFilename.name)

@pytest.mark.parametrize('analyzer, pytest_rel', (
	pytest.param(analyzeTempogramMean, 1e-4, id='analyzeTempogramMean'),
	pytest.param(analyzeTempoMean, 1e-6, id='analyzeTempoMean'),
))
def test_aspectWaveformSampleRate(analyzer: CallableFunction[[Audio, int], float], pytest_rel: float, pathFilename: Path, waveform: Audio, sampleRate: int, expectedAspect: float) -> None:
	"""`analyzeTempogramMean` keeps the relative tolerance 1e-4 with which the maintainer accepted its recordings."""
	assert_approx(analyzer(waveform, sampleRate), expectedAspect, pytest_rel, 1e-12, analyzer.__name__, pathFilename.name, sampleRate)
