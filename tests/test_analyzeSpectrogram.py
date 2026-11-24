"""Expected values are the accepted recordings in `tests.dataSamples.expected.expectedSpectrogram`.

No recording exists for `analyzeRMSSpectrogramMean` or `analyzeRMSSpectrogram_dBMean`.
"""
from __future__ import annotations

from analyzeAudio.analyzeSpectrogram import (
	analyzeChromagramMean, analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean,
	analyzeSpectralFlatness_dBMean, analyzeSpectralFlatnessMean)
from tests import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from analyzeAudio import SpectrogramMagnitude, SpectrogramPower
	from hunterMakesPy import CallableFunction
	from pathlib import Path

@pytest.mark.parametrize('analyzer', (
	analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean, analyzeSpectralFlatness_dBMean,
	analyzeSpectralFlatnessMean,
))
def test_aspectSpectrogramMagnitude(analyzer: CallableFunction[[SpectrogramMagnitude], float], pathFilename: Path, spectrogramMagnitude: SpectrogramMagnitude, expectedAspect: float) -> None:
	assert_approx(analyzer(spectrogramMagnitude), expectedAspect, 1e-6, 1e-12, analyzer.__name__, pathFilename.name)

@pytest.mark.parametrize('analyzer', (analyzeChromagramMean,))
def test_aspectSpectrogramPower(analyzer: CallableFunction[[SpectrogramPower, int], float], pathFilename: Path, spectrogramPower: SpectrogramPower, sampleRate: int, expectedAspect: float) -> None:
	assert_approx(analyzer(spectrogramPower, sampleRate), expectedAspect, 1e-6, 1e-12, analyzer.__name__, pathFilename.name, sampleRate)
