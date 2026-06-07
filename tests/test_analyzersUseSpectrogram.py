from __future__ import annotations

from analyzeAudio.analyzersUseSpectrogram import (
	analyzeChromagramMean, analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean,
	analyzeSpectralFlatness_dBMean, analyzeSpectralFlatnessMean)
from tests.dataSamples.expected import expectedSpectrogram
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import SpectrogramMagnitudeSampleRate, SpectrogramPowerSampleRate

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeChromagramMean']])
def test_analyzeChromagramMean(spectrogramPower_sampleRate: SpectrogramPowerSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeChromagramMean(spectrogramPower_sampleRate.spectrogramPower, spectrogramPower_sampleRate.sampleRate)
	_standardizedEqualScalars('analyzeChromagramMean', spectrogramPower_sampleRate.pathFilename, actual, expected[spectrogramPower_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralBandwidthMean']])
def test_analyzeSpectralBandwidthMean(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeSpectralBandwidthMean(spectrogramMagnitude_sampleRate.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralBandwidthMean', spectrogramMagnitude_sampleRate.pathFilename, actual, expected[spectrogramMagnitude_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralCentroidMean']])
def test_analyzeSpectralCentroidMean(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeSpectralCentroidMean(spectrogramMagnitude_sampleRate.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralCentroidMean', spectrogramMagnitude_sampleRate.pathFilename, actual, expected[spectrogramMagnitude_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralContrastMean']])
def test_analyzeSpectralContrastMean(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeSpectralContrastMean(spectrogramMagnitude_sampleRate.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralContrastMean', spectrogramMagnitude_sampleRate.pathFilename, actual, expected[spectrogramMagnitude_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralFlatnessMean']])
def test_analyzeSpectralFlatnessMean(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeSpectralFlatnessMean(spectrogramMagnitude_sampleRate.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatnessMean', spectrogramMagnitude_sampleRate.pathFilename, actual, expected[spectrogramMagnitude_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralFlatness_dBMean']])
def test_analyzeSpectralFlatness_dBMean(spectrogramMagnitude_sampleRate: SpectrogramMagnitudeSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeSpectralFlatness_dBMean(spectrogramMagnitude_sampleRate.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatness_dBMean', spectrogramMagnitude_sampleRate.pathFilename, actual, expected[spectrogramMagnitude_sampleRate.pathFilename.name])
