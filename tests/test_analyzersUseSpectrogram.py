from __future__ import annotations

from analyzeAudio.analyzersUseSpectrogram import (
	analyzeChromagramMean, analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean,
	analyzeSpectralFlatness_dBMean, analyzeSpectralFlatnessMean)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import SpectrogramMagnitudeAndData, SpectrogramPowerAndData

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedAspect', ['analyzeChromagramMean'], indirect=True)
def test_analyzeChromagramMean(spectrogramPowerAndData: SpectrogramPowerAndData, expectedAspect: float) -> None:
	actual = analyzeChromagramMean(spectrogramPowerAndData.spectrogramPower, spectrogramPowerAndData.sampleRate)
	_standardizedEqualScalars('analyzeChromagramMean', spectrogramPowerAndData.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralBandwidthMean'], indirect=True)
def test_analyzeSpectralBandwidthMean(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData, expectedAspect: float) -> None:
	actual = analyzeSpectralBandwidthMean(spectrogramMagnitudeAndData.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralBandwidthMean', spectrogramMagnitudeAndData.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralCentroidMean'], indirect=True)
def test_analyzeSpectralCentroidMean(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData, expectedAspect: float) -> None:
	actual = analyzeSpectralCentroidMean(spectrogramMagnitudeAndData.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralCentroidMean', spectrogramMagnitudeAndData.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralContrastMean'], indirect=True)
def test_analyzeSpectralContrastMean(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData, expectedAspect: float) -> None:
	actual = analyzeSpectralContrastMean(spectrogramMagnitudeAndData.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralContrastMean', spectrogramMagnitudeAndData.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralFlatnessMean'], indirect=True)
def test_analyzeSpectralFlatnessMean(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData, expectedAspect: float) -> None:
	actual = analyzeSpectralFlatnessMean(spectrogramMagnitudeAndData.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatnessMean', spectrogramMagnitudeAndData.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralFlatness_dBMean'], indirect=True)
def test_analyzeSpectralFlatness_dBMean(spectrogramMagnitudeAndData: SpectrogramMagnitudeAndData, expectedAspect: float) -> None:
	actual = analyzeSpectralFlatness_dBMean(spectrogramMagnitudeAndData.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatness_dBMean', spectrogramMagnitudeAndData.pathFilename, actual, expectedAspect)
