from __future__ import annotations

from analyzeAudio.analyzersUseSpectrogram import (
	analyzeChromagramMean, analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean,
	analyzeSpectralFlatness_dBMean, analyzeSpectralFlatnessMean)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectSpectrogramMagnitude, AspectSpectrogramPower

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedAspect', ['analyzeChromagramMean'], indirect=True)
def test_analyzeChromagramMean(aspectSpectrogramPower: AspectSpectrogramPower, expectedAspect: float) -> None:
	actual = analyzeChromagramMean(aspectSpectrogramPower.spectrogramPower, aspectSpectrogramPower.sampleRate)
	_standardizedEqualScalars('analyzeChromagramMean', aspectSpectrogramPower.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralBandwidthMean'], indirect=True)
def test_analyzeSpectralBandwidthMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expectedAspect: float) -> None:
	actual = analyzeSpectralBandwidthMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralBandwidthMean', aspectSpectrogramMagnitude.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralCentroidMean'], indirect=True)
def test_analyzeSpectralCentroidMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expectedAspect: float) -> None:
	actual = analyzeSpectralCentroidMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralCentroidMean', aspectSpectrogramMagnitude.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralContrastMean'], indirect=True)
def test_analyzeSpectralContrastMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expectedAspect: float) -> None:
	actual = analyzeSpectralContrastMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralContrastMean', aspectSpectrogramMagnitude.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralFlatnessMean'], indirect=True)
def test_analyzeSpectralFlatnessMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expectedAspect: float) -> None:
	actual = analyzeSpectralFlatnessMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatnessMean', aspectSpectrogramMagnitude.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectralFlatness_dBMean'], indirect=True)
def test_analyzeSpectralFlatness_dBMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expectedAspect: float) -> None:
	actual = analyzeSpectralFlatness_dBMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars('analyzeSpectralFlatness_dBMean', aspectSpectrogramMagnitude.pathFilename, actual, expectedAspect)
