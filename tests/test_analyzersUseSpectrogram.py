from __future__ import annotations

from analyzeAudio.analyzersUseSpectrogram import (
	analyzeChromagramMean, analyzeSpectralBandwidthMean, analyzeSpectralCentroidMean, analyzeSpectralContrastMean,
	analyzeSpectralFlatness_dBMean, analyzeSpectralFlatnessMean)
from tests.dataSamples.expected import expectedSpectrogram
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectSpectrogramMagnitude, AspectSpectrogramPower

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeChromagramMean']])
def test_analyzeChromagramMean(aspectSpectrogramPower: AspectSpectrogramPower, expected: dict[str, float]) -> None:
	actual = analyzeChromagramMean(aspectSpectrogramPower.spectrogramPower, aspectSpectrogramPower.sampleRate)
	_standardizedEqualScalars(
		'analyzeChromagramMean', aspectSpectrogramPower.pathFilename, actual, expected[aspectSpectrogramPower.pathFilename.name]
	)

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralBandwidthMean']])
def test_analyzeSpectralBandwidthMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expected: dict[str, float]) -> None:
	actual = analyzeSpectralBandwidthMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars(
		'analyzeSpectralBandwidthMean'
		, aspectSpectrogramMagnitude.pathFilename
		, actual
		, expected[aspectSpectrogramMagnitude.pathFilename.name]
	)

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralCentroidMean']])
def test_analyzeSpectralCentroidMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expected: dict[str, float]) -> None:
	actual = analyzeSpectralCentroidMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars(
		'analyzeSpectralCentroidMean'
		, aspectSpectrogramMagnitude.pathFilename
		, actual
		, expected[aspectSpectrogramMagnitude.pathFilename.name]
	)

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralContrastMean']])
def test_analyzeSpectralContrastMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expected: dict[str, float]) -> None:
	actual = analyzeSpectralContrastMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars(
		'analyzeSpectralContrastMean'
		, aspectSpectrogramMagnitude.pathFilename
		, actual
		, expected[aspectSpectrogramMagnitude.pathFilename.name]
	)

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralFlatnessMean']])
def test_analyzeSpectralFlatnessMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expected: dict[str, float]) -> None:
	actual = analyzeSpectralFlatnessMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars(
		'analyzeSpectralFlatnessMean'
		, aspectSpectrogramMagnitude.pathFilename
		, actual
		, expected[aspectSpectrogramMagnitude.pathFilename.name]
	)

@pytest.mark.parametrize('expected', [expectedSpectrogram['analyzeSpectralFlatness_dBMean']])
def test_analyzeSpectralFlatness_dBMean(aspectSpectrogramMagnitude: AspectSpectrogramMagnitude, expected: dict[str, float]) -> None:
	actual = analyzeSpectralFlatness_dBMean(aspectSpectrogramMagnitude.spectrogramMagnitude)
	_standardizedEqualScalars(
		'analyzeSpectralFlatness_dBMean'
		, aspectSpectrogramMagnitude.pathFilename
		, actual
		, expected[aspectSpectrogramMagnitude.pathFilename.name]
	)
