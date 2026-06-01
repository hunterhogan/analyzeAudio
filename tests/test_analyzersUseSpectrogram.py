from __future__ import annotations

from analyzeAudio import analyzersUseSpectrogram, audioAspects
from typing import TYPE_CHECKING
import math
import pytest

if TYPE_CHECKING:
	import numpy

@pytest.mark.parametrize(
	('dictionaryKeywordArguments',),
	[
		({"n_fft": 513, "hop_length": 137},),
	],
)
def test_analyzeChromagramMean_returns_registered_float(
	spectrogramFeatureCase: tuple[numpy.ndarray, numpy.ndarray, int], dictionaryKeywordArguments: dict[str, int],
) -> None:
	arraySpectrogramMagnitudeUnused, arraySpectrogramPower, sampleRate = spectrogramFeatureCase
	valueChromagramMean = analyzersUseSpectrogram.analyzeChromagramMean(
		arraySpectrogramPower,
		sampleRate,
		**dictionaryKeywordArguments,
	)
	assert isinstance(valueChromagramMean, float), (
		f"analyzeChromagramMean returned {type(valueChromagramMean).__name__}, expected float for {sampleRate=} and {dictionaryKeywordArguments=}."
	)
	assert math.isfinite(valueChromagramMean), (
		f"analyzeChromagramMean returned non-finite value {valueChromagramMean} for {sampleRate=} and {dictionaryKeywordArguments=}."
	)
	assert 'Chromagram mean' in audioAspects, (
		"audioAspects did not register 'Chromagram mean', expected the spectrogram chromagram aspect key."
	)
	assert audioAspects['Chromagram mean']['analyzer'] is analyzersUseSpectrogram.analyzeChromagramMean, (
		f"audioAspects['Chromagram mean'] registered {audioAspects['Chromagram mean']['analyzer']}, expected {analyzersUseSpectrogram.analyzeChromagramMean}."
	)

@pytest.mark.parametrize(
	('dictionaryKeywordArguments',),
	[
		({"n_fft": 513, "hop_length": 137},),
	],
)
def test_analyzeSpectralContrastMean_returns_registered_float(
	spectrogramFeatureCase: tuple[numpy.ndarray, numpy.ndarray, int], dictionaryKeywordArguments: dict[str, int],
) -> None:
	arraySpectrogramMagnitude, arraySpectrogramPowerUnused, sampleRateUnused = spectrogramFeatureCase
	valueSpectralContrastMean = analyzersUseSpectrogram.analyzeSpectralContrastMean(
		arraySpectrogramMagnitude,
		**dictionaryKeywordArguments,
	)
	assert isinstance(valueSpectralContrastMean, float), (
		f"analyzeSpectralContrastMean returned {type(valueSpectralContrastMean).__name__}, expected float for {dictionaryKeywordArguments=}."
	)
	assert math.isfinite(valueSpectralContrastMean), (
		f"analyzeSpectralContrastMean returned non-finite value {valueSpectralContrastMean} for {dictionaryKeywordArguments=}."
	)
	assert 'Spectral Contrast mean' in audioAspects, (
		"audioAspects did not register 'Spectral Contrast mean', expected the spectral contrast aspect key."
	)
	assert audioAspects['Spectral Contrast mean']['analyzer'] is analyzersUseSpectrogram.analyzeSpectralContrastMean, (
		f"audioAspects['Spectral Contrast mean'] registered {audioAspects['Spectral Contrast mean']['analyzer']}, expected {analyzersUseSpectrogram.analyzeSpectralContrastMean}."
	)

@pytest.mark.parametrize(
	('dictionaryKeywordArguments',),
	[
		({"n_fft": 513, "hop_length": 137},),
	],
)
def test_analyzeSpectralBandwidthMean_returns_registered_float(
	spectrogramFeatureCase: tuple[numpy.ndarray, numpy.ndarray, int], dictionaryKeywordArguments: dict[str, int],
) -> None:
	arraySpectrogramMagnitude, arraySpectrogramPowerUnused, sampleRateUnused = spectrogramFeatureCase
	valueSpectralBandwidthMean = analyzersUseSpectrogram.analyzeSpectralBandwidthMean(
		arraySpectrogramMagnitude,
		**dictionaryKeywordArguments,
	)
	assert isinstance(valueSpectralBandwidthMean, float), (
		f"analyzeSpectralBandwidthMean returned {type(valueSpectralBandwidthMean).__name__}, expected float for {dictionaryKeywordArguments=}."
	)
	assert math.isfinite(valueSpectralBandwidthMean), (
		f"analyzeSpectralBandwidthMean returned non-finite value {valueSpectralBandwidthMean} for {dictionaryKeywordArguments=}."
	)
	assert 'Spectral Bandwidth mean' in audioAspects, (
		"audioAspects did not register 'Spectral Bandwidth mean', expected the spectral bandwidth aspect key."
	)
	assert audioAspects['Spectral Bandwidth mean']['analyzer'] is analyzersUseSpectrogram.analyzeSpectralBandwidthMean, (
		f"audioAspects['Spectral Bandwidth mean'] registered {audioAspects['Spectral Bandwidth mean']['analyzer']}, expected {analyzersUseSpectrogram.analyzeSpectralBandwidthMean}."
	)

@pytest.mark.parametrize(
	('dictionaryKeywordArguments',),
	[
		({"n_fft": 513, "hop_length": 137},),
	],
)
def test_analyzeSpectralCentroidMean_returns_registered_float(
	spectrogramFeatureCase: tuple[numpy.ndarray, numpy.ndarray, int], dictionaryKeywordArguments: dict[str, int],
) -> None:
	arraySpectrogramMagnitude, arraySpectrogramPowerUnused, sampleRateUnused = spectrogramFeatureCase
	valueSpectralCentroidMean = analyzersUseSpectrogram.analyzeSpectralCentroidMean(
		arraySpectrogramMagnitude,
		**dictionaryKeywordArguments,
	)
	assert isinstance(valueSpectralCentroidMean, float), (
		f"analyzeSpectralCentroidMean returned {type(valueSpectralCentroidMean).__name__}, expected float for {dictionaryKeywordArguments=}."
	)
	assert math.isfinite(valueSpectralCentroidMean), (
		f"analyzeSpectralCentroidMean returned non-finite value {valueSpectralCentroidMean} for {dictionaryKeywordArguments=}."
	)
	assert 'Spectral Centroid mean' in audioAspects, (
		"audioAspects did not register 'Spectral Centroid mean', expected the spectral centroid aspect key."
	)
	assert audioAspects['Spectral Centroid mean']['analyzer'] is analyzersUseSpectrogram.analyzeSpectralCentroidMean, (
		f"audioAspects['Spectral Centroid mean'] registered {audioAspects['Spectral Centroid mean']['analyzer']}, expected {analyzersUseSpectrogram.analyzeSpectralCentroidMean}."
	)

@pytest.mark.parametrize(
	('dictionaryKeywordArguments',),
	[
		({"n_fft": 513, "hop_length": 137},),
	],
)
def test_analyzeSpectralFlatnessMean_returns_registered_float(
	spectrogramFeatureCase: tuple[numpy.ndarray, numpy.ndarray, int], dictionaryKeywordArguments: dict[str, int],
) -> None:
	arraySpectrogramMagnitude, arraySpectrogramPowerUnused, sampleRateUnused = spectrogramFeatureCase
	valueSpectralFlatnessMean = analyzersUseSpectrogram.analyzeSpectralFlatnessMean(
		arraySpectrogramMagnitude,
		**dictionaryKeywordArguments,
	)
	assert isinstance(valueSpectralFlatnessMean, float), (
		f"analyzeSpectralFlatnessMean returned {type(valueSpectralFlatnessMean).__name__}, expected float for {dictionaryKeywordArguments=}."
	)
	assert math.isfinite(valueSpectralFlatnessMean), (
		f"analyzeSpectralFlatnessMean returned non-finite value {valueSpectralFlatnessMean} for {dictionaryKeywordArguments=}."
	)
	assert 'Spectral Flatness mean' in audioAspects, (
		"audioAspects did not register 'Spectral Flatness mean', expected the spectral flatness aspect key."
	)
	assert audioAspects['Spectral Flatness mean']['analyzer'] is analyzersUseSpectrogram.analyzeSpectralFlatnessMean, (
		f"audioAspects['Spectral Flatness mean'] registered {audioAspects['Spectral Flatness mean']['analyzer']}, expected {analyzersUseSpectrogram.analyzeSpectralFlatnessMean}."
	)
