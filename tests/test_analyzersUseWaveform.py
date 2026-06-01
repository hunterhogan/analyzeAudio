from __future__ import annotations

from analyzeAudio import analyzersUseWaveform, audioAspects
from typing import Any
import math
import numpy
import pytest

@pytest.mark.parametrize(
	('analyzerName', 'boolNeedsSampleRate', 'dictionaryKeywordArguments'),
	[
		('analyzeTempogram', True, {'hop_length': 521}),
		('analyzeRMS', False, {'frame_length': 4097, 'hop_length': 307}),
		('analyzeTempo', True, {'hop_length': 521}),
		('analyzeZeroCrossingRate', False, {'frame_length': 4097, 'hop_length': 307}),
	],
)
def test_waveform_analyzers_return_finite_numpy_arrays(
	waveformLibrosaCase: tuple[numpy.ndarray, int], analyzerName: str, boolNeedsSampleRate: bool, dictionaryKeywordArguments: dict[str, Any],
) -> None:
	arrayWaveform, sampleRate = waveformLibrosaCase
	analyzer = getattr(analyzersUseWaveform, analyzerName)
	arrayAnalysis = analyzer(arrayWaveform, sampleRate, **dictionaryKeywordArguments) if boolNeedsSampleRate else analyzer(arrayWaveform, **dictionaryKeywordArguments)
	assert isinstance(arrayAnalysis, numpy.ndarray), (
		f"{analyzerName} returned {type(arrayAnalysis).__name__}, expected numpy.ndarray."
	)
	assert arrayAnalysis.ndim >= 1, (
		f"{analyzerName} returned array with {arrayAnalysis.ndim} dimensions, expected at least one dimension."
	)
	assert arrayAnalysis.size > 0, (
		f"{analyzerName} returned empty array for {sampleRate=} and {dictionaryKeywordArguments=}."
	)
	assert bool(numpy.isfinite(arrayAnalysis).all()), (
		f"{analyzerName} returned non-finite values for {sampleRate=} and {dictionaryKeywordArguments=}."
	)

@pytest.mark.parametrize(
	('stringAspectName', 'analyzerName', 'analyzerNameRaw', 'boolNeedsSampleRate', 'dictionaryKeywordArguments'),
	[
		('Tempogram mean', 'analyzeTempogramMean', 'analyzeTempogram', True, {'hop_length': 521}),
		('RMS from waveform mean', 'analyzeRMSMean', 'analyzeRMS', False, {'frame_length': 4097, 'hop_length': 307}),
		('Tempo mean', 'analyzeTempoMean', 'analyzeTempo', True, {'hop_length': 521}),
		('Zero-crossing rate mean', 'analyzeZeroCrossingRateMean', 'analyzeZeroCrossingRate', False, {'frame_length': 4097, 'hop_length': 307}),
	],
)
def test_waveform_mean_analyzers_match_registered_functions(
	waveformLibrosaCase: tuple[numpy.ndarray, int], stringAspectName: str, analyzerName: str, analyzerNameRaw: str, boolNeedsSampleRate: bool,
	dictionaryKeywordArguments: dict[str, Any],
) -> None:
	arrayWaveform, sampleRate = waveformLibrosaCase
	analyzerMean = getattr(analyzersUseWaveform, analyzerName)
	analyzerRaw = getattr(analyzersUseWaveform, analyzerNameRaw)
	valueMean = analyzerMean(arrayWaveform, sampleRate, **dictionaryKeywordArguments) if boolNeedsSampleRate else analyzerMean(arrayWaveform, **dictionaryKeywordArguments)
	arrayRaw = analyzerRaw(arrayWaveform, sampleRate, **dictionaryKeywordArguments) if boolNeedsSampleRate else analyzerRaw(arrayWaveform, **dictionaryKeywordArguments)
	valueExpected = float(arrayRaw.mean().item())
	assert isinstance(valueMean, float), (
		f"{analyzerName} returned {type(valueMean).__name__}, expected float for aspect {stringAspectName}."
	)
	assert math.isfinite(valueMean), (
		f"{analyzerName} returned non-finite value {valueMean} for aspect {stringAspectName}."
	)
	assert valueMean == pytest.approx(valueExpected), (
		f"{analyzerName} returned {valueMean}, expected approx {valueExpected} for {dictionaryKeywordArguments=}."
	)
	assert stringAspectName in audioAspects, (
		f"audioAspects did not register {stringAspectName}; available keys do not include the expected waveform aspect name."
	)
	assert audioAspects[stringAspectName]['analyzer'] is analyzerMean, (
		f"audioAspects[{stringAspectName!r}] registered {audioAspects[stringAspectName]['analyzer']}, expected {analyzerMean}."
	)
