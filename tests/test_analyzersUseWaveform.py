from __future__ import annotations

from analyzeAudio.analyzersUseWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean)
from tests.dataSamples.expected import expectedWaveform
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import WaveformSampleRate

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeRMSWaveformMean']])
def test_analyzeRMSWaveformMean(waveform_sampleRate: WaveformSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeRMSWaveformMean(waveform_sampleRate.waveform)
	_standardizedEqualScalars('analyzeRMSWaveformMean', waveform_sampleRate.pathFilename, actual, expected[waveform_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeRMSWaveform_dBMean']])
def test_analyzeRMSWaveform_dBMean(waveform_sampleRate: WaveformSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeRMSWaveform_dBMean(waveform_sampleRate.waveform)
	_standardizedEqualScalars('analyzeRMSWaveform_dBMean', waveform_sampleRate.pathFilename, actual, expected[waveform_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeTempogramMean']])
def test_analyzeTempogramMean(waveform_sampleRate: WaveformSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeTempogramMean(waveform_sampleRate.waveform, waveform_sampleRate.sampleRate)
	_standardizedEqualScalars('analyzeTempogramMean', waveform_sampleRate.pathFilename, actual, expected[waveform_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeTempoMean']])
def test_analyzeTempoMean(waveform_sampleRate: WaveformSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeTempoMean(waveform_sampleRate.waveform, waveform_sampleRate.sampleRate)
	_standardizedEqualScalars('analyzeTempoMean', waveform_sampleRate.pathFilename, actual, expected[waveform_sampleRate.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeZeroCrossingRateMean']])
def test_analyzeZeroCrossingRateMean(waveform_sampleRate: WaveformSampleRate, expected: dict[str, float]) -> None:
	actual = analyzeZeroCrossingRateMean(waveform_sampleRate.waveform)
	_standardizedEqualScalars('analyzeZeroCrossingRateMean', waveform_sampleRate.pathFilename, actual, expected[waveform_sampleRate.pathFilename.name])
