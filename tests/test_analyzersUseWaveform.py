from __future__ import annotations

from analyzeAudio.analyzersUseWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean,
	analyzeZeroCrossingsTotal)
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import WaveformAndData

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveformMean'], indirect=True)
def test_analyzeRMSWaveformMean(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMSWaveformMean(waveformAndData.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMSWaveformMean', waveformAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveform_dBMean'], indirect=True)
def test_analyzeRMSWaveform_dBMean(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMSWaveform_dBMean(waveformAndData.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMSWaveform_dBMean', waveformAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeTempogramMean'], indirect=True)
def test_analyzeTempogramMean(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-4
	if waveformAndData.pathFilename.name == 'ch2_44100_29s_LUFS23_10000Hz.wav':
		approx_rel = 1e-3
	actual = analyzeTempogramMean(waveformAndData.waveform, waveformAndData.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeTempogramMean', waveformAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeTempoMean'], indirect=True)
def test_analyzeTempoMean(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeTempoMean(waveformAndData.waveform, waveformAndData.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeTempoMean', waveformAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeZeroCrossingRateMean'], indirect=True)
def test_analyzeZeroCrossingRateMean(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeZeroCrossingRateMean(waveformAndData.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeZeroCrossingRateMean', waveformAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeZeroCrossingsTotal'], indirect=True)
def test_analyzeZeroCrossingsTotal(waveformAndData: WaveformAndData, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeZeroCrossingsTotal(waveformAndData.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeZeroCrossingsTotal', waveformAndData.pathFilename)
