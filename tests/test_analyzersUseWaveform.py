from __future__ import annotations

from analyzeAudio.analyzersUseWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean)
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import AspectWaveform

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveformMean'], indirect=True)
def test_analyzeRMSWaveformMean(aspectWaveform: AspectWaveform, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMSWaveformMean(aspectWaveform.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMSWaveformMean', aspectWaveform.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveform_dBMean'], indirect=True)
def test_analyzeRMSWaveform_dBMean(aspectWaveform: AspectWaveform, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMSWaveform_dBMean(aspectWaveform.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMSWaveform_dBMean', aspectWaveform.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeTempogramMean'], indirect=True)
def test_analyzeTempogramMean(aspectWaveform: AspectWaveform, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeTempogramMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeTempogramMean', aspectWaveform.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeTempoMean'], indirect=True)
def test_analyzeTempoMean(aspectWaveform: AspectWaveform, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeTempoMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeTempoMean', aspectWaveform.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeZeroCrossingRateMean'], indirect=True)
def test_analyzeZeroCrossingRateMean(aspectWaveform: AspectWaveform, expectedAspect: float, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeZeroCrossingRateMean(aspectWaveform.waveform)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeZeroCrossingRateMean', aspectWaveform.pathFilename)
