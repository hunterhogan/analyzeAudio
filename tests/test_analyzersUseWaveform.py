# ruff: noqa: ERA001
from __future__ import annotations

from analyzeAudio.analyzersUseWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean)
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectWaveform

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveformMean'], indirect=True)
def test_analyzeRMSWaveformMean(aspectWaveform: AspectWaveform, expectedAspect: float) -> None:
	actual = analyzeRMSWaveformMean(aspectWaveform.waveform)
	_standardizedEqualScalars('analyzeRMSWaveformMean', aspectWaveform.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMSWaveform_dBMean'], indirect=True)
def test_analyzeRMSWaveform_dBMean(aspectWaveform: AspectWaveform, expectedAspect: float) -> None:
	actual = analyzeRMSWaveform_dBMean(aspectWaveform.waveform)
	_standardizedEqualScalars('analyzeRMSWaveform_dBMean', aspectWaveform.pathFilename, actual, expectedAspect)

# TODO tolerances are quite high
# @pytest.mark.parametrize('expectedAspect', ['analyzeTempogramMean'], indirect=True)
# def test_analyzeTempogramMean(aspectWaveform: AspectWaveform, expectedAspect: float) -> None:
# 	actual = analyzeTempogramMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
# 	_standardizedEqualScalars('analyzeTempogramMean', aspectWaveform.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeTempoMean'], indirect=True)
def test_analyzeTempoMean(aspectWaveform: AspectWaveform, expectedAspect: float) -> None:
	actual = analyzeTempoMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
	_standardizedEqualScalars('analyzeTempoMean', aspectWaveform.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeZeroCrossingRateMean'], indirect=True)
def test_analyzeZeroCrossingRateMean(aspectWaveform: AspectWaveform, expectedAspect: float) -> None:
	actual = analyzeZeroCrossingRateMean(aspectWaveform.waveform)
	_standardizedEqualScalars('analyzeZeroCrossingRateMean', aspectWaveform.pathFilename, actual, expectedAspect)
