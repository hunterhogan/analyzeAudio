# ruff: noqa: ERA001
from __future__ import annotations

from analyzeAudio.analyzersUseWaveform import (
	analyzeRMSWaveform_dBMean, analyzeRMSWaveformMean, analyzeTempogramMean, analyzeTempoMean, analyzeZeroCrossingRateMean)
from tests.dataSamples.expected import expectedWaveform
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectWaveform

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float) -> None:
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeRMSWaveformMean']])
def test_analyzeRMSWaveformMean(aspectWaveform: AspectWaveform, expected: dict[str, float]) -> None:
	actual = analyzeRMSWaveformMean(aspectWaveform.waveform)
	_standardizedEqualScalars('analyzeRMSWaveformMean', aspectWaveform.pathFilename, actual, expected[aspectWaveform.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeRMSWaveform_dBMean']])
def test_analyzeRMSWaveform_dBMean(aspectWaveform: AspectWaveform, expected: dict[str, float]) -> None:
	actual = analyzeRMSWaveform_dBMean(aspectWaveform.waveform)
	_standardizedEqualScalars('analyzeRMSWaveform_dBMean', aspectWaveform.pathFilename, actual, expected[aspectWaveform.pathFilename.name])

# TODO tolerances are quite high
# @pytest.mark.parametrize('expected', [expectedWaveform['analyzeTempogramMean']])
# def test_analyzeTempogramMean(aspectWaveform: AspectWaveform, expected: dict[str, float]) -> None:
# 	actual = analyzeTempogramMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
# 	_standardizedEqualScalars('analyzeTempogramMean', aspectWaveform.pathFilename, actual, expected[aspectWaveform.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeTempoMean']])
def test_analyzeTempoMean(aspectWaveform: AspectWaveform, expected: dict[str, float]) -> None:
	actual = analyzeTempoMean(aspectWaveform.waveform, aspectWaveform.sampleRate)
	_standardizedEqualScalars('analyzeTempoMean', aspectWaveform.pathFilename, actual, expected[aspectWaveform.pathFilename.name])

@pytest.mark.parametrize('expected', [expectedWaveform['analyzeZeroCrossingRateMean']])
def test_analyzeZeroCrossingRateMean(aspectWaveform: AspectWaveform, expected: dict[str, float]) -> None:
	actual = analyzeZeroCrossingRateMean(aspectWaveform.waveform)
	_standardizedEqualScalars(
		'analyzeZeroCrossingRateMean', aspectWaveform.pathFilename, actual, expected[aspectWaveform.pathFilename.name]
	)
