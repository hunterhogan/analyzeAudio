from __future__ import annotations

from analyzeAudio.analyzersUseFilename import (
	analyzeAbs_Peak_countTotal, analyzeBit_depthMean, analyzeCrest_factorMean, analyzeDC_offsetMean, analyzeDynamic_rangeOverall,
	analyzeEntropyMean, analyzeFlat_factorMean, analyzeLRAOverall, analyzeLUFShighOverall, analyzeLUFSIntegratedOverall, analyzeLUFSlowOverall,
	analyzeLUFSMomentaryOverall, analyzeLUFSShortTermOverall, analyzeMax_differenceOverall, analyzeMax_levelOverall,
	analyzeMean_differenceMean, analyzeMin_differenceOverall, analyzeMin_levelOverall, analyzeNoise_floor_countTotal,
	analyzeNoise_floorOverall, analyzeNumber_of_samplesTotal, analyzePeak_countTotal, analyzePeak_levelOverall, analyzeRMS_differenceOverall,
	analyzeRMS_levelOverall, analyzeRMS_peakOverall, analyzeRMS_troughOverall, analyzeSpectral_centroid_mean, analyzeSpectral_crest_mean,
	analyzeSpectral_decrease_mean, analyzeSpectral_entropy_mean, analyzeSpectral_flatness_mean, analyzeSpectral_flux_mean,
	analyzeSpectral_kurtosis_mean, analyzeSpectral_mean_mean, analyzeSpectral_rolloff_mean, analyzeSpectral_skewness_mean,
	analyzeSpectral_slope_mean, analyzeSpectral_spread_mean, analyzeSpectral_variance_mean, analyzeTruePeakOverall,
	analyzeZero_crossings_rateOverall, analyzeZero_crossingsTotal)
from typing import TYPE_CHECKING
import os
import pytest

if TYPE_CHECKING:
	from pathlib import Path

pytestmark: pytest.MarkDecorator = pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Skipped in GitHub Actions')

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float | None, expected: float | None) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedAspect', ['analyzeAbs_Peak_countTotal'], indirect=True)
def test_analyzeAbs_Peak_countTotal(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeAbs_Peak_countTotal(pathFilename)
	_standardizedEqualScalars('analyzeAbs_Peak_countTotal', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeBit_depthMean'], indirect=True)
def test_analyzeBit_depthMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeBit_depthMean(pathFilename)
	_standardizedEqualScalars('analyzeBit_depthMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeCrest_factorMean'], indirect=True)
def test_analyzeCrest_factorMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeCrest_factorMean(pathFilename)
	_standardizedEqualScalars('analyzeCrest_factorMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeDC_offsetMean'], indirect=True)
def test_analyzeDC_offsetMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeDC_offsetMean(pathFilename)
	_standardizedEqualScalars('analyzeDC_offsetMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeDynamic_rangeOverall'], indirect=True)
def test_analyzeDynamic_rangeOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeDynamic_rangeOverall(pathFilename)
	_standardizedEqualScalars('analyzeDynamic_rangeOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeEntropyMean'], indirect=True)
def test_analyzeEntropyMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeEntropyMean(pathFilename)
	_standardizedEqualScalars('analyzeEntropyMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeFlat_factorMean'], indirect=True)
def test_analyzeFlat_factorMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeFlat_factorMean(pathFilename)
	_standardizedEqualScalars('analyzeFlat_factorMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLRAOverall'], indirect=True)
def test_analyzeLRAOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLRAOverall(pathFilename)
	_standardizedEqualScalars('analyzeLRAOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSIntegratedOverall'], indirect=True)
def test_analyzeLUFSIntegratedOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLUFSIntegratedOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSIntegratedOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSMomentaryOverall'], indirect=True)
def test_analyzeLUFSMomentaryOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLUFSMomentaryOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSMomentaryOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSShortTermOverall'], indirect=True)
def test_analyzeLUFSShortTermOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLUFSShortTermOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSShortTermOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFShighOverall'], indirect=True)
def test_analyzeLUFShighOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLUFShighOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFShighOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSlowOverall'], indirect=True)
def test_analyzeLUFSlowOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeLUFSlowOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSlowOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeMax_differenceOverall'], indirect=True)
def test_analyzeMax_differenceOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeMax_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeMax_differenceOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeMax_levelOverall'], indirect=True)
def test_analyzeMax_levelOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeMax_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeMax_levelOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeMean_differenceMean'], indirect=True)
def test_analyzeMean_differenceMean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeMean_differenceMean(pathFilename)
	_standardizedEqualScalars('analyzeMean_differenceMean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeMin_differenceOverall'], indirect=True)
def test_analyzeMin_differenceOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeMin_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeMin_differenceOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeMin_levelOverall'], indirect=True)
def test_analyzeMin_levelOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeMin_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeMin_levelOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeNoise_floor_countTotal'], indirect=True)
def test_analyzeNoise_floor_countTotal(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeNoise_floor_countTotal(pathFilename)
	_standardizedEqualScalars('analyzeNoise_floor_countTotal', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeNoise_floorOverall'], indirect=True)
def test_analyzeNoise_floorOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeNoise_floorOverall(pathFilename)
	_standardizedEqualScalars('analyzeNoise_floorOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeNumber_of_samplesTotal'], indirect=True)
def test_analyzeNumber_of_samplesTotal(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeNumber_of_samplesTotal(pathFilename)
	_standardizedEqualScalars('analyzeNumber_of_samplesTotal', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzePeak_countTotal'], indirect=True)
def test_analyzePeak_countTotal(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzePeak_countTotal(pathFilename)
	_standardizedEqualScalars('analyzePeak_countTotal', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzePeak_levelOverall'], indirect=True)
def test_analyzePeak_levelOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzePeak_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzePeak_levelOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_differenceOverall'], indirect=True)
def test_analyzeRMS_differenceOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeRMS_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_differenceOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_levelOverall'], indirect=True)
def test_analyzeRMS_levelOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeRMS_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_levelOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_peakOverall'], indirect=True)
def test_analyzeRMS_peakOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeRMS_peakOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_peakOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_troughOverall'], indirect=True)
def test_analyzeRMS_troughOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeRMS_troughOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_troughOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_centroid_mean'], indirect=True)
def test_analyzeSpectral_centroid_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_centroid_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_centroid_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_crest_mean'], indirect=True)
def test_analyzeSpectral_crest_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_crest_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_crest_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_decrease_mean'], indirect=True)
def test_analyzeSpectral_decrease_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_decrease_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_decrease_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_entropy_mean'], indirect=True)
def test_analyzeSpectral_entropy_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_entropy_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_entropy_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_flatness_mean'], indirect=True)
def test_analyzeSpectral_flatness_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_flatness_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_flatness_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_flux_mean'], indirect=True)
def test_analyzeSpectral_flux_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_flux_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_flux_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_kurtosis_mean'], indirect=True)
def test_analyzeSpectral_kurtosis_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_kurtosis_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_kurtosis_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_mean_mean'], indirect=True)
def test_analyzeSpectral_mean_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_mean_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_mean_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_rolloff_mean'], indirect=True)
def test_analyzeSpectral_rolloff_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_rolloff_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_rolloff_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_skewness_mean'], indirect=True)
def test_analyzeSpectral_skewness_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_skewness_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_skewness_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_slope_mean'], indirect=True)
def test_analyzeSpectral_slope_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_slope_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_slope_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_spread_mean'], indirect=True)
def test_analyzeSpectral_spread_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_spread_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_spread_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_variance_mean'], indirect=True)
def test_analyzeSpectral_variance_mean(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeSpectral_variance_mean(pathFilename)
	_standardizedEqualScalars('analyzeSpectral_variance_mean', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeTruePeakOverall'], indirect=True)
def test_analyzeTruePeakOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeTruePeakOverall(pathFilename)
	_standardizedEqualScalars('analyzeTruePeakOverall', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeZero_crossingsTotal'], indirect=True)
def test_analyzeZero_crossingsTotal(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeZero_crossingsTotal(pathFilename)
	_standardizedEqualScalars('analyzeZero_crossingsTotal', pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeZero_crossings_rateOverall'], indirect=True)
def test_analyzeZero_crossings_rateOverall(pathFilename: Path, expectedAspect: float | None) -> None:
	actual = analyzeZero_crossings_rateOverall(pathFilename)
	_standardizedEqualScalars('analyzeZero_crossings_rateOverall', pathFilename, actual, expectedAspect)
