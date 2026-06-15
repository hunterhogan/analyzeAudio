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
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path

@pytest.mark.parametrize('expectedAspect', ['analyzeAbs_Peak_countTotal'], indirect=True)
def test_analyzeAbs_Peak_countTotal(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeAbs_Peak_countTotal(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeAbs_Peak_countTotal', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeBit_depthMean'], indirect=True)
def test_analyzeBit_depthMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeBit_depthMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeBit_depthMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeCrest_factorMean'], indirect=True)
def test_analyzeCrest_factorMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeCrest_factorMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeCrest_factorMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeDC_offsetMean'], indirect=True)
def test_analyzeDC_offsetMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeDC_offsetMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeDC_offsetMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeDynamic_rangeOverall'], indirect=True)
def test_analyzeDynamic_rangeOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeDynamic_rangeOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeDynamic_rangeOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeEntropyMean'], indirect=True)
def test_analyzeEntropyMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeEntropyMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeEntropyMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeFlat_factorMean'], indirect=True)
def test_analyzeFlat_factorMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeFlat_factorMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeFlat_factorMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLRAOverall'], indirect=True)
def test_analyzeLRAOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLRAOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLRAOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSIntegratedOverall'], indirect=True)
def test_analyzeLUFSIntegratedOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLUFSIntegratedOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLUFSIntegratedOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSMomentaryOverall'], indirect=True)
def test_analyzeLUFSMomentaryOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLUFSMomentaryOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLUFSMomentaryOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSShortTermOverall'], indirect=True)
def test_analyzeLUFSShortTermOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLUFSShortTermOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLUFSShortTermOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFShighOverall'], indirect=True)
def test_analyzeLUFShighOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLUFShighOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLUFShighOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeLUFSlowOverall'], indirect=True)
def test_analyzeLUFSlowOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeLUFSlowOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeLUFSlowOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeMax_differenceOverall'], indirect=True)
def test_analyzeMax_differenceOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMax_differenceOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeMax_differenceOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeMax_levelOverall'], indirect=True)
def test_analyzeMax_levelOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMax_levelOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeMax_levelOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeMean_differenceMean'], indirect=True)
def test_analyzeMean_differenceMean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMean_differenceMean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeMean_differenceMean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeMin_differenceOverall'], indirect=True)
def test_analyzeMin_differenceOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMin_differenceOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeMin_differenceOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeMin_levelOverall'], indirect=True)
def test_analyzeMin_levelOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeMin_levelOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeMin_levelOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNoise_floor_countTotal'], indirect=True)
def test_analyzeNoise_floor_countTotal(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeNoise_floor_countTotal(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNoise_floor_countTotal', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNoise_floorOverall'], indirect=True)
def test_analyzeNoise_floorOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeNoise_floorOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNoise_floorOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNumber_of_samplesTotal'], indirect=True)
def test_analyzeNumber_of_samplesTotal(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeNumber_of_samplesTotal(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNumber_of_samplesTotal', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzePeak_countTotal'], indirect=True)
def test_analyzePeak_countTotal(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzePeak_countTotal(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzePeak_countTotal', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzePeak_levelOverall'], indirect=True)
def test_analyzePeak_levelOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzePeak_levelOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzePeak_levelOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_differenceOverall'], indirect=True)
def test_analyzeRMS_differenceOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMS_differenceOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMS_differenceOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_levelOverall'], indirect=True)
def test_analyzeRMS_levelOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMS_levelOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMS_levelOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_peakOverall'], indirect=True)
def test_analyzeRMS_peakOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMS_peakOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMS_peakOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeRMS_troughOverall'], indirect=True)
def test_analyzeRMS_troughOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeRMS_troughOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeRMS_troughOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_centroid_mean'], indirect=True)
def test_analyzeSpectral_centroid_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_centroid_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_centroid_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_crest_mean'], indirect=True)
def test_analyzeSpectral_crest_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_crest_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_crest_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_decrease_mean'], indirect=True)
def test_analyzeSpectral_decrease_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_decrease_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_decrease_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_entropy_mean'], indirect=True)
def test_analyzeSpectral_entropy_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_entropy_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_entropy_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_flatness_mean'], indirect=True)
def test_analyzeSpectral_flatness_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_flatness_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_flatness_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_flux_mean'], indirect=True)
def test_analyzeSpectral_flux_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_flux_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_flux_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_kurtosis_mean'], indirect=True)
def test_analyzeSpectral_kurtosis_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_kurtosis_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_kurtosis_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_mean_mean'], indirect=True)
def test_analyzeSpectral_mean_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_mean_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_mean_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_rolloff_mean'], indirect=True)
def test_analyzeSpectral_rolloff_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_rolloff_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_rolloff_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_skewness_mean'], indirect=True)
def test_analyzeSpectral_skewness_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_skewness_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_skewness_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_slope_mean'], indirect=True)
def test_analyzeSpectral_slope_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_slope_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_slope_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_spread_mean'], indirect=True)
def test_analyzeSpectral_spread_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_spread_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_spread_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeSpectral_variance_mean'], indirect=True)
def test_analyzeSpectral_variance_mean(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSpectral_variance_mean(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSpectral_variance_mean', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeTruePeakOverall'], indirect=True)
def test_analyzeTruePeakOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeTruePeakOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeTruePeakOverall', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeZero_crossingsTotal'], indirect=True)
def test_analyzeZero_crossingsTotal(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeZero_crossingsTotal(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeZero_crossingsTotal', pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeZero_crossings_rateOverall'], indirect=True)
def test_analyzeZero_crossings_rateOverall(pathFilename: Path, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeZero_crossings_rateOverall(pathFilename)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeZero_crossings_rateOverall', pathFilename)
