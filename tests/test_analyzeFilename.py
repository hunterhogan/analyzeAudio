"""Expected values are the accepted recordings in `tests.dataSamples.expected.expectedFilename`."""
from __future__ import annotations

from analyzeAudio.analyzeFilename import (
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
from tests import assert_approx
from typing import TYPE_CHECKING
import pytest
import shutil

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from pathlib import Path

pytestmark: pytest.MarkDecorator = pytest.mark.skipif(shutil.which('ffprobe') is None, reason='Each analyzer runs FFprobe, which is not in PATH.')

@pytest.mark.parametrize('analyzer', (
	analyzeAbs_Peak_countTotal, analyzeBit_depthMean, analyzeCrest_factorMean, analyzeDC_offsetMean, analyzeDynamic_rangeOverall,
	analyzeEntropyMean, analyzeFlat_factorMean, analyzeLRAOverall, analyzeLUFShighOverall, analyzeLUFSIntegratedOverall, analyzeLUFSlowOverall,
	analyzeLUFSMomentaryOverall, analyzeLUFSShortTermOverall, analyzeMax_differenceOverall, analyzeMax_levelOverall,
	analyzeMean_differenceMean, analyzeMin_differenceOverall, analyzeMin_levelOverall, analyzeNoise_floor_countTotal,
	analyzeNoise_floorOverall, analyzeNumber_of_samplesTotal, analyzePeak_countTotal, analyzePeak_levelOverall, analyzeRMS_differenceOverall,
	analyzeRMS_levelOverall, analyzeRMS_peakOverall, analyzeRMS_troughOverall, analyzeSpectral_centroid_mean, analyzeSpectral_crest_mean,
	analyzeSpectral_decrease_mean, analyzeSpectral_entropy_mean, analyzeSpectral_flatness_mean, analyzeSpectral_flux_mean,
	analyzeSpectral_kurtosis_mean, analyzeSpectral_mean_mean, analyzeSpectral_rolloff_mean, analyzeSpectral_skewness_mean,
	analyzeSpectral_slope_mean, analyzeSpectral_spread_mean, analyzeSpectral_variance_mean, analyzeTruePeakOverall,
	analyzeZero_crossings_rateOverall, analyzeZero_crossingsTotal,
))
def test_aspectFilename(analyzer: CallableFunction[[Path], float | None], pathFilename: Path, expectedAspect: float | None) -> None:
	assert_approx(analyzer(pathFilename), expectedAspect, 1e-6, 1e-12, analyzer.__name__, pathFilename.name)
