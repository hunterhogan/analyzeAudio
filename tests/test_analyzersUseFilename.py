from __future__ import annotations

from analyzeAudio.analyzersUseFilename import (
	analyzeAbs_Peak_countTotal, analyzeBit_depthMean, analyzeCrest_factorMean, analyzeDC_offsetMean, analyzeDynamic_rangeOverall,
	analyzeEntropyMean, analyzeFlat_factorMean, analyzeLRAOverall, analyzeLUFShighOverall, analyzeLUFSIntegratedOverall, analyzeLUFSlowOverall,
	analyzeLUFSMomentaryOverall, analyzeLUFSShortTermOverall, analyzeMax_differenceOverall, analyzeMax_levelOverall,
	analyzeMean_differenceMean, analyzeMin_differenceOverall, analyzeMin_levelOverall, analyzeNoise_floor_countTotal,
	analyzeNoise_floorOverall, analyzeNumber_of_samplesTotal, analyzePeak_countTotal, analyzePeak_levelOverall, analyzeRMS_differenceOverall,
	analyzeRMS_levelOverall, analyzeRMS_peakOverall, analyzeRMS_troughOverall, analyzeSpectralCentroidMean, analyzeSpectralCrestMean,
	analyzeSpectralDecreaseMean, analyzeSpectralEntropyMean, analyzeSpectralFlatnessMean, analyzeSpectralFluxMean, analyzeSpectralKurtosisMean,
	analyzeSpectralMeanMean, analyzeSpectralRolloffMean, analyzeSpectralSkewnessMean, analyzeSpectralSlopeMean, analyzeSpectralSpreadMean,
	analyzeSpectralVarianceMean, analyzeTruePeakOverall, analyzeZero_crossings_rateOverall, analyzeZero_crossingsTotal)
from tests.dataSamples.expected import expectedFilename
from typing import TYPE_CHECKING
import os
import pytest

if TYPE_CHECKING:
	from pathlib import Path

pytestmark: pytest.MarkDecorator = pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason='Skipped in GitHub Actions')

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float | None, expected: float | None) -> None:
	assert actual == pytest.approx(expected), f'{analyzer}({pathFilename.name}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expected', [expectedFilename['analyzeAbs_Peak_countTotal']])
def test_analyzeAbs_Peak_countTotal(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeAbs_Peak_countTotal(pathFilename)
	_standardizedEqualScalars('analyzeAbs_Peak_countTotal', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeBit_depthMean']])
def test_analyzeBit_depthMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeBit_depthMean(pathFilename)
	_standardizedEqualScalars('analyzeBit_depthMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeCrest_factorMean']])
def test_analyzeCrest_factorMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeCrest_factorMean(pathFilename)
	_standardizedEqualScalars('analyzeCrest_factorMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeDC_offsetMean']])
def test_analyzeDC_offsetMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeDC_offsetMean(pathFilename)
	_standardizedEqualScalars('analyzeDC_offsetMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeDynamic_rangeOverall']])
def test_analyzeDynamic_rangeOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeDynamic_rangeOverall(pathFilename)
	_standardizedEqualScalars('analyzeDynamic_rangeOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeEntropyMean']])
def test_analyzeEntropyMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeEntropyMean(pathFilename)
	_standardizedEqualScalars('analyzeEntropyMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeFlat_factorMean']])
def test_analyzeFlat_factorMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeFlat_factorMean(pathFilename)
	_standardizedEqualScalars('analyzeFlat_factorMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLRAOverall']])
def test_analyzeLRAOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLRAOverall(pathFilename)
	_standardizedEqualScalars('analyzeLRAOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLUFSIntegratedOverall']])
def test_analyzeLUFSIntegratedOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLUFSIntegratedOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSIntegratedOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLUFSMomentaryOverall']])
def test_analyzeLUFSMomentaryOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLUFSMomentaryOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSMomentaryOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLUFSShortTermOverall']])
def test_analyzeLUFSShortTermOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLUFSShortTermOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSShortTermOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLUFShighOverall']])
def test_analyzeLUFShighOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLUFShighOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFShighOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeLUFSlowOverall']])
def test_analyzeLUFSlowOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeLUFSlowOverall(pathFilename)
	_standardizedEqualScalars('analyzeLUFSlowOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeMax_differenceOverall']])
def test_analyzeMax_differenceOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeMax_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeMax_differenceOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeMax_levelOverall']])
def test_analyzeMax_levelOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeMax_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeMax_levelOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeMean_differenceMean']])
def test_analyzeMean_differenceMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeMean_differenceMean(pathFilename)
	_standardizedEqualScalars('analyzeMean_differenceMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeMin_differenceOverall']])
def test_analyzeMin_differenceOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeMin_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeMin_differenceOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeMin_levelOverall']])
def test_analyzeMin_levelOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeMin_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeMin_levelOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeNoise_floor_countTotal']])
def test_analyzeNoise_floor_countTotal(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeNoise_floor_countTotal(pathFilename)
	_standardizedEqualScalars('analyzeNoise_floor_countTotal', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeNoise_floorOverall']])
def test_analyzeNoise_floorOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeNoise_floorOverall(pathFilename)
	_standardizedEqualScalars('analyzeNoise_floorOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeNumber_of_samplesTotal']])
def test_analyzeNumber_of_samplesTotal(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeNumber_of_samplesTotal(pathFilename)
	_standardizedEqualScalars('analyzeNumber_of_samplesTotal', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzePeak_countTotal']])
def test_analyzePeak_countTotal(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzePeak_countTotal(pathFilename)
	_standardizedEqualScalars('analyzePeak_countTotal', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzePeak_levelOverall']])
def test_analyzePeak_levelOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzePeak_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzePeak_levelOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeRMS_differenceOverall']])
def test_analyzeRMS_differenceOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeRMS_differenceOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_differenceOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeRMS_levelOverall']])
def test_analyzeRMS_levelOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeRMS_levelOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_levelOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeRMS_peakOverall']])
def test_analyzeRMS_peakOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeRMS_peakOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_peakOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeRMS_troughOverall']])
def test_analyzeRMS_troughOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeRMS_troughOverall(pathFilename)
	_standardizedEqualScalars('analyzeRMS_troughOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralCentroidMean']])
def test_analyzeSpectralCentroidMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralCentroidMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralCentroidMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralCrestMean']])
def test_analyzeSpectralCrestMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralCrestMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralCrestMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralDecreaseMean']])
def test_analyzeSpectralDecreaseMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralDecreaseMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralDecreaseMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralEntropyMean']])
def test_analyzeSpectralEntropyMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralEntropyMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralEntropyMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralFlatnessMean']])
def test_analyzeSpectralFlatnessMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralFlatnessMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralFlatnessMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralFluxMean']])
def test_analyzeSpectralFluxMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralFluxMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralFluxMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralKurtosisMean']])
def test_analyzeSpectralKurtosisMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralKurtosisMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralKurtosisMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralMeanMean']])
def test_analyzeSpectralMeanMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralMeanMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralMeanMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralRolloffMean']])
def test_analyzeSpectralRolloffMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralRolloffMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralRolloffMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralSkewnessMean']])
def test_analyzeSpectralSkewnessMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralSkewnessMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralSkewnessMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralSlopeMean']])
def test_analyzeSpectralSlopeMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralSlopeMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralSlopeMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralSpreadMean']])
def test_analyzeSpectralSpreadMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralSpreadMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralSpreadMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeSpectralVarianceMean']])
def test_analyzeSpectralVarianceMean(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeSpectralVarianceMean(pathFilename)
	_standardizedEqualScalars('analyzeSpectralVarianceMean', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeTruePeakOverall']])
def test_analyzeTruePeakOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeTruePeakOverall(pathFilename)
	_standardizedEqualScalars('analyzeTruePeakOverall', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeZero_crossingsTotal']])
def test_analyzeZero_crossingsTotal(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeZero_crossingsTotal(pathFilename)
	_standardizedEqualScalars('analyzeZero_crossingsTotal', pathFilename, actual, expected[pathFilename.name])

@pytest.mark.parametrize('expected', [expectedFilename['analyzeZero_crossings_rateOverall']])
def test_analyzeZero_crossings_rateOverall(pathFilename: Path, expected: dict[str, float | None]) -> None:
	actual = analyzeZero_crossings_rateOverall(pathFilename)
	_standardizedEqualScalars('analyzeZero_crossings_rateOverall', pathFilename, actual, expected[pathFilename.name])
