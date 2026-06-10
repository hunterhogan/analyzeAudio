from __future__ import annotations

from analyzeAudio.contestsTensor import (
	analyzeChromaSTFTLossMean, analyzeDCLossMean, analyzeESRLossMean, analyzeL1SNRDBMean, analyzeL1SNRMean, analyzeLogCoshLossMean,
	analyzeLogWMSEMean, analyzeMelSTFTLossMean, analyzeMultiL1SNRDBMean, analyzeMultiResolutionSTFTLossMean,
	analyzeRandomResolutionSTFTLossMean, analyzeSDSDRLossMean, analyzeSISDRLossMean, analyzeSNRLossMean, analyzeSTFTL1SNRDBMean,
	analyzeSTFTLossMean, analyzeSumAndDifferenceSTFTLossMean)
from typing import TYPE_CHECKING
import numpy
import pytest

if TYPE_CHECKING:
	from tests import ContestFilename, ContestTensor
	from torch import Tensor

def _standardizedEqualScalars(analyzer: str, paths: ContestFilename, actual: float, expected: float, sampleRate: int) -> None:
	parameters = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedContestTensor', ['analyzeLogWMSEMean'], indirect=True)
def test_analyzeLogWMSEMean(contestTensor: ContestTensor, tensorAudioMixture: Tensor, expectedContestTensor: float) -> None:
	actual = analyzeLogWMSEMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, tensorAudioMixture, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeLogWMSEMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeL1SNRMean'], indirect=True)
def test_analyzeL1SNRMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeL1SNRMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeL1SNRMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeL1SNRDBMean'], indirect=True)
def test_analyzeL1SNRDBMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeL1SNRDBMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeMultiL1SNRDBMean'], indirect=True)
def test_analyzeMultiL1SNRDBMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeMultiL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeMultiL1SNRDBMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSTFTL1SNRDBMean'], indirect=True)
def test_analyzeSTFTL1SNRDBMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSTFTL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSTFTL1SNRDBMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeDCLossMean'], indirect=True)
def test_analyzeDCLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeDCLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeDCLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeESRLossMean'], indirect=True)
def test_analyzeESRLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeESRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeESRLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeLogCoshLossMean'], indirect=True)
def test_analyzeLogCoshLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeLogCoshLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeLogCoshLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSNRLossMean'], indirect=True)
def test_analyzeSNRLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSNRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSNRLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSISDRLossMean'], indirect=True)
def test_analyzeSISDRLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSISDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSISDRLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSDSDRLossMean'], indirect=True)
def test_analyzeSDSDRLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSDSDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSDSDRLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSTFTLossMean'], indirect=True)
def test_analyzeSTFTLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeMelSTFTLossMean'], indirect=True)
def test_analyzeMelSTFTLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeMelSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeMelSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeChromaSTFTLossMean'], indirect=True)
def test_analyzeChromaSTFTLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeChromaSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeChromaSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeMultiResolutionSTFTLossMean'], indirect=True)
def test_analyzeMultiResolutionSTFTLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeMultiResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeMultiResolutionSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('randomSeed', [1597])
@pytest.mark.parametrize('expectedContestTensor', ['analyzeRandomResolutionSTFTLossMean'], indirect=True)
def test_analyzeRandomResolutionSTFTLossMean(contestTensor: ContestTensor, randomSeed: int, expectedContestTensor: float) -> None:
	numpy.random.seed(randomSeed)  # noqa: NPY002
	actual = analyzeRandomResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeRandomResolutionSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSumAndDifferenceSTFTLossMean'], indirect=True)
def test_analyzeSumAndDifferenceSTFTLossMean(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSumAndDifferenceSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeSumAndDifferenceSTFTLossMean', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)
