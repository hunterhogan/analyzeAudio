from __future__ import annotations

from analyzeAudio.contestsTensor import (
	analyzeChromaSTFTLoss, analyzeDCLoss, analyzeESRLoss, analyzeL1SNRDBMean, analyzeL1SNRMean, analyzeLogCoshLoss, analyzeLogWMSEMean,
	analyzeMelSTFTLoss, analyzeMultiL1SNRDBMean, analyzeMultiResolutionSTFTLoss, analyzeRandomResolutionSTFTLoss, analyzeSDSDRLoss,
	analyzeSISDRLoss, analyzeSNRLoss, analyzeSTFTL1SNRDBMean, analyzeSTFTLoss, analyzeSumAndDifferenceSTFTLoss)
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
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8), f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

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

@pytest.mark.parametrize('expectedContestTensor', ['analyzeDCLoss'], indirect=True)
def test_analyzeDCLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeDCLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeDCLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeESRLoss'], indirect=True)
def test_analyzeESRLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeESRLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeESRLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeLogCoshLoss'], indirect=True)
def test_analyzeLogCoshLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeLogCoshLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeLogCoshLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSNRLoss'], indirect=True)
def test_analyzeSNRLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSNRLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSNRLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSISDRLoss'], indirect=True)
def test_analyzeSISDRLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSISDRLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSISDRLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSDSDRLoss'], indirect=True)
def test_analyzeSDSDRLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSDSDRLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSDSDRLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSTFTLoss'], indirect=True)
def test_analyzeSTFTLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeMelSTFTLoss'], indirect=True)
def test_analyzeMelSTFTLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeMelSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeMelSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeChromaSTFTLoss'], indirect=True)
def test_analyzeChromaSTFTLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeChromaSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeChromaSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeMultiResolutionSTFTLoss'], indirect=True)
def test_analyzeMultiResolutionSTFTLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeMultiResolutionSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeMultiResolutionSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('randomSeed', [1597])
@pytest.mark.parametrize('expectedContestTensor', ['analyzeRandomResolutionSTFTLoss'], indirect=True)
def test_analyzeRandomResolutionSTFTLoss(contestTensor: ContestTensor, randomSeed: int, expectedContestTensor: float) -> None:
	numpy.random.seed(randomSeed)  # noqa: NPY002
	actual = analyzeRandomResolutionSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeRandomResolutionSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContestTensor', ['analyzeSumAndDifferenceSTFTLoss'], indirect=True)
def test_analyzeSumAndDifferenceSTFTLoss(contestTensor: ContestTensor, expectedContestTensor: float) -> None:
	actual = analyzeSumAndDifferenceSTFTLoss(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeSumAndDifferenceSTFTLoss', contestTensor.paths, actual, expectedContestTensor, contestTensor.sampleRateAlfa
	)
