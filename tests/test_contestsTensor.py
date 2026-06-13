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
	parameters: str = (
		f'pathFilenameAlfa={paths.pathFilenameAlfa.name!r}, pathFilenameBeta={paths.pathFilenameBeta.name!r}, sampleRate={sampleRate!r}'
	)
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedContest', ['analyzeLogWMSEMean'], indirect=True)
def test_analyzeLogWMSEMean(contestTensor: ContestTensor, tensorAudioMixture: Tensor, expectedContest: float) -> None:
	actual = analyzeLogWMSEMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, tensorAudioMixture, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeLogWMSEMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeL1SNRMean'], indirect=True)
def test_analyzeL1SNRMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeL1SNRMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeL1SNRMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeL1SNRDBMean'], indirect=True)
def test_analyzeL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeL1SNRDBMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMultiL1SNRDBMean'], indirect=True)
def test_analyzeMultiL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeMultiL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeMultiL1SNRDBMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTL1SNRDBMean'], indirect=True)
def test_analyzeSTFTL1SNRDBMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSTFTL1SNRDBMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSTFTL1SNRDBMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeDCLossMean'], indirect=True)
def test_analyzeDCLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeDCLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeDCLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeESRLossMean'], indirect=True)
def test_analyzeESRLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeESRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeESRLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeLogCoshLossMean'], indirect=True)
def test_analyzeLogCoshLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeLogCoshLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeLogCoshLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSNRLossMean'], indirect=True)
def test_analyzeSNRLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSNRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSNRLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSISDRLossMean'], indirect=True)
def test_analyzeSISDRLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSISDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSISDRLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSDSDRLossMean'], indirect=True)
def test_analyzeSDSDRLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSDSDRLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSDSDRLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeSTFTLossMean'], indirect=True)
def test_analyzeSTFTLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars('analyzeSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMelSTFTLossMean'], indirect=True)
def test_analyzeMelSTFTLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeMelSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeMelSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeChromaSTFTLossMean'], indirect=True)
def test_analyzeChromaSTFTLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeChromaSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta, contestTensor.sampleRateAlfa)
	_standardizedEqualScalars('analyzeChromaSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa)

@pytest.mark.parametrize('expectedContest', ['analyzeMultiResolutionSTFTLossMean'], indirect=True)
def test_analyzeMultiResolutionSTFTLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeMultiResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeMultiResolutionSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('randomSeed', [1597])
@pytest.mark.parametrize('expectedContest', ['analyzeRandomResolutionSTFTLossMean'], indirect=True)
def test_analyzeRandomResolutionSTFTLossMean(contestTensor: ContestTensor, randomSeed: int, expectedContest: float) -> None:
	numpy.random.seed(randomSeed)  # noqa: NPY002
	actual = analyzeRandomResolutionSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeRandomResolutionSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa
	)

@pytest.mark.parametrize('expectedContest', ['analyzeSumAndDifferenceSTFTLossMean'], indirect=True)
def test_analyzeSumAndDifferenceSTFTLossMean(contestTensor: ContestTensor, expectedContest: float) -> None:
	actual = analyzeSumAndDifferenceSTFTLossMean(contestTensor.tensorAlfa, contestTensor.tensorBeta)
	_standardizedEqualScalars(
		'analyzeSumAndDifferenceSTFTLossMean', contestTensor.paths, actual, expectedContest, contestTensor.sampleRateAlfa
	)
