from __future__ import annotations

from analyzeAudio.analyzersUseTensor import analyzeDNSMOSMean, analyzeNISQAMean, analyzeSRMRMean
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import TensorAndData

@pytest.mark.parametrize('expectedAspect', ['analyzeDNSMOSMean'], indirect=True)
def test_analyzeDNSMOSMean(tensorAndData: TensorAndData, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	if tensorAndData.pathFilename.name == 'ch2_44100_29s_LUFS23_10000Hz.wav':
		approx_rel = 1e-3
	actual = analyzeDNSMOSMean(tensorAndData.tensorAudio, tensorAndData.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeDNSMOSMean', tensorAndData.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNISQAMean'], indirect=True)
def test_analyzeNISQAMean(tensorAndData: TensorAndData, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	actual = analyzeNISQAMean(tensorAndData.tensorAudio, tensorAndData.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNISQAMean', tensorAndData.pathFilename)

@pytest.mark.parametrize('pytorchOnCPU', [True])
@pytest.mark.parametrize('expectedAspect', ['analyzeSRMRMean'], indirect=True)
def test_analyzeSRMRMean(tensorAndData: TensorAndData, pytorchOnCPU: bool, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSRMRMean(tensorAndData.tensorAudio, tensorAndData.sampleRate, pytorchOnCPU)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSRMRMean', tensorAndData.pathFilename, pytorchOnCPU)
