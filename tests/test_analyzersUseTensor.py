from __future__ import annotations

from analyzeAudio.analyzersUseTensor import analyzeDNSMOSMean, analyzeNISQAMean, analyzeSRMRMean
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import AspectTensor

@pytest.mark.parametrize('expectedAspect', ['analyzeDNSMOSMean'], indirect=True)
def test_analyzeDNSMOSMean(aspectTensor: AspectTensor, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	if aspectTensor.pathFilename.name == 'ch2_44100_29s_LUFS23_10000Hz.wav':
		approx_rel = 1e-3
	actual = analyzeDNSMOSMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeDNSMOSMean', aspectTensor.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNISQAMean'], indirect=True)
def test_analyzeNISQAMean(aspectTensor: AspectTensor, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	approx_rel = 1e-5
	actual = analyzeNISQAMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNISQAMean', aspectTensor.pathFilename)

@pytest.mark.parametrize('pytorchOnCPU', [True])
@pytest.mark.parametrize('expectedAspect', ['analyzeSRMRMean'], indirect=True)
def test_analyzeSRMRMean(aspectTensor: AspectTensor, pytorchOnCPU: bool, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSRMRMean(aspectTensor.tensorAudio, aspectTensor.sampleRate, pytorchOnCPU)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSRMRMean', aspectTensor.pathFilename, pytorchOnCPU)
