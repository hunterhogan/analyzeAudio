from __future__ import annotations

from analyzeAudio.analyzersUseTensor import analyzeDNSMOSMean, analyzeNISQAMean, analyzeSRMRMean
from tests.conftest import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from tests import AspectTensor

@pytest.mark.parametrize('expectedAspect', ['analyzeDNSMOSMean'], indirect=True)
def test_analyzeDNSMOSMean(aspectTensor: AspectTensor, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	if aspectTensor.pathFilename.name == 'ch2_44100_29s_LUFS23_10000Hz.wav':
		pytest.skip('Values differ in GitHub Actions vs. locally by more than the test tolerances.')
	actual = analyzeDNSMOSMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeDNSMOSMean', aspectTensor.pathFilename)

@pytest.mark.parametrize('expectedAspect', ['analyzeNISQAMean'], indirect=True)
def test_analyzeNISQAMean(aspectTensor: AspectTensor, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeNISQAMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeNISQAMean', aspectTensor.pathFilename)

@pytest.mark.parametrize('pytorchOnCPU', [True])
@pytest.mark.parametrize('expectedAspect', ['analyzeSRMRMean'], indirect=True)
def test_analyzeSRMRMean(aspectTensor: AspectTensor, pytorchOnCPU: bool, expectedAspect: float | None, approx_rel: float, approx_abs: float) -> None:
	actual = analyzeSRMRMean(aspectTensor.tensorAudio, aspectTensor.sampleRate, pytorchOnCPU)
	assert_approx(actual, expectedAspect, approx_rel, approx_abs, 'analyzeSRMRMean', aspectTensor.pathFilename, pytorchOnCPU)
