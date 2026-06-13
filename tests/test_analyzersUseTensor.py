from __future__ import annotations

from analyzeAudio.analyzersUseTensor import analyzeDNSMOSMean, analyzeNISQAMean, analyzeSRMRMean
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectTensor

def _standardizedEqualScalars(
	analyzer: str, pathFilename: Path, actual: float | None, expected: float | None, pytorchOnCPU: bool | None = None
) -> None:
	parameters: str = f'pathFilename={pathFilename.name!r}'
	if pytorchOnCPU is not None:
		parameters = f'{parameters}, pytorchOnCPU={pytorchOnCPU!r}'
	assert actual == pytest.approx(expected, rel=1e-4, abs=1e-6), f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('expectedAspect', ['analyzeDNSMOSMean'], indirect=True)
def test_analyzeDNSMOSMean(aspectTensor: AspectTensor, expectedAspect: float | None) -> None:
	if aspectTensor.pathFilename.name == 'ch2_44100_29s_LUFS23_10000Hz.wav':
		pytest.skip('Values differ in GitHub Actions vs. locally by more than the test tolerances.')
	actual = analyzeDNSMOSMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	_standardizedEqualScalars('analyzeDNSMOSMean', aspectTensor.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('expectedAspect', ['analyzeNISQAMean'], indirect=True)
def test_analyzeNISQAMean(aspectTensor: AspectTensor, expectedAspect: float | None) -> None:
	actual = analyzeNISQAMean(aspectTensor.tensorAudio, aspectTensor.sampleRate)
	_standardizedEqualScalars('analyzeNISQAMean', aspectTensor.pathFilename, actual, expectedAspect)

@pytest.mark.parametrize('pytorchOnCPU', [True])
@pytest.mark.parametrize('expectedAspect', ['analyzeSRMRMean'], indirect=True)
def test_analyzeSRMRMean(aspectTensor: AspectTensor, pytorchOnCPU: bool, expectedAspect: float | None) -> None:
	actual = analyzeSRMRMean(aspectTensor.tensorAudio, aspectTensor.sampleRate, pytorchOnCPU)
	_standardizedEqualScalars('analyzeSRMRMean', aspectTensor.pathFilename, actual, expectedAspect, pytorchOnCPU)
