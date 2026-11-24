"""Expected values are the accepted recordings in `tests.dataSamples.expected.expectedTensor`."""
from __future__ import annotations

from analyzeAudio.analyzeTensor import analyzeDNSMOSMean, analyzeNISQAMean, analyzeSRMRMean
from importlib.util import find_spec
from tests import assert_approx
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from pathlib import Path
	from torch import Tensor

@pytest.mark.parametrize('analyzer', (
	pytest.param(analyzeDNSMOSMean, id='analyzeDNSMOSMean', marks=pytest.mark.skipif(
		find_spec('onnxruntime') is None, reason='DNSMOS runs in onnxruntime, which publishes no wheel for this interpreter.')),
	pytest.param(analyzeNISQAMean, id='analyzeNISQAMean'),
))
def test_aspectTensor(analyzer: CallableFunction[[Tensor, int], float | None], pathFilename: Path, tensorAudio: Tensor, sampleRate: int, expectedAspect: float | None) -> None:
	"""Both analyzers keep the relative tolerance 1e-5 with which the maintainer accepted their recordings."""
	assert_approx(analyzer(tensorAudio, sampleRate), expectedAspect, 1e-5, 1e-12, analyzer.__name__, pathFilename.name, sampleRate)

@pytest.mark.parametrize('pytorchOnCPU', (True,))
@pytest.mark.parametrize('analyzer', (analyzeSRMRMean,))
def test_aspectTensorPytorchOnCPU(analyzer: CallableFunction[[Tensor, int, bool], float], pathFilename: Path, tensorAudio: Tensor, sampleRate: int, pytorchOnCPU: bool, expectedAspect: float | None) -> None:
	assert_approx(analyzer(tensorAudio, sampleRate, pytorchOnCPU), expectedAspect, 1e-6, 1e-12, analyzer.__name__, pathFilename.name, sampleRate, pytorchOnCPU)
