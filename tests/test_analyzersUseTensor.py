from __future__ import annotations

from analyzeAudio.analyzersUseTensor import analyzeSRMRMean
from tests.dataSamples.expected import expectedTensor
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path
	from tests import AspectTensor

def _standardizedEqualScalars(analyzer: str, pathFilename: Path, actual: float, expected: float, pytorchOnCPU: bool) -> None:
	parameters = f'pathFilename={pathFilename.name!r}, pytorchOnCPU={pytorchOnCPU!r}'
	assert actual == pytest.approx(expected, rel=1e-5, abs=1e-8), f'{analyzer}({parameters}) = {actual!r}, but {expected = }.'  # pyright: ignore[reportUnknownMemberType]

@pytest.mark.parametrize('pytorchOnCPU', [True])
@pytest.mark.parametrize('expected', [expectedTensor['analyzeSRMRMean']])
def test_analyzeSRMRMean(aspectTensor: AspectTensor, pytorchOnCPU: bool, expected: dict[str, float]) -> None:
	actual = analyzeSRMRMean(aspectTensor.tensorAudio, aspectTensor.sampleRate, pytorchOnCPU)
	_standardizedEqualScalars('analyzeSRMRMean', aspectTensor.pathFilename, actual, expected[aspectTensor.pathFilename.name], pytorchOnCPU)
