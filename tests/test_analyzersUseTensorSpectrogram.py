from __future__ import annotations

from analyzeAudio import analyzersUseTensorSpectrogram, audioAspects
from typing import TYPE_CHECKING
import math
import pytest

if TYPE_CHECKING:
	import torch

@pytest.mark.parametrize(
	('stringAspectName', 'analyzerName'),
	[
		('SpectralConvergenceLoss', 'analyzeSpectralConvergenceLoss'),
		('STFTMagnitudeLoss', 'analyzeSTFTMagnitudeLoss'),
	],
)
def test_auraloss_tensor_spectrogram_losses_return_registered_floats(
	tensorSpectrogramMagnitudeAuralossCase: tuple[torch.Tensor, torch.Tensor], stringAspectName: str, analyzerName: str,
) -> None:
	tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta = tensorSpectrogramMagnitudeAuralossCase
	analyzer = getattr(analyzersUseTensorSpectrogram, analyzerName)
	valueLoss = analyzer(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)
	assert isinstance(valueLoss, float), (
		f"{analyzerName} returned {type(valueLoss).__name__}, expected float for aspect {stringAspectName}."
	)
	assert math.isfinite(valueLoss), (
		f"{analyzerName} returned non-finite value {valueLoss} for aspect {stringAspectName}."
	)
	assert stringAspectName in audioAspects, (
		f"audioAspects did not register {stringAspectName}; available keys do not include the expected tensor-spectrogram auraloss aspect name."
	)
	assert audioAspects[stringAspectName]['analyzer'] is analyzer, (
		f"audioAspects[{stringAspectName!r}] registered {audioAspects[stringAspectName]['analyzer']}, expected {analyzer}."
	)
