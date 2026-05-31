from __future__ import annotations

from analyzeAudio import analyzersUseTensor, audioAspects
from typing import TYPE_CHECKING
import math
import pytest

if TYPE_CHECKING:
	import torch

@pytest.mark.parametrize(
	('stringAspectName', 'analyzerName', 'boolNeedsSampleRate'),
	[
		('DCLoss', 'analyzeDCLoss', False),
		('ESRLoss', 'analyzeESRLoss', False),
		('LogCoshLoss', 'analyzeLogCoshLoss', False),
		('SNRLoss', 'analyzeSNRLoss', False),
		('SISDRLoss', 'analyzeSISDRLoss', False),
		('SDSDRLoss', 'analyzeSDSDRLoss', False),
		('STFTLoss', 'analyzeSTFTLoss', False),
		('MelSTFTLoss', 'analyzeMelSTFTLoss', True),
		('ChromaSTFTLoss', 'analyzeChromaSTFTLoss', True),
		('MultiResolutionSTFTLoss', 'analyzeMultiResolutionSTFTLoss', False),
		('RandomResolutionSTFTLoss', 'analyzeRandomResolutionSTFTLoss', False),
		('SumAndDifferenceSTFTLoss', 'analyzeSumAndDifferenceSTFTLoss', False),
	],
)
def test_auraloss_waveform_losses_return_registered_floats(
	tensorAudioAuralossCase: tuple[torch.Tensor, torch.Tensor, int], stringAspectName: str, analyzerName: str, boolNeedsSampleRate: bool,
) -> None:
	tensorAudioAlfa, tensorAudioBeta, sampleRate = tensorAudioAuralossCase
	analyzer = getattr(analyzersUseTensor, analyzerName)
	valueLoss = analyzer(tensorAudioAlfa, tensorAudioBeta, sampleRate) if boolNeedsSampleRate else analyzer(tensorAudioAlfa, tensorAudioBeta)
	assert isinstance(valueLoss, float), (
		f"{analyzerName} returned {type(valueLoss).__name__}, expected float for aspect {stringAspectName}."
	)
	assert math.isfinite(valueLoss), (
		f"{analyzerName} returned non-finite value {valueLoss} for aspect {stringAspectName}."
	)
	assert stringAspectName in audioAspects, (
		f"audioAspects did not register {stringAspectName}; available keys do not include the expected auraloss aspect name."
	)
	assert audioAspects[stringAspectName]['analyzer'] is analyzer, (
		f"audioAspects[{stringAspectName!r}] registered {audioAspects[stringAspectName]['analyzer']}, expected {analyzer}."
	)
