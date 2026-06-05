from __future__ import annotations

from analyzeAudio import analyzersUseTensorSpectrogram, audioContests
from typing import TYPE_CHECKING
import math
import pytest

if TYPE_CHECKING:
	import torch

@pytest.mark.parametrize(
	('stringAspectName', 'analyzerName'),
	[('SpectralConvergenceLoss', 'analyzeSpectralConvergenceLoss'), ('STFTMagnitudeLoss', 'analyzeSTFTMagnitudeLoss')],
)
def test_auraloss_tensor_spectrogram_losses_return_registered_floats(
	tensorSpectrogramMagnitudeAuralossCase: tuple[torch.Tensor, torch.Tensor], stringAspectName: str, analyzerName: str
) -> None:
	tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta = tensorSpectrogramMagnitudeAuralossCase
	analyzer = getattr(analyzersUseTensorSpectrogram, analyzerName)
	valueLoss = analyzer(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)
	assert isinstance(valueLoss, float), (
		f'{analyzerName} returned {type(valueLoss).__name__}, expected float for aspect {stringAspectName}.'
	)
	assert math.isfinite(valueLoss), f'{analyzerName} returned non-finite value {valueLoss} for aspect {stringAspectName}.'
	assert stringAspectName in audioContests, (
		f'audioContests did not register {stringAspectName}; available keys do not include the expected tensor-spectrogram auraloss aspect name.'
	)
	assert audioContests[stringAspectName]['analyzer'] is analyzer, (
		f'audioContests[{stringAspectName!r}] registered {audioContests[stringAspectName]["analyzer"]}, expected {analyzer}.'
	)

@pytest.mark.parametrize(
	('stringCaseName', 'valueIsIdenticalInput'), [('identicalMagnitudeSpectrogram', True), ('distinctMagnitudeSpectrogram', False)]
)
def test_analyze_l1_frequency_loss_returns_registered_bounded_similarity(
	tensorSpectrogramMagnitudeAuralossCase: tuple[torch.Tensor, torch.Tensor], stringCaseName: str, valueIsIdenticalInput: bool
) -> None:
	tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta = tensorSpectrogramMagnitudeAuralossCase
	tensorSpectrogramMagnitudeTarget = tensorSpectrogramMagnitudeAlfa if valueIsIdenticalInput else tensorSpectrogramMagnitudeBeta
	valueLoss = analyzersUseTensorSpectrogram.analyzeL1FrequencyLoss(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeTarget)
	assert isinstance(valueLoss, float), f'analyzeL1FrequencyLoss returned {type(valueLoss).__name__}, expected float for {stringCaseName}.'
	assert math.isfinite(valueLoss), f'analyzeL1FrequencyLoss returned non-finite value {valueLoss} for {stringCaseName}.'
	assert 0 < valueLoss <= 100, (
		f'analyzeL1FrequencyLoss returned {valueLoss}, expected bounded similarity in interval (0, 100] for {stringCaseName}.'
	)
	if valueIsIdenticalInput:
		assert valueLoss == 100.0, (
			f'analyzeL1FrequencyLoss returned {valueLoss}, expected 100.0 for identical spectrogram magnitudes in {stringCaseName}.'
		)
	else:
		assert valueLoss < 100.0, (
			f'analyzeL1FrequencyLoss returned {valueLoss}, expected value strictly below 100.0 for non-identical spectrogram magnitudes in {stringCaseName}.'
		)
	assert 'L1FrequencyLoss' in audioContests, (
		'audioContests did not register L1FrequencyLoss; available keys do not include the expected tensor-spectrogram aspect name.'
	)
	assert audioContests['L1FrequencyLoss']['analyzer'] is analyzersUseTensorSpectrogram.analyzeL1FrequencyLoss, (
		f"audioContests['L1FrequencyLoss'] registered {audioContests['L1FrequencyLoss']['analyzer']}, expected {analyzersUseTensorSpectrogram.analyzeL1FrequencyLoss}."
	)
