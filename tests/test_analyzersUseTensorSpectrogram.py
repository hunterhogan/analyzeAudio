from __future__ import annotations

from analyzeAudio import analyzersUseTensorSpectrogram, audioAspects
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
	assert stringAspectName in audioAspects, (
		f'audioAspects did not register {stringAspectName}; available keys do not include the expected tensor-spectrogram auraloss aspect name.'
	)
	assert audioAspects[stringAspectName]['analyzer'] is analyzer, (
		f'audioAspects[{stringAspectName!r}] registered {audioAspects[stringAspectName]["analyzer"]}, expected {analyzer}.'
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
	assert 'L1FrequencyLoss' in audioAspects, (
		'audioAspects did not register L1FrequencyLoss; available keys do not include the expected tensor-spectrogram aspect name.'
	)
	assert audioAspects['L1FrequencyLoss']['analyzer'] is analyzersUseTensorSpectrogram.analyzeL1FrequencyLoss, (
		f"audioAspects['L1FrequencyLoss'] registered {audioAspects['L1FrequencyLoss']['analyzer']}, expected {analyzersUseTensorSpectrogram.analyzeL1FrequencyLoss}."
	)
