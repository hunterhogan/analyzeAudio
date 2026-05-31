# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from auraloss import freq
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from torch import nn, Tensor

def _analyzeLoss(aspect: nn.Module, tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""I use this shared adapter to evaluate one spectrogram-loss module on aligned frame counts.

	(AI generated docstring)

	I use this function to keep frame-alignment behavior consistent for multiple public analyzer functions.
	The function truncates both input spectrogram magnitudes to the same trailing time-frame length,
	then evaluates `aspect` and converts the scalar `Tensor` result to `float`.

	Parameters
	----------
	aspect : nn.Module
		Loss module instance that accepts two magnitude spectrogram `Tensor` values and returns a scalar
		loss `Tensor`.
	tensorSpectrogramMagnitudeAlfa : Tensor
		First spectrogram magnitude `Tensor`.
	tensorSpectrogramMagnitudeBeta : Tensor
		Second spectrogram magnitude `Tensor`.

	Returns
	-------
	valueLoss : float
		Loss value produced by `aspect` after frame-length alignment.

	Sequence Trimming
	-----------------
	frameCountShared : int
		`frameCountShared` is `min(tensorSpectrogramMagnitudeAlfa.shape[-1],
		tensorSpectrogramMagnitudeBeta.shape[-1])`. The function slices both values along axis `-1`
		with `0:frameCountShared` before computing the loss.

	References
	----------
	[1] PyTorch `torch.nn.Module`
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	"""
	truncate: int = min(tensorSpectrogramMagnitudeAlfa.shape[-1], tensorSpectrogramMagnitudeBeta.shape[-1])
	return float(aspect(tensorSpectrogramMagnitudeBeta[..., 0:truncate], tensorSpectrogramMagnitudeAlfa[..., 0:truncate]).item())

aspectName = 'SpectralConvergenceLoss'
@registrationAudioAspect(aspectName)
def analyzeSpectralConvergenceLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""Compute spectral convergence loss for two spectrogram magnitude `Tensor` values.

	(AI generated docstring)

	You can use this function to measure spectral-convergence distance between
	`tensorSpectrogramMagnitudeAlfa` and `tensorSpectrogramMagnitudeBeta` with
	`auraloss.freq.SpectralConvergenceLoss` [1]. The function aligns time-frame length by truncating
	both values to the shortest trailing dimension through `_analyzeLoss` [2], then returns one scalar
	`float`.

	Parameters
	----------
	tensorSpectrogramMagnitudeAlfa : Tensor
		Reference spectrogram magnitude `Tensor`.
	tensorSpectrogramMagnitudeBeta : Tensor
		Compared spectrogram magnitude `Tensor`.

	Returns
	-------
	valueLoss : float
		Spectral convergence loss value.

	See Also
	--------
	`analyzeSTFTMagnitudeLoss`
		Compute STFT-magnitude loss with configurable keyword arguments.

	References
	----------
	[1] auraloss `freq.SpectralConvergenceLoss`
		https://github.com/csteinmetz1/auraloss
	[2] `_analyzeLoss`

	"""
	return _analyzeLoss(freq.SpectralConvergenceLoss(), tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)

aspectName = 'STFTMagnitudeLoss'
@registrationAudioAspect(aspectName)
def analyzeSTFTMagnitudeLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor, **keywordArguments: Any) -> float:
	"""Compute STFT magnitude loss for two spectrogram magnitude `Tensor` values.

	(AI generated docstring)

	You can use this function to measure magnitude-distance between
	`tensorSpectrogramMagnitudeAlfa` and `tensorSpectrogramMagnitudeBeta` with
	`auraloss.freq.STFTMagnitudeLoss` [1]. The function forwards `keywordArguments` to
	`freq.STFTMagnitudeLoss`, aligns trailing frame length through `_analyzeLoss` [2], and returns one
	scalar `float`.

	Parameters
	----------
	tensorSpectrogramMagnitudeAlfa : Tensor
		Reference spectrogram magnitude `Tensor`.
	tensorSpectrogramMagnitudeBeta : Tensor
		Compared spectrogram magnitude `Tensor`.
	keywordArguments : Any
		Keyword argument mapping forwarded to `freq.STFTMagnitudeLoss`.

	Returns
	-------
	valueLoss : float
		STFT magnitude loss value.

	See Also
	--------
	`analyzeSpectralConvergenceLoss`
		Compute spectral convergence loss for the same input representation.

	References
	----------
	[1] auraloss `freq.STFTMagnitudeLoss`
		https://github.com/csteinmetz1/auraloss
	[2] `_analyzeLoss`

	"""
	return _analyzeLoss(freq.STFTMagnitudeLoss(**keywordArguments), tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)
