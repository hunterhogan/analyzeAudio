# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio import truncateTensors
from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from auraloss import freq
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	from torch import nn, Tensor

def _analyzeLoss(aspect: nn.Module, tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""I use this function to evaluate one spectrogram-loss module on two spectrogram tensors.

	(AI generated docstring)

	I use this function to compute one scalar loss value from `tensorSpectrogramMagnitudeAlfa`
	and `tensorSpectrogramMagnitudeBeta` with `aspect`.

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
		Loss value produced by `aspect`.

	Input Alignment
	-----------------
	frameCountShared : int
		`frameCountShared` is `min(tensorSpectrogramMagnitudeAlfa.shape[-1],
		tensorSpectrogramMagnitudeBeta.shape[-1])`. The function slices both values along axis `-1`
		with `0:frameCountShared` before evaluating `aspect`.

	References
	----------
	[1] PyTorch `torch.nn.Module`
		https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	"""
	return float(aspect(*truncateTensors([tensorSpectrogramMagnitudeBeta, tensorSpectrogramMagnitudeAlfa])).item())

aspectName = 'SpectralConvergenceLoss'
@registrationAudioAspect(aspectName)
def analyzeSpectralConvergenceLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""Compute spectral convergence loss for two spectrogram magnitude `Tensor` values.

	(AI generated docstring)

	You can use this function to measure spectral-convergence distance between
	`tensorSpectrogramMagnitudeAlfa` and `tensorSpectrogramMagnitudeBeta` with
	`auraloss.freq.SpectralConvergenceLoss` [1].

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

	Mathematics
	-----------
	spectral convergence objective : equation
	```
		Let Â ≜ `tensorSpectrogramMagnitudeAlfa`
			B̂ ≜ `tensorSpectrogramMagnitudeBeta`
			[Â′, B̂′] ≜ `truncateTensors([Â, B̂])`

		valueLoss = ∥B̂′ − Â′∥_F / (∥Â′∥_F + ε)
	```

	References
	----------
	[1] auraloss `freq.SpectralConvergenceLoss`
		https://github.com/csteinmetz1/auraloss

	"""
	return _analyzeLoss(freq.SpectralConvergenceLoss(), tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)

aspectName = 'STFTMagnitudeLoss'
@registrationAudioAspect(aspectName)
def analyzeSTFTMagnitudeLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor, **keywordArguments: Any) -> float:
	"""Compute STFT magnitude loss for two spectrogram magnitude `Tensor` values.

	(AI generated docstring)

	You can use this function to measure magnitude-distance between
	`tensorSpectrogramMagnitudeAlfa` and `tensorSpectrogramMagnitudeBeta` with
	`auraloss.freq.STFTMagnitudeLoss` [1].

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

	Mathematics
	-----------
	STFT magnitude objective : equation
	```
		Let Â ≜ `tensorSpectrogramMagnitudeAlfa`
			B̂ ≜ `tensorSpectrogramMagnitudeBeta`
			θ ≜ `keywordArguments`
			[Â′, B̂′] ≜ `truncateTensors([Â, B̂])`
			ℒ_STFTMag ≜ `freq.STFTMagnitudeLoss(**θ)`

		valueLoss = ℒ_STFTMag(B̂′, Â′)
	```

	References
	----------
	[1] auraloss `freq.STFTMagnitudeLoss`
		https://github.com/csteinmetz1/auraloss

	"""
	return _analyzeLoss(freq.STFTMagnitudeLoss(**keywordArguments), tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)

aspectName = 'L1FrequencyLoss'
@registrationAudioAspect(aspectName)
def analyzeL1FrequencyLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""Compute L1 frequency loss for two spectrogram magnitude `Tensor` values.

	Parameters
	----------
	tensorSpectrogramMagnitudeAlfa : Tensor
		Reference spectrogram magnitude `Tensor`.
	tensorSpectrogramMagnitudeBeta : Tensor
		Compared spectrogram magnitude `Tensor`.

	Returns
	-------
	valueLoss : float
		L1 frequency loss value.

	Mathematics
	-----------
	L1 frequency loss : equation
	```
		Let 𝒮 ≜ complex-valued spectrogram
			ℒ ≜ L1 frequency loss

		ℒ = ∥|𝒮(ŵ)| − |𝒮(w)|∥₁
		y = 100 / (1 + (λ * ℒ))

		where λ = a scaling factor.
	```
	"""
	keywordArguments = dict(log=False, distance="L1", reduction="sum")  # noqa: C408
	L1: float = analyzeSTFTMagnitudeLoss(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta, **keywordArguments)
	λ = 10
	return 100 / (1 + (λ * L1))
