# ruff: noqa: D100 DOC201
from __future__ import annotations

from analyzeAudio import truncateTensors
from analyzeAudio.registry import registrationAudioContest
from auraloss import freq
from torchmetrics.functional.audio import complex_scale_invariant_signal_noise_ratio
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from torch import nn, Tensor
	from typing import Any

def _analyzeLoss(aspect: nn.Module, tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> Tensor:
	"""I use this function to evaluate one spectrogram-loss module on two spectrogram tensors.

	(AI generated docstring)

	I use this function to compute one tensor loss value from `tensorSpectrogramMagnitudeAlfa`
	and `tensorSpectrogramMagnitudeBeta` with `aspect`.

	Parameters
	----------
	aspect : nn.Module
		Loss module instance that accepts two magnitude spectrogram `Tensor` values and returns a loss `Tensor`.
	tensorSpectrogramMagnitudeAlfa : Tensor
		First spectrogram magnitude `Tensor`.
	tensorSpectrogramMagnitudeBeta : Tensor
		Second spectrogram magnitude `Tensor`.

	Returns
	-------
	tensorLoss : Tensor
		Loss tensor produced by `aspect`.

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
	return aspect(*truncateTensors([tensorSpectrogramMagnitudeBeta, tensorSpectrogramMagnitudeAlfa]))

def analyzeComplexScaleInvariantSignalNoiseRatio(
		tensorSpectrogramAlfa: Tensor, tensorSpectrogramBeta: Tensor, **keywordArguments: Any
) -> Tensor:
	"""Compute C-SI-SNR values for two complex spectrogram tensors."""
	preds, target = truncateTensors([tensorSpectrogramBeta, tensorSpectrogramAlfa])
	return complex_scale_invariant_signal_noise_ratio(preds, target, **keywordArguments)

@registrationAudioContest('C-SI-SNR mean')
def analyzeComplexScaleInvariantSignalNoiseRatioMean(
		tensorSpectrogramAlfa: Tensor, tensorSpectrogramBeta: Tensor, **keywordArguments: Any
) -> float:
	"""Contest 'C-SI-SNR mean': mean complex scale-invariant signal-to-noise ratio."""
	return float(analyzeComplexScaleInvariantSignalNoiseRatio(tensorSpectrogramAlfa, tensorSpectrogramBeta, **keywordArguments).mean().item())

def analyzeComplexScaleInvariantSignalNoiseRatioLoss(
		tensorSpectrogramAlfa: Tensor, tensorSpectrogramBeta: Tensor, **keywordArguments: Any
) -> Tensor:
	"""Compute negative C-SI-SNR values for loss minimization."""
	return -analyzeComplexScaleInvariantSignalNoiseRatio(tensorSpectrogramAlfa, tensorSpectrogramBeta, **keywordArguments)

@registrationAudioContest('C-SI-SNR loss mean')
def analyzeComplexScaleInvariantSignalNoiseRatioLossMean(
		tensorSpectrogramAlfa: Tensor, tensorSpectrogramBeta: Tensor, **keywordArguments: Any
) -> float:
	"""Contest 'C-SI-SNR loss mean': mean negative C-SI-SNR loss."""
	return float(analyzeComplexScaleInvariantSignalNoiseRatioLoss(tensorSpectrogramAlfa, tensorSpectrogramBeta, **keywordArguments).mean().item())

def analyzeSpectralConvergenceLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> Tensor:
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
	tensorLoss : Tensor
		Spectral convergence loss tensor.

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

@registrationAudioContest('SpectralConvergenceLoss mean')
def analyzeSpectralConvergenceLossMean(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor) -> float:
	"""Contest 'SpectralConvergenceLoss mean': mean of the spectral convergence loss tensor."""
	return float(analyzeSpectralConvergenceLoss(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta).mean().item())

def analyzeSTFTMagnitudeLoss(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor, **keywordArguments: Any) -> Tensor:
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
	tensorLoss : Tensor
		STFT magnitude loss tensor.

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
	return _analyzeLoss(freq.STFTMagnitudeLoss(**{'reduction': 'none', **keywordArguments}), tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta)

@registrationAudioContest('STFTMagnitudeLoss mean')
def analyzeSTFTMagnitudeLossMean(tensorSpectrogramMagnitudeAlfa: Tensor, tensorSpectrogramMagnitudeBeta: Tensor, **keywordArguments: Any) -> float:
	"""Contest 'STFTMagnitudeLoss mean': mean of the STFT magnitude loss tensor."""
	return float(analyzeSTFTMagnitudeLoss(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta, **keywordArguments).mean().item())

@registrationAudioContest('L1FrequencyLoss')
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
			λ ≜ a scaling factor

		ℒ = ∥|𝒮(ŵ)| − |𝒮(w)|∥₁
		y = 100 / (1 + (λ * ℒ))

	```
	"""
	λ = 10
	keywordArguments = dict(log=False, distance="L1", reduction="mean")
	L1: float = analyzeSTFTMagnitudeLossMean(tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta, **keywordArguments)
	return 100 / (1 + (λ * L1))
