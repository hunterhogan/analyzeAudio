# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio.analyzersUseSpectrogram import analyzeChromagram
from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from torch import nn, tensor
from torch._tensor import Tensor
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import Any, cast, Protocol, TYPE_CHECKING
import auraloss
import numpy
import torch_l1_snr
import torch_log_wmse

if TYPE_CHECKING:
	from analyzeAudio._theTypes import SpectrogramPower
	from torch import device, Tensor

	# NOTE This is necessary because the original package thinks that type annotations and discipline are for pussies.
	class _AuralossChromaSTFTLoss(Protocol):
		fft_size: int
		window: Tensor
		device: device | None
		scale: str
		n_bins: int
		fb: Tensor

		def __call__(self, tensorInput: Tensor, tensorTarget: Tensor) -> Tensor: ...

dictionaryDefaultSumAndDifferenceSTFTLossKeywordArguments: dict[str, list[int]] = {
	'fft_sizes': [1024, 2048, 8192],
	'hop_sizes': [256, 512, 2048],
	'win_lengths': [1024, 2048, 8192],
}

def analyzeSRMR(tensorAudio: Tensor, sampleRate: int, *, pytorchOnCPU: bool | None, **keywordArguments: Any) -> Tensor:
	"""Compute speech-to-reverberation modulation energy ratio values from `tensorAudio`.

	(AI generated docstring)

	You can use this function to estimate speech-to-reverberation modulation energy
	ratio (SRMR) values from waveform data stored in `tensorAudio` [1]. The
	function analyzes `tensorAudio` at `sampleRate` and returns a `Tensor`
	of SRMR values. Use `analyzeSRMRMean` [2] when you need one scalar summary
	instead of the full result.

	Mathematics
	-----------
	framewise modulation energy : equation
	```
		Let eⱼ(m, n) ≜ temporal envelope of acoustic band j in frame m

		Eⱼ(m, f) = |ℱ(eⱼ(m, n))|²
	```
	SRMR ratio : equation
	```
		Let Ēₖ ≜ average modulation energy in modulation band k
			fₖ ≜ center frequency of modulation band k
			b₉₀ ≜ lowest acoustic band accounting for 90% of total modulation energy
			K* ≜ max{k : fₖ ≤ BW(b₉₀)}
			s ≜ returned `Tensor`

		3 Hz ≲ f₁…f₄ ≲ 20 Hz
		20 Hz ≲ f₅…f₈ ≲ 160 Hz

		SRMR = (∑ₖ₌₁⁴ Ēₖ) / (∑ₖ₌₅ᴷ* Ēₖ)

		s contains the SRMR values computed from `tensorAudio`
	```

	Parameters
	----------
	tensorAudio : Tensor
		Audio waveform data to analyze.
	sampleRate : int
		Sampling frequency of `tensorAudio` in hertz.
	pytorchOnCPU : bool | None
		Whether CPU execution should force a truthy `fast` setting when
		`keywordArguments` does not already provide one.

	Other Parameters
	----------------
	n_cochlear_filters : int = 23
		Number of gammatone acoustic filters used by the upstream SRMR
		implementation [2].
	low_freq : float = 125
		Lowest center frequency of the gammatone filterbank in hertz [2].
	min_cf : float = 4
		Center frequency of the first modulation filter in hertz [2].
	max_cf : float | None = None
		Center frequency of the last modulation filter in hertz. When `None`, the
		upstream implementation selects the value from `norm` [2].
	norm : bool = False
		Whether the upstream implementation should clamp modulation energy into a
		30 dB dynamic range [2].
	fast : bool = False
		Whether the upstream implementation should use the gammatonegram-based fast
		path [2]. A truthy `pytorchOnCPU` also enables this path.

	Returns
	-------
	tensorSRMR : Tensor
		SRMR values computed from `tensorAudio`.

	See Also
	--------
	`analyzeSRMRMean`
		Reduce the returned SRMR values to one registered scalar aspect.

	References
	----------
	[1] Falk, T. H., Zheng, C., & Chan, W.-Y. (2010). A non-intrusive
		quality and intelligibility measure of reverberant and dereverberated
		speech. IEEE Transactions on Audio, Speech, and Language Processing,
		18(7), 1766–1774.
		https://musaelab.ca/pdfs/J19.pdf
	[2] TorchMetrics documentation for
		`torchmetrics.functional.audio.srmr.speech_reverberation_modulation_energy_ratio`
		https://lightning.ai/docs/torchmetrics/stable/audio/speech_reverberation_modulation_energy_ratio.html

	"""
	keywordArguments['fast'] = keywordArguments.get('fast') or pytorchOnCPU or None
	return speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, **keywordArguments)

aspectName: str = 'SRMR mean'
@registrationAudioAspect(aspectName)
def analyzeSRMRMean(tensorAudio: Tensor, sampleRate: int, pytorchOnCPU: bool | None, **keywordArguments: Any) -> float:  # noqa: FBT001
	"""Compute the mean SRMR value for `tensorAudio`.

	(AI generated docstring)

	You can use this function when you need one scalar summary from
	`analyzeSRMR` [1]. The registered aspect name is `SRMR mean`.

	Returns
	-------
	srmrMean : float
		Mean of the values returned by `analyzeSRMR` [1].

	See Also
	--------
	`analyzeSRMR`
		Compute the full SRMR tensor before averaging.

	References
	----------
	[1] `analyzeSRMR`

	"""
	return float(analyzeSRMR(tensorAudio, sampleRate, pytorchOnCPU=pytorchOnCPU, **keywordArguments).mean().item())

#======== Contests ========================================

def _alignTensorAudioLengths(*tupleTensorAudio: Tensor) -> tuple[Tensor, ...]:
	"""I use this to trim multiple audio tensors to a shared sample length.

	Parameters
	----------
	tupleTensorAudio : Tensor
		Variadic collection of waveform tensors whose last axis stores samples.

	Returns
	-------
	tupleTensorAudioAligned : tuple[Tensor, ...]
		Tuple of tensors truncated to the minimum trailing sample length.

	Shape Transformation
	--------------------
	shared sample axis : transformation
	```
		Let N ≜ min(`tensorAudio`.shape[-1] for each `tensorAudio` in `tupleTensorAudio`)

		`tensorAudio` ↦ `tensorAudio`[..., :N]
	```
	"""
	intSharedLength = min(tensorAudio.shape[-1] for tensorAudio in tupleTensorAudio)
	return tuple(tensorAudio[..., :intSharedLength] for tensorAudio in tupleTensorAudio)

def _formatTensorAudioForBatchFirstLoss(tensorAudio: Tensor) -> Tensor:
	if tensorAudio.ndim < 4:
		return tensorAudio.unsqueeze(0)
	return tensorAudio

def _formatTensorAudio(tensorAudio: Tensor, *, processed: bool = False) -> Tensor:
	if not processed:
		if tensorAudio.ndim == 1:
			tensorFormatted: Tensor = tensorAudio.unsqueeze(0).unsqueeze(0)
		elif tensorAudio.ndim == 2:
			tensorFormatted = tensorAudio.unsqueeze(0)
		else:
			tensorFormatted = tensorAudio
	elif tensorAudio.ndim == 1:
		tensorFormatted = tensorAudio.unsqueeze(0).unsqueeze(0).unsqueeze(2)
	elif tensorAudio.ndim == 2:
		tensorFormatted = tensorAudio.unsqueeze(0).unsqueeze(2)
	elif tensorAudio.ndim == 3:
		tensorFormatted = tensorAudio.unsqueeze(0)
	else:
		tensorFormatted = tensorAudio
	return tensorFormatted

aspectName = 'LogWMSE'
@registrationAudioAspect(aspectName)
def analyzeLogWMSEMean(
	tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, tensorAudioCharlie: Tensor, sampleRate: int, **keywordArguments: Any
) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` and `tensorAudioCharlie` with logWMSE.

	(AI generated docstring)

	You can use this function to compute the registered `LogWMSE` audio
	aspect from a reference source `tensorAudioAlfa`, an estimated source
	`tensorAudioBeta`, and an unprocessed mixture `tensorAudioCharlie`. The function
	trims all three tensors to a shared sample length, reshapes the tensors for
	the upstream logWMSE implementation [1][2], and returns one scalar score.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated processed audio.
	tensorAudioCharlie : Tensor
		Unprocessed mixture or noisy input audio.
	sampleRate : int
		Sampling frequency of all three tensors in hertz.

	Returns
	-------
	logwmseMean : float
		Positive logWMSE score. Higher values indicate smaller weighted errors when
		`return_as_loss` keeps its wrapper default of `False` [2].

	Mathematics
	-----------
	frequency-weighted error score : equation
	```
		Let x ≜ `tensorAudioCharlie`,  y^ ≜ `tensorAudioBeta`,  y ≜ `tensorAudioAlfa`
			F(·) ≜ human-hearing weighting filter
			ρ(·) ≜ RMS
			α ≜ 1 / (ρ(F(x)) + ε_rms)
			τ ≜ imperceptible-error threshold

		d = F(α y^) - F(α y)
		dₙ = 0    if |dₙ| < τ
		WMSE = mean(d²)
		logWMSE = -4 log(WMSE + ε)
	```

	Shape Transformation
	--------------------
	shared trimming and layout : transformation
	```
		Let N ≜ min(
			`tensorAudioAlfa`.shape[-1],
			`tensorAudioBeta`.shape[-1],
			`tensorAudioCharlie`.shape[-1],
		)

		`tensorAudioCharlie`[..., :N]  ↦ [1, 1, sample], [1, channel, sample], or unchanged 3D+ shape
		`tensorAudioBeta`[..., :N]   ↦ [1, 1, 1, sample], [1, channel, 1, sample],
			[1, channel, stem, sample], or unchanged 4D+ shape
		`tensorAudioAlfa`[..., :N]   ↦ [1, 1, 1, sample], [1, channel, 1, sample],
			[1, channel, stem, sample], or unchanged 4D+ shape
	```

	Other Parameters
	----------------
	impulse_response : Tensor | None = None
		Optional finite impulse response filter for custom frequency weighting [2].
	impulse_response_sample_rate : int = 44100
		Sampling rate of `impulse_response` in hertz [2].
	return_as_loss : bool = False
		Whether the upstream implementation should return a negative loss instead of
		a positive loss. This wrapper sets `False` by default, but a caller-provided
		value overrides that default [2].
	bypass_filter : bool = False
		Whether the upstream implementation should skip frequency weighting [2].
	reduction : str = 'mean'
		Reduction mode forwarded to the upstream implementation [2].

	References
	----------
	[1] Jordal, I. `nomonosound/log-wmse-audio-quality`.
		https://github.com/nomonosound/log-wmse-audio-quality
	[2] Landschoot, C. `crlandsc/torch-log-wmse`.
		https://github.com/crlandsc/torch-log-wmse
	"""
	tensorAudioAlfa, tensorAudioBeta, tensorAudioCharlie = _alignTensorAudioLengths(
		tensorAudioAlfa, tensorAudioBeta, tensorAudioCharlie
	)
	dictionaryKeywordArguments: dict[str, Any] = {'return_as_loss': False, **keywordArguments}
	aspect = torch_log_wmse.LogWMSE(
		audio_length=tensorAudioCharlie.shape[-1] // sampleRate, sample_rate=sampleRate, **dictionaryKeywordArguments
	)
	return float(
		aspect(
			_formatTensorAudio(tensorAudioCharlie)
			, _formatTensorAudio(tensorAudioBeta, processed=True)
			, _formatTensorAudio(tensorAudioAlfa, processed=True)
		).item()
	)

name: str = 'L1SNR'
aspectName = name
@registrationAudioAspect(aspectName)
def analyzeL1SNRMean(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with L1SNR.

	(AI generated docstring)

	You can use this function to compute the registered `L1SNR` audio aspect
	from a reference tensor `tensorAudioAlfa` and an estimate tensor
	`tensorAudioBeta`. The function trims both tensors to a shared sample length,
	reshapes the tensors for the upstream `torch_l1_snr.L1SNRLoss`
	implementation [1][2][3], negates the upstream loss value, and returns a
	higher-is-better score.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to be scored against `tensorAudioAlfa`.

	Returns
	-------
	l1snrMean : float
		Positive score equal to the negative of the upstream L1SNR loss.

	Mathematics
	-----------
	batchwise L1SNR score : equation
	```
		Let y^ᵦ ≜ batch item b from `tensorAudioBeta`
			yᵦ ≜ batch item b from `tensorAudioAlfa`
			eᵦ ≜ mean(|vec(y^ᵦ - yᵦ)|)
			rᵦ ≜ mean(|vec(yᵦ)|)
			B ≜ batch size after local reshaping

		D₁,ᵦ = 10 log₁₀((eᵦ + ε) / (rᵦ + ε))
		l1snrMean = -(1 / B) ∑_(b = 1)^B D₁,ᵦ
	```

	Shape Transformation
	--------------------
	shared trimming and batch axis : transformation
	```
		Let N ≜ min(`tensorAudioAlfa`.shape[-1], `tensorAudioBeta`.shape[-1])

		`tensorAudioAlfa` ↦ `tensorAudioAlfa`[..., :N]
		`tensorAudioBeta` ↦ `tensorAudioBeta`[..., :N]
		If ndim < 4, prepend one singleton batch axis before the upstream loss call.
	```

	Other Parameters
	----------------
	weight : float = 1.0
		Overall multiplier used by the upstream loss object [3].
	eps : float = 1e-3
		Stability term used in the logarithmic ratio [1][3].
	l1_weight : float = 0.0
		When non-zero, the upstream package blends the pure L1SNR objective with an
		auto-scaled L1 term [3].

	See Also
	--------
	`analyzeL1SNRDBMean`
		Add adaptive level-matching regularization to the time-domain score.
	`analyzeSTFTL1SNRDBMean`
		Apply a related objective in the spectrogram domain.

	References
	----------
	[1] Watcharasupat, K. N., & Lerch, A. (2024). A Stem-Agnostic Single-Decoder
		System for Music Source Separation Beyond Four Stems.
		https://arxiv.org/html/2406.18747v2
	[2] Watcharasupat, K. N., Wu, C.-W., Ding, Y., Orife, I., Hipple, A. J.,
		Williams, P. A., Kramer, S., Lerch, A., & Wolcott, W. (2023).
		A Generalized Bandsplit Neural Network for Cinematic Audio Source Separation.
		https://arxiv.org/abs/2309.02539
	[3] Landschoot, C. `crlandsc/torch-l1-snr`.
		https://github.com/crlandsc/torch-l1-snr
	"""
	tensorAudioAlfa, tensorAudioBeta = _alignTensorAudioLengths(tensorAudioAlfa, tensorAudioBeta)
	aspect = torch_l1_snr.L1SNRLoss(name, **keywordArguments)
	return -float(
		aspect(_formatTensorAudioForBatchFirstLoss(tensorAudioBeta), _formatTensorAudioForBatchFirstLoss(tensorAudioAlfa)).item()
	)

name = 'L1SNRDB'
aspectName = name
@registrationAudioAspect(aspectName)
def analyzeL1SNRDBMean(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with regularized L1SNR.

	(AI generated docstring)

	You can use this function to compute the registered `L1SNRDB` audio
	aspect from a reference tensor `tensorAudioAlfa` and an estimate tensor
	`tensorAudioBeta`. The function trims both tensors to a shared sample length,
	reshapes the tensors for the upstream `torch_l1_snr.L1SNRDBLoss`
	implementation [1][2][3], negates the upstream loss value, and returns a
	higher-is-better score.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to be scored against `tensorAudioAlfa`.

	Returns
	-------
	l1snrdbMean : float
		Positive score equal to the negative of the upstream regularized L1SNR loss.

	Mathematics
	-----------
	default regularized objective : equation
	```
		Let y^ᵦ ≜ batch item b from `tensorAudioBeta`
			yᵦ ≜ batch item b from `tensorAudioAlfa`
			eᵦ ≜ mean(|vec(y^ᵦ - yᵦ)|)
			rᵦ ≜ mean(|vec(yᵦ)|)
			L^ᵦ ≜ dBRMS(y^ᵦ)
			Lᵦ ≜ dBRMS(yᵦ)
			Rᵦ ≜ |L^ᵦ - Lᵦ|
			ηᵦ ≜ 𝕀⟦Lᵦ > max(L^ᵦ, L_min)⟧

		D₁,ᵦ = 10 log₁₀((eᵦ + ε_snr) / (rᵦ + ε_snr))
		λᵦ = λ₀ + ηᵦ Δλ clamp_[0,1](Rᵦ / (Lᵦ - L_min))
		Jᵦ = D₁,ᵦ + sg[λᵦ] Rᵦ
		l1snrdbMean = -(1 / B) ∑_(b = 1)^B Jᵦ
	```

	Shape Transformation
	--------------------
	shared trimming and batch axis : transformation
	```
		Let N ≜ min(`tensorAudioAlfa`.shape[-1], `tensorAudioBeta`.shape[-1])

		`tensorAudioAlfa` ↦ `tensorAudioAlfa`[..., :N]
		`tensorAudioBeta` ↦ `tensorAudioBeta`[..., :N]
		If ndim < 4, prepend one singleton batch axis before the upstream loss call.
	```

	Other Parameters
	----------------
	weight : float = 1.0
		Overall multiplier used by the upstream loss object [3].
	lambda0 : float = 0.1
		Minimum adaptive weight for the level-matching term [1][3].
	delta_lambda : float = 0.9
		Additional adaptive weight range for the level-matching term [1][3].
	l1snr_eps : float = 1e-3
		Stability term used in the L1SNR ratio [1][3].
	dbrms_eps : float = 1e-8
		Stability term used in `dBRMS` calculations [1][3].
	lmin : float = -60.0
		Minimum dB level used in the adaptive weighting rule [1][3].
	use_regularization : bool = True
		Whether the upstream implementation should include the level-matching term [3].
	l1_weight : float = 0.0
		When non-zero, the upstream package blends the regularized objective with a
		separately scaled L1 term [3].

	See Also
	--------
	`analyzeL1SNRMean`
		Compute the corresponding unregularized time-domain score.
	`analyzeMultiL1SNRDBMean`
		Combine the time-domain and STFT-domain regularized objectives.

	References
	----------
	[1] Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of these
		Things Around It: Music Source Separation via Hyperellipsoidal Queries.
		https://arxiv.org/html/2501.16171v1
	[2] Watcharasupat, K. N., & Lerch, A. (2024). A Stem-Agnostic Single-Decoder
		System for Music Source Separation Beyond Four Stems.
		https://arxiv.org/html/2406.18747v2
	[3] Landschoot, C. `crlandsc/torch-l1-snr`.
		https://github.com/crlandsc/torch-l1-snr
	"""
	tensorAudioAlfa, tensorAudioBeta = _alignTensorAudioLengths(tensorAudioAlfa, tensorAudioBeta)
	aspect = torch_l1_snr.L1SNRDBLoss(name, **keywordArguments)
	return -float(
		aspect(_formatTensorAudioForBatchFirstLoss(tensorAudioBeta), _formatTensorAudioForBatchFirstLoss(tensorAudioAlfa)).item()
	)

name = 'MultiL1SNRDB'
aspectName = name
@registrationAudioAspect(aspectName)
def analyzeMultiL1SNRDBMean(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with combined time and STFT L1SNRDB.

	(AI generated docstring)

	You can use this function to compute the registered `MultiL1SNRDB` audio
	aspect from a reference tensor `tensorAudioAlfa` and an estimate tensor
	`tensorAudioBeta`. The function trims both tensors to a shared sample length,
	reshapes the tensors for the upstream `torch_l1_snr.MultiL1SNRDBLoss`
	implementation [1][2][3], negates the upstream loss value, and returns a
	higher-is-better score.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to be scored against `tensorAudioAlfa`.

	Returns
	-------
	multiL1snrdb : float
		Positive score equal to the negative of the upstream combined loss.

	Mathematics
	-----------
	combined multi-domain objective : equation
	```
		Let J_time ≜ time-domain L1SNRDB objective
			J_spec ≜ spectrogram-domain L1SNRDB objective
			ω ≜ `spec_weight`

		J_multi = (1 - ω) J_time + ω J_spec
		multiL1snrdb = -J_multi
	```

	Shape Transformation
	--------------------
	shared trimming and batch axis : transformation
	```
		Let N ≜ min(`tensorAudioAlfa`.shape[-1], `tensorAudioBeta`.shape[-1])

		`tensorAudioAlfa` ↦ `tensorAudioAlfa`[..., :N]
		`tensorAudioBeta` ↦ `tensorAudioBeta`[..., :N]
		If ndim < 4, prepend one singleton batch axis before the upstream loss call.
	```

	Other Parameters
	----------------
	weight : float = 1.0
		Overall multiplier for the combined loss [3].
	spec_weight : float = 0.5
		Relative weight of the spectrogram-domain objective within the combined loss [3].
	l1_weight : float = 0.0
		Amount of L1 blending used inside both component losses [3].
	use_time_regularization : bool = True
		Whether the time-domain component should include level matching [1][3].
	use_spec_regularization : bool = False
		Whether the spectrogram-domain component should include level matching [1][3].
	lambda0 : float = 0.1
		Minimum adaptive weight shared by both component losses [1][3].
	delta_lambda : float = 0.9
		Additional adaptive weight range shared by both component losses [1][3].
	n_ffts : list[int] = [512, 1024, 2048]
		FFT sizes used by the spectrogram-domain component [3].
	hop_lengths : list[int] = [128, 256, 512]
		Hop lengths used by the spectrogram-domain component [3].
	win_lengths : list[int] = [512, 1024, 2048]
		Window lengths used by the spectrogram-domain component [3].
	window_fn : str = 'hann'
		Window function name used by the spectrogram-domain component [3].
	min_audio_length : int = 512
		Minimum sample length required by the STFT component before fallback [3].
	time_loss_params : dict | None = None
		Optional override dictionary for the time-domain sub-loss [3].
	spec_loss_params : dict | None = None
		Optional override dictionary for the spectrogram-domain sub-loss [3].
	mps_cpu_fallback : bool = True
		Whether the upstream implementation should route STFT work through CPU on
		MPS devices to avoid incorrect gradients [3].

	See Also
	--------
	`analyzeL1SNRDBMean`
		Use only the time-domain regularized objective.
	`analyzeSTFTL1SNRDBMean`
		Use only the spectrogram-domain regularized objective.

	References
	----------
	[1] Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of these
		Things Around It: Music Source Separation via Hyperellipsoidal Queries.
		https://arxiv.org/html/2501.16171v1
	[2] Watcharasupat, K. N., & Lerch, A. (2024). A Stem-Agnostic Single-Decoder
		System for Music Source Separation Beyond Four Stems.
		https://arxiv.org/html/2406.18747v2
	[3] Landschoot, C. `crlandsc/torch-l1-snr`.
		https://github.com/crlandsc/torch-l1-snr
	"""
	tensorAudioAlfa, tensorAudioBeta = _alignTensorAudioLengths(tensorAudioAlfa, tensorAudioBeta)
	aspect = torch_l1_snr.MultiL1SNRDBLoss(name, **keywordArguments)
	return -float(
		aspect(_formatTensorAudioForBatchFirstLoss(tensorAudioBeta), _formatTensorAudioForBatchFirstLoss(tensorAudioAlfa)).item()
	)

name = 'STFTL1SNRDB'
aspectName = name
@registrationAudioAspect(aspectName)
def analyzeSTFTL1SNRDBMean(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with spectrogram-domain L1SNRDB.

	(AI generated docstring)

	You can use this function to compute the registered `STFTL1SNRDB`
	audio aspect from a reference tensor `tensorAudioAlfa` and an estimate tensor
	`tensorAudioBeta`. The function trims both tensors to a shared sample length,
	reshapes the tensors for the upstream `torch_l1_snr.STFTL1SNRDBLoss`
	implementation [1][2][3], negates the upstream loss value, and returns a
	higher-is-better score.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to be scored against `tensorAudioAlfa`.

	Returns
	-------
	stftL1snrdb : float
		Positive score equal to the negative of the upstream spectrogram-domain loss.

	Mathematics
	-----------
	default spectrogram-domain objective : equation
	```
		Let S^ᵦ ≜ STFT(y^ᵦ),  Sᵦ ≜ STFT(yᵦ)
			e_Re,ᵦ ≜ mean(|vec(Re S^ᵦ - Re Sᵦ)|)
			r_Re,ᵦ ≜ mean(|vec(Re Sᵦ)|)
			e_Im,ᵦ ≜ mean(|vec(Im S^ᵦ - Im Sᵦ)|)
			r_Im,ᵦ ≜ mean(|vec(Im Sᵦ)|)

		D_Re,ᵦ = 10 log₁₀((e_Re,ᵦ + ε) / (r_Re,ᵦ + ε))
		D_Im,ᵦ = 10 log₁₀((e_Im,ᵦ + ε) / (r_Im,ᵦ + ε))
		J_spec,ᵦ = D_Re,ᵦ + D_Im,ᵦ
		stftL1snrdb = -(1 / B) ∑_(b = 1)^B J_spec,ᵦ
	```

	Shape Transformation
	--------------------
	shared trimming and batch axis : transformation
	```
		Let N ≜ min(`tensorAudioAlfa`.shape[-1], `tensorAudioBeta`.shape[-1])

		`tensorAudioAlfa` ↦ `tensorAudioAlfa`[..., :N]
		`tensorAudioBeta` ↦ `tensorAudioBeta`[..., :N]
		If ndim < 4, prepend one singleton batch axis before the upstream loss call.
	```

	Other Parameters
	----------------
	weight : float = 1.0
		Overall multiplier used by the upstream loss object [3].
	lambda0 : float = 0.1
		Minimum adaptive weight for the optional spectrogram level-matching term [1][3].
	delta_lambda : float = 0.9
		Additional adaptive weight range for the optional spectrogram level-matching term [1][3].
	l1snr_eps : float = 1e-3
		Stability term used in the complex L1SNR ratios [2][3].
	dbrms_eps : float = 1e-8
		Stability term used in the optional magnitude `dBRMS` computation [1][3].
	lmin : float = -60.0
		Minimum dB level used in the adaptive weighting rule [1][3].
	n_ffts : list[int] = [512, 1024, 2048]
		FFT sizes used for multi-resolution STFT analysis [3].
	hop_lengths : list[int] = [128, 256, 512]
		Hop lengths paired with `n_ffts` [3].
	win_lengths : list[int] = [512, 1024, 2048]
		Window lengths paired with `n_ffts` [3].
	window_fn : str = 'hann'
		Window function name used by the upstream spectrogram transforms [3].
	min_audio_length : int = 512
		Minimum sample length required before the upstream implementation falls back
		to a time-domain objective [3].
	use_regularization : bool = False
		Whether the upstream implementation should add the spectrogram-magnitude
		level-matching term [1][3].
	spec_reg_coef : float = 0.1
		Scale factor for the optional spectrogram regularization term [3].
	l1_weight : float = 0.0
		When non-zero, the upstream package blends the spectrogram objective with an
		auto-scaled L1 term [3].
	mps_cpu_fallback : bool = True
		Whether the upstream implementation should route STFT work through CPU on
		MPS devices to avoid incorrect gradients [3].

	See Also
	--------
	`analyzeL1SNRDBMean`
		Compute the corresponding time-domain regularized score.
	`analyzeMultiL1SNRDBMean`
		Blend the time-domain and spectrogram-domain regularized objectives.

	References
	----------
	[1] Watcharasupat, K. N., & Lerch, A. (2025). Separate This, and All of these
		Things Around It: Music Source Separation via Hyperellipsoidal Queries.
		https://arxiv.org/html/2501.16171v1
	[2] Watcharasupat, K. N., & Lerch, A. (2024). A Stem-Agnostic Single-Decoder
		System for Music Source Separation Beyond Four Stems.
		https://arxiv.org/html/2406.18747v2
	[3] Landschoot, C. `crlandsc/torch-l1-snr`.
		https://github.com/crlandsc/torch-l1-snr
	"""
	tensorAudioAlfa, tensorAudioBeta = _alignTensorAudioLengths(tensorAudioAlfa, tensorAudioBeta)
	aspect = torch_l1_snr.STFTL1SNRDBLoss(name, **keywordArguments)
	return -float(
		aspect(_formatTensorAudioForBatchFirstLoss(tensorAudioBeta), _formatTensorAudioForBatchFirstLoss(tensorAudioAlfa)).item()
	)

def _analyzeAuralossWaveformLoss(aspect: nn.Module, tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor) -> float:
	tensorAudioAlfa, tensorAudioBeta = _alignTensorAudioLengths(tensorAudioAlfa, tensorAudioBeta)
	return float(aspect(_formatTensorAudio(tensorAudioBeta), _formatTensorAudio(tensorAudioAlfa)).item())

aspectName = 'DCLoss'
@registrationAudioAspect(aspectName)
def analyzeDCLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with direct-current offset loss.

	(AI generated docstring)

	You can use this function to compute the registered `DCLoss` audio aspect for a
	reference waveform `tensorAudioAlfa` and an estimated waveform `tensorAudioBeta`
	using the upstream `auraloss.time.DCLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	dcLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.DCLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.DCLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'ESRLoss'
@registrationAudioAspect(aspectName)
def analyzeESRLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with error-to-signal ratio loss.

	(AI generated docstring)

	You can use this function to compute the registered `ESRLoss` audio aspect for
	`tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.time.ESRLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	esrLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.ESRLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.ESRLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'LogCoshLoss'
@registrationAudioAspect(aspectName)
def analyzeLogCoshLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with log-cosh waveform loss.

	(AI generated docstring)

	You can use this function to compute the registered `LogCoshLoss` audio aspect
	for `tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.time.LogCoshLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	logCoshLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.LogCoshLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.LogCoshLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'SNRLoss'
@registrationAudioAspect(aspectName)
def analyzeSNRLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with signal-to-noise ratio loss.

	(AI generated docstring)

	You can use this function to compute the registered `SNRLoss` audio aspect for
	`tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.time.SNRLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	snrLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.SNRLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.SNRLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'SISDRLoss'
@registrationAudioAspect(aspectName)
def analyzeSISDRLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with scale-invariant SDR loss.

	(AI generated docstring)

	You can use this function to compute the registered `SISDRLoss` audio aspect
	for `tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.time.SISDRLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	siSdrLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.SISDRLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.SISDRLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'SDSDRLoss'
@registrationAudioAspect(aspectName)
def analyzeSDSDRLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with scale-dependent SDR loss.

	(AI generated docstring)

	You can use this function to compute the registered `SDSDRLoss` audio aspect
	for `tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.time.SDSDRLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	sdSdrLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.time.SDSDRLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.time.SDSDRLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'STFTLoss'
@registrationAudioAspect(aspectName)
def analyzeSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with STFT-domain loss.

	(AI generated docstring)

	You can use this function to compute the registered `STFTLoss` audio aspect for
	`tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.freq.STFTLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	stftLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.freq.STFTLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.freq.STFTLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'MelSTFTLoss'
@registrationAudioAspect(aspectName)
def analyzeMelSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, sampleRate: int, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with mel-scaled STFT loss.

	(AI generated docstring)

	You can use this function to compute the registered `MelSTFTLoss` audio aspect
	for `tensorAudioAlfa` and `tensorAudioBeta` at `sampleRate` with the upstream
	`auraloss.freq.MelSTFTLoss` implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.
	sampleRate : int
		Sampling frequency used to configure the mel filterbank in hertz.

	Returns
	-------
	melStftLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.freq.MelSTFTLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	dictionaryKeywordArguments: dict[str, Any] = {'sample_rate': sampleRate, **keywordArguments}
	return _analyzeAuralossWaveformLoss(auraloss.freq.MelSTFTLoss(**dictionaryKeywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'ChromaSTFTLoss'
@registrationAudioAspect(aspectName)
def analyzeChromaSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, sampleRate: int, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with chroma-weighted STFT loss.

	(AI generated docstring)

	You can use this function to compute the registered `ChromaSTFTLoss` audio
	aspect for `tensorAudioAlfa` and `tensorAudioBeta` at `sampleRate`. This
	function configures `auraloss.freq.STFTLoss` to `scale='chroma'` and replaces
	the frequency basis with a chromagram transform from `analyzeChromagram` [1, 2].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.
	sampleRate : int
		Sampling frequency used to compute the chroma basis in hertz.

	Returns
	-------
	chromaStftLoss : float
		Loss value produced by the configured chroma-domain objective.

	Other Parameters
	----------------
	n_chroma : int = 12
		Number of chroma bins used to build the chroma basis.
	n_bins : int = 12
		Alias of `n_chroma`. This function prioritizes `n_chroma` when both values are present.
	keywordArguments : Any
		Additional keyword argument mapping forwarded to `auraloss.freq.STFTLoss` [1],
		except `scale`, which this function overrides to `chroma`.

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	[2] `analyzeChromagram`
	"""
	dictionaryKeywordArguments: dict[str, Any] = {'sample_rate': sampleRate, **keywordArguments}
	integerChromaBins: int = int(dictionaryKeywordArguments.pop('n_chroma', dictionaryKeywordArguments.pop('n_bins', 12)))
	dictionaryKeywordArguments.pop('scale', None)
	aspect = cast('_AuralossChromaSTFTLoss', auraloss.freq.STFTLoss(**dictionaryKeywordArguments))
	aspect.scale = 'chroma'
	aspect.n_bins = integerChromaBins
	integerFrequencyBins: int = (aspect.fft_size // 2) + 1
	arrayIdentityFrequencyPower: numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float32]] = numpy.identity(
		integerFrequencyBins
		, dtype=numpy.float32
	)
	aspect.fb = tensor(
		analyzeChromagram(
			cast('SpectrogramPower', arrayIdentityFrequencyPower)
			, sampleRate
			, n_fft=aspect.fft_size
			, n_chroma=integerChromaBins
			, norm=None
		)
		, dtype=aspect.window.dtype
	).unsqueeze(0)
	if aspect.device is not None:
		aspect.fb = aspect.fb.to(aspect.device)
	return _analyzeAuralossWaveformLoss(cast('nn.Module', aspect), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'MultiResolutionSTFTLoss'
@registrationAudioAspect(aspectName)
def analyzeMultiResolutionSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with multi-resolution STFT loss.

	(AI generated docstring)

	You can use this function to compute the registered
	`MultiResolutionSTFTLoss` audio aspect for `tensorAudioAlfa` and
	`tensorAudioBeta` using the upstream `auraloss.freq.MultiResolutionSTFTLoss`
	implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	multiResolutionStftLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.freq.MultiResolutionSTFTLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.freq.MultiResolutionSTFTLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'RandomResolutionSTFTLoss'
@registrationAudioAspect(aspectName)
def analyzeRandomResolutionSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with random-resolution STFT loss.

	(AI generated docstring)

	You can use this function to compute the registered
	`RandomResolutionSTFTLoss` audio aspect for `tensorAudioAlfa` and
	`tensorAudioBeta` using the upstream `auraloss.freq.RandomResolutionSTFTLoss`
	implementation [1].

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	randomResolutionStftLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	keywordArguments : Any
		Keyword argument mapping forwarded to `auraloss.freq.RandomResolutionSTFTLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	return _analyzeAuralossWaveformLoss(auraloss.freq.RandomResolutionSTFTLoss(**keywordArguments), tensorAudioAlfa, tensorAudioBeta)

aspectName = 'SumAndDifferenceSTFTLoss'
@registrationAudioAspect(aspectName)
def analyzeSumAndDifferenceSTFTLoss(tensorAudioAlfa: Tensor, tensorAudioBeta: Tensor, **keywordArguments: Any) -> float:
	"""Score `tensorAudioBeta` against `tensorAudioAlfa` with sum-and-difference STFT loss.

	(AI generated docstring)

	You can use this function to compute the registered `SumAndDifferenceSTFTLoss`
	audio aspect for `tensorAudioAlfa` and `tensorAudioBeta` using the upstream
	`auraloss.freq.SumAndDifferenceSTFTLoss` implementation [1]. This function
	starts from `dictionaryDefaultSumAndDifferenceSTFTLossKeywordArguments` and
	then applies caller-provided overrides.

	Parameters
	----------
	tensorAudioAlfa : Tensor
		Reference target audio.
	tensorAudioBeta : Tensor
		Estimated audio to score against `tensorAudioAlfa`.

	Returns
	-------
	sumAndDifferenceStftLoss : float
		Loss value produced by the upstream `auraloss` implementation [1].

	Other Parameters
	----------------
	fft_sizes : list[int] = [1024, 2048, 8192]
		FFT size list used by the upstream loss object when not overridden.
	hop_sizes : list[int] = [256, 512, 2048]
		Hop size list used by the upstream loss object when not overridden.
	win_lengths : list[int] = [1024, 2048, 8192]
		Window length list used by the upstream loss object when not overridden.
	keywordArguments : Any
		Additional keyword argument mapping forwarded to
		`auraloss.freq.SumAndDifferenceSTFTLoss` [1].

	References
	----------
	[1] Steinmetz, C. J., Reiss, J. D., & Bryan, N. J. `csteinmetz1/auraloss`.
		https://github.com/csteinmetz1/auraloss
	"""
	dictionaryKeywordArguments: dict[str, Any] = {**dictionaryDefaultSumAndDifferenceSTFTLossKeywordArguments, **keywordArguments}
	return _analyzeAuralossWaveformLoss(auraloss.freq.SumAndDifferenceSTFTLoss(**dictionaryKeywordArguments), tensorAudioAlfa, tensorAudioBeta)
