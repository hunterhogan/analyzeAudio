# pyright: reportArgumentType=false
# ty:ignore[invalid-argument-type]
# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio.registry import registrationAudioAspect
from torch import tensor
from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import TYPE_CHECKING
import sys

if TYPE_CHECKING:
	from torch import Tensor
	from typing import Any

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
	keywordArguments['fast'] = keywordArguments.get('fast', pytorchOnCPU) or False
	return speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, **keywordArguments)

@registrationAudioAspect('SRMR mean')
def analyzeSRMRMean(tensorAudio: Tensor, sampleRate: int, pytorchOnCPU: bool | None, **keywordArguments: Any) -> float:  # noqa: FBT001
	"""Aspect 'SRMR mean': mean speech-to-reverberation modulation energy ratio.

	Returns
	-------
	srmrMean : float
		Mean SRMR value for `tensorAudio`.

	"""
	return float(analyzeSRMR(tensorAudio, sampleRate, pytorchOnCPU=pytorchOnCPU, **keywordArguments).mean().item())

# TODO Requires a lot of memory, and concurrency is causing crashes.
def analyzeDNSMOS(tensorAudio: Tensor, sampleRate: int, **keywordArguments: Any) -> Tensor:
	defaults: dict[str, bool] = {'personalized': False}
	return deep_noise_suppression_mean_opinion_score(tensorAudio, sampleRate, {**defaults, **keywordArguments})

# @registrationAudioAspect('DNSMOS mean')
def analyzeDNSMOSMean(tensorAudio: Tensor, sampleRate: int) -> float:
	"""Aspect 'DNSMOS mean': mean Deep Noise Suppression MOS score.

	Returns
	-------
	dnsmosMean : float
		Mean DNSMOS component score.

	"""
	return float(analyzeDNSMOS(tensorAudio, sampleRate).mean().item())

def analyzeNISQA(tensorAudio: Tensor, sampleRate: int) -> Tensor:
	try:
		return non_intrusive_speech_quality_assessment(tensorAudio, sampleRate)
	except RuntimeError as ERRORmessage:
		message: str = f'I could not compute `analyzeNISQA({tensorAudio.shape = }, {sampleRate = })` because "{ERRORmessage}".'
		sys.stderr.write(message + '\n')
		return tensor([])

@registrationAudioAspect('NISQA mean')
def analyzeNISQAMean(tensorAudio: Tensor, sampleRate: int) -> float | None:
	"""Aspect 'NISQA mean': mean non-intrusive speech quality score.

	Returns
	-------
	nisqaMean : float | None
		Mean NISQA component score.

	"""
	tensorAspect: Tensor = analyzeNISQA(tensorAudio, sampleRate)
	if len(tensorAspect) == 0:
		return None
	return float(tensorAspect.mean().item())
