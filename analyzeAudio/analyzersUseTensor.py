"""Analyzers that use the tensor to analyze audio data."""
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	import torch

def analyzeSRMR(tensorAudio: torch.Tensor, sampleRate: int, *, pytorchOnCPU: bool | None, **keywordArguments: Any) -> torch.Tensor:
	"""Compute speech-to-reverberation modulation energy ratio values from `tensorAudio`.

	(AI generated docstring)

	You can use this function to estimate speech-to-reverberation modulation energy
	ratio (SRMR) values from waveform data stored in `tensorAudio` [1]. The
	function analyzes `tensorAudio` at `sampleRate` and returns a `torch.Tensor`
	of SRMR values. Use `analyzeSRMRMean` [2] when you need one scalar summary
	instead of the full result.

	Mathematics
	-----------
	modulation spectral energy : equation
	```
		Let  eⱼ(m, n) ≜ temporal envelope of acoustic band j in frame m

		Eⱼ(m, f) = |ℱ(eⱼ(m, n))|²
	```
	SRMR ratio : equation
	```
		Let Ēₖ ≜ average modulation energy in modulation band k
			fₖ ≜ center frequency of modulation band k
			b₉₀ ≜ lowest acoustic band accounting for 90% of total modulation energy
			K* ≜ max{k : fₖ ≤ BW(b₉₀)}
			s ≜ returned `torch.Tensor`

		3 Hz ≲ f₁…f₄ ≲ 20 Hz
		20 Hz ≲ f₅…f₈ ≲ 160 Hz

		SRMR = (∑ₖ₌₁⁴ Ēₖ) / (∑ₖ₌₅ᴷ* Ēₖ)

		s contains the SRMR values computed from `tensorAudio`
	```

	Parameters
	----------
	tensorAudio : torch.Tensor
		Audio waveform data to analyze.
	sampleRate : int
		Sampling frequency of `tensorAudio` in hertz.
	pytorchOnCPU : bool | None
		Whether CPU execution should force a truthy `fast` setting when
		`keywordArguments` does not already provide one.

	Other Parameters
	----------------
	keywordArguments : Any
		Additional SRMR configuration values. A truthy `fast` value in
		`keywordArguments` takes precedence over `pytorchOnCPU`.

	Returns
	-------
	tensorSRMR : torch.Tensor
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
	[2] `analyzeSRMRMean`

	"""
	keywordArguments['fast'] = keywordArguments.get('fast') or pytorchOnCPU or None
	return speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, **keywordArguments)

aspectName = 'SRMR mean'
@registrationAudioAspect(aspectName)
def analyzeSRMRMean(tensorAudio: torch.Tensor, sampleRate: int, pytorchOnCPU: bool | None, **keywordArguments: Any) -> float:  # noqa: FBT001
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
