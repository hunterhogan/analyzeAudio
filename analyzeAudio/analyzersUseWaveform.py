"""Analyzers that use the waveform of audio data."""
from __future__ import annotations

from analyzeAudio.registry import registrationAudioAspect
from typing import TYPE_CHECKING
import librosa
import numpy

if TYPE_CHECKING:
	from analyzeAudio import ArrayAspect, ArrayAspectWaveformFramewise, Audio
	from numpy import dtype, floating, ndarray
	from typing import Any

# TODO `librosa.zero_crossings`.

def analyzeRMSWaveform(waveform: Audio, **keywordArguments: Any) -> ArrayAspectWaveformFramewise:
	"""Compute framewise root-mean-square amplitude.

	(AI generated docstring)

	You can use this function to compute per-frame root-mean-square (RMS) amplitude from `waveform`.
	This is a thin wrapper around `librosa.feature.rms` that returns an ArrayAspect of RMS amplitudes
	for each analysis frame.

	Parameters
	----------
	waveform : Audio
		Waveform whose framewise RMS amplitude is measured.
	keywordArguments : Any
		Additional keyword arguments passed to `librosa.feature.rms`.

	Returns
	-------
	rootMeanSquare : ArrayAspectWaveformFramewise
		Framewise root-mean-square amplitude (linear units). Values are amplitudes, not decibels.

	Mathematics
	-----------
	root-mean-square amplitude : equation
	```
		Let xₜ[k] ≜ k-th waveform sample in frame t
			N ≜ number of samples in frame t

		RMS(t) = √((1 / N) ∑_(k = 0)^(N - 1) xₜ[k]²)
	```

	References
	----------
	[1] Constantinescu, C., & Brad, R. (2023). An overview on sound features
		in time and frequency domain. International Journal of Advanced Statistics and IT&C for
		Economics and Life Sciences, 13(1), 51–56.
		https://reference-global.com/download/article/10.2478/ijasitels-2023-0006.pdf
	[2] Panagiotakis, C., & Tziritas, G. (2005). A speech/music discriminator
		based on RMS and zero-crossings. IEEE Transactions on Multimedia, 7(1), 155–166.
		https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf
	[3] `librosa.feature.rms`
		https://librosa.org/doc/latest/generated/librosa.feature.rms.html
	"""
	return librosa.feature.rms(y=waveform, **keywordArguments)

@registrationAudioAspect('RMS Waveform mean')
def analyzeRMSWaveformMean(waveform: Audio, **keywordArguments: Any) -> float:
	"""Aspect 'RMS Waveform mean': mean framewise RMS level in decibels.

	Returns
	-------
	rmsMean : float
		Arithmetic mean of the framewise RMS amplitude.

	"""
	return float(analyzeRMSWaveform(waveform, **keywordArguments).mean().item())

def analyzeRMSWaveform_dB(waveform: Audio, **keywordArguments: Any) -> ArrayAspectWaveformFramewise:
	"""Compute framewise RMS level in decibels.

	(AI generated docstring)

	You can use this function to convert framewise RMS amplitudes into decibel units using a 20 ·
	log10 mapping. The function obtains linear RMS values from `analyzeRMSWaveform` and applies a
	logarithmic mapping. Callers should handle zero values if they wish to replace −inf or placeholder
	values produced by the logarithm.

	Parameters
	----------
	waveform : Audio
		Waveform whose framewise RMS amplitude is measured.
	keywordArguments : Any
		Additional arguments forwarded to `analyzeRMSWaveform`.

	Returns
	-------
	rms_dB : ArrayAspectWaveformFramewise
		Framewise RMS level in decibels (dB).

	See Also
	--------
	`analyzeRMSWaveform`
		Compute framewise RMS amplitude.

	References
	----------
	[1] `librosa.feature.rms`
		https://librosa.org/doc/latest/generated/librosa.feature.rms.html
	[2] `numpy.log10`
		https://numpy.org/doc/stable/reference/generated/numpy.log10.html
	"""
	arrayRMS: ArrayAspectWaveformFramewise = analyzeRMSWaveform(waveform, **keywordArguments)
	return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0), out=None)

@registrationAudioAspect('RMS Waveform dB mean')
def analyzeRMSWaveform_dBMean(waveform: Audio, **keywordArguments: Any) -> float:
	"""Aspect 'RMS Waveform dB mean': mean framewise RMS level in decibels.

	Returns
	-------
	rootMeanSquare_dBMean : float
		Mean value of the time-varying RMS level in decibels.

	"""
	return float(analyzeRMSWaveform_dB(waveform, **keywordArguments).mean().item())

def analyzeTempogram(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> ArrayAspect:
	"""Compute a local autocorrelation tempogram from the waveform.

	(AI generated docstring)

	You can use this function to measure how strongly different pulse periods are present over time in
	`waveform`. The function returns a time-varying tempo representation whose values reflect local
	periodicity on a beats-per-minute axis [1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose local periodic structure is analyzed.
	sampleRate : int
		Sampling rate of `waveform` in hertz.
	keywordArguments : Any
		Additional tempogram-analysis settings forwarded to `librosa.feature.tempogram`.

	Returns
	-------
	tempogram : ArrayAspect
		Time-varying autocorrelation tempogram of `waveform`.

	Mathematics
	-----------
	local autocorrelation : equation
	```
		Let Δ(n) ≜ onset-strength curve derived from `waveform`
			W ≜ centered analysis window with support [−N : N]
			A(t, ℓ) ≜ local autocorrelation at frame t and lag ℓ

		A(t, ℓ) = (∑_n Δ(n) Δ(n + ℓ) W(n − t)) / (2N + 1 − ℓ)
	```

	tempo-domain mapping : equation
	```
		Let r ≜ frame step in seconds
			τ ≜ tempo in beats per minute
			Tᴬ(t, τ) ≜ autocorrelation tempogram

		τ = 60 / (r ℓ)
		Tᴬ(t, τ) = A(t, ℓ)
	```

	References
	----------
	[1] Grosche, P., Müller, M., & Kurth, F. (2010). Cyclic tempogram — a
		mid-level tempo representation for music signals. Proceedings of the IEEE International
		Conference on Acoustics, Speech and Signal Processing, 5522–5525.
		https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf
	[2] Müller, M., Ellis, D. P. W., Klapuri, A., & Richard, G. (2011). Signal
		processing for music analysis. IEEE Journal of Selected Topics in Signal Processing, 5(6),
		1088–1110. https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf
	[3] `librosa.feature.tempogram`
		https://librosa.org/doc/latest/generated/librosa.feature.tempogram.html
	"""
	return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)

@registrationAudioAspect('Tempogram mean')
def analyzeTempogramMean(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> float:
	"""Aspect 'Tempogram mean': mean of the framewise tempogram.

	Returns
	-------
	tempogramMean : float
		Mean value of the time-varying tempogram.

	"""
	return float(analyzeTempogram(waveform, sampleRate, **keywordArguments).mean().item())

def analyzeTempo(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> ndarray[tuple[int], dtype[floating[Any]]]:
	"""Estimate tempo in beats per minute from waveform periodicity.

	(AI generated docstring)

	You can use this function to estimate the dominant tempo implied by `waveform`. The function
	interprets the strongest periodicities of a tempogram as tempo candidates in beats per minute, and
	`keywordArguments` can request either a single estimate or a time-varying tempo trajectory [1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose predominant tempo is estimated.
	sampleRate : int
		Sampling rate of `waveform` in hertz.
	keywordArguments : Any
		Additional tempo-estimation settings forwarded to `librosa.feature.tempo`.

	Returns
	-------
	tempo : ndarray[tuple[int], dtype[floating[Any]]]
		Tempo estimate or tempo trajectory in beats per minute.

	Mathematics
	-----------
	dominant tempo selection : equation
	```
		Let T(t, τ) ≜ tempogram value at frame t and tempo τ
			Θ ≜ admissible tempo set in beats per minute
			τ̂(t) ≜ dominant local tempo

		τ̂(t) = argmax_(τ ∈ Θ) T(t, τ)
	```

	tempo aggregation : equation
	```
		Let Γ ≜ set of analysis frames

		τ̂_global = aggregate({τ̂(t) : t ∈ Γ})
	```

	References
	----------
	[1] Grosche, P., Müller, M., & Kurth, F. (2010). Cyclic tempogram — a
		mid-level tempo representation for music signals. Proceedings of the IEEE International
		Conference on Acoustics, Speech and Signal Processing, 5522–5525.
		https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf
	[2] Müller, M., Ellis, D. P. W., Klapuri, A., & Richard, G. (2011). Signal
		processing for music analysis. IEEE Journal of Selected Topics in Signal Processing, 5(6),
		1088–1110. https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf
	[3] `librosa.feature.tempo`
		https://librosa.org/doc/latest/generated/librosa.feature.tempo.html
	"""
	tempogram: ArrayAspect = analyzeTempogram(waveform, sampleRate)
	return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments)

@registrationAudioAspect('Tempo mean')
def analyzeTempoMean(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> float:
	"""Aspect 'Tempo mean': mean tempo estimate in beats per minute.

	Returns
	-------
	tempoMean : float
		Mean value of the tempo estimate in beats per minute.

	"""
	return float(analyzeTempo(waveform, sampleRate, **keywordArguments).mean().item())

def analyzeZeroCrossingRate(waveform: Audio, **keywordArguments: Any) -> ArrayAspectWaveformFramewise:
	"""Compute the zero-crossing rate of the waveform.

	(AI generated docstring)

	You can use this function to measure how often `waveform` changes sign within each analysis frame.
	Higher values indicate more frequent sign alternations, which often accompany noisier or
	higher-frequency content [1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose sign changes are counted frame by frame.
	keywordArguments : Any
		Additional keyword arguments passed to `librosa.feature.zero_crossing_rate`.

	Returns
	-------
	zeroCrossingRate : ArrayAspectWaveformFramewise
		Framewise zero-crossing rate (ratio of sign changes per frame).

	Mathematics
	-----------
	framewise zero-crossing rate : equation
	```
		Let xₜ[h] ≜ h-th waveform sample in frame t
			N ≜ number of samples in frame t
			sgn(·) ≜ sign function

		ZCR(t) = (1 / (2(N - 1)))
				∑_(h = 1)^(N - 1) |sgn(xₜ[h]) - sgn(xₜ[h - 1])|
	```

	References
	----------
	[1] Panagiotakis, C., & Tziritas, G. (2005). A speech/music discriminator
		based on RMS and zero-crossings. IEEE Transactions on Multimedia, 7(1), 155–166.
		https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf
	[2] Zero-crossing rate. Introduction to Speech Processing.
		https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html
	[3] `librosa.feature.zero_crossing_rate`
		https://librosa.org/doc/latest/generated/librosa.feature.zero_crossing_rate.html
	"""
	return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments)

@registrationAudioAspect('Zero Crossing Rate mean')
def analyzeZeroCrossingRateMean(waveform: Audio, **keywordArguments: Any) -> float:
	"""Aspect 'Zero Crossing Rate mean': mean framewise zero-crossing rate.

	Returns
	-------
	zeroCrossingRateMean : float
		Mean value of the framewise zero-crossing rate.

	"""
	return float(analyzeZeroCrossingRate(waveform, **keywordArguments).mean().item())
