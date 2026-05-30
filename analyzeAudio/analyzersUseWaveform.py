"""Analyzers that use the waveform of audio data."""
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from typing import Any, TYPE_CHECKING
import librosa
import numpy

if TYPE_CHECKING:
	from analyzeAudio import Audio, libturd

def analyzeTempogram(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> libturd:
	"""Compute a local autocorrelation tempogram from the waveform.

	(AI generated docstring)

	You can use this function to measure how strongly different pulse periods are
	present over time in `waveform`. The function returns a time-varying tempo
	representation whose values reflect local periodicity on a beats-per-minute
	axis [1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose local periodic structure is analyzed.
	sampleRate : int
		Sampling rate of `waveform` in hertz.
	keywordArguments : Any
		Additional tempogram-analysis settings.

	Returns
	-------
	tempogram : libturd
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
		mid-level tempo representation for music signals. Proceedings of the
		IEEE International Conference on Acoustics, Speech and Signal Processing,
		5522–5525.
		https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf
	[2] Müller, M., Ellis, D. P. W., Klapuri, A., & Richard, G. (2011). Signal
		processing for music analysis. IEEE Journal of Selected Topics in Signal
		Processing, 5(6), 1088–1110.
		https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf
	"""
	return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)

aspectName = 'Tempogram mean'
@registrationAudioAspect(aspectName)
def analyzeTempogramMean(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> float:
	"""Compute the mean of the tempogram.

	(AI generated docstring)

	The registered audio aspect name is `Tempogram mean`.

	Returns
	-------
	tempogramMean : float
		Mean value of the time-varying tempogram.

	See Also
	--------
	`analyzeTempogram`
		Compute the full tempogram and describe the autocorrelation formulation.
	"""
	return float(analyzeTempogram(waveform, sampleRate, **keywordArguments).mean().item())

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
def analyzeRMS(waveform: Audio, **keywordArguments: Any) -> libturd:
	"""Compute root-mean-square level from the waveform in decibels.

	(AI generated docstring)

	You can use this function to summarize framewise signal level from
	`waveform`. The function measures root-mean-square amplitude for each
	analysis frame and expresses the result on a decibel scale [1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose framewise level is measured.
	keywordArguments : Any
		Additional RMS-analysis settings.

	Returns
	-------
	rootMeanSquareDecibels : libturd
		Framewise RMS level expressed in decibels.

	Mathematics
	-----------
	root-mean-square amplitude : equation
	```
		Let xₜ[k] ≜ k-th waveform sample in frame t
			N ≜ number of samples in frame t

		RMS(t) = √((1 / N) ∑_(k = 0)^(N - 1) xₜ[k]²)
	```

	decibel mapping : equation
	```
		L_RMS(t) = 20 log10(RMS(t))
	```

	References
	----------
	[1] Constantinescu, C., & Brad, R. (2023). An overview on sound features
		in time and frequency domain. International Journal of Advanced
		Statistics and IT&C for Economics and Life Sciences, 13(1), 51–56.
		https://reference-global.com/download/article/10.2478/ijasitels-2023-0006.pdf
	[2] Panagiotakis, C., & Tziritas, G. (2005). A speech/music discriminator
		based on RMS and zero-crossings. IEEE Transactions on Multimedia,
		7(1), 155–166.
		https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf
	"""
	arrayRMS: libturd = librosa.feature.rms(y=waveform, **keywordArguments)
	return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0), out=None)  # dB

aspectName = 'RMS from waveform mean'
@registrationAudioAspect(aspectName)
def analyzeRMSMean(waveform: Audio, **keywordArguments: Any) -> float:
	"""Compute the mean of the RMS level.

	(AI generated docstring)

	The registered audio aspect name is `RMS from waveform mean`.

	Returns
	-------
	rootMeanSquareMean : float
		Mean value of the time-varying RMS level in decibels.

	See Also
	--------
	`analyzeRMS`
		Compute the full RMS-level trajectory and describe the framewise formula.
	"""
	return float(analyzeRMS(waveform, **keywordArguments).mean().item())

def analyzeTempo(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> libturd:
	"""Estimate tempo in beats per minute from waveform periodicity.

	(AI generated docstring)

	You can use this function to estimate the dominant tempo implied by
	`waveform`. The function interprets the strongest periodicities of a
	tempogram as tempo candidates in beats per minute, and `keywordArguments`
	can request either a single estimate or a time-varying tempo trajectory
	[1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose predominant tempo is estimated.
	sampleRate : int
		Sampling rate of `waveform` in hertz.
	keywordArguments : Any
		Additional tempo-estimation settings.

	Returns
	-------
	tempo : libturd
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
		mid-level tempo representation for music signals. Proceedings of the
		IEEE International Conference on Acoustics, Speech and Signal Processing,
		5522–5525.
		https://www.audiolabs-erlangen.de/content/resources/MIR/tempogramtoolbox/2010_GroscheMuellerKurth_TempogramCyclic_ICASSP.pdf
	[2] Müller, M., Ellis, D. P. W., Klapuri, A., & Richard, G. (2011). Signal
		processing for music analysis. IEEE Journal of Selected Topics in Signal
		Processing, 5(6), 1088–1110.
		https://www.ee.columbia.edu/~dpwe/pubs/MuEKR11-spmus.pdf
	"""
	tempogram: libturd = analyzeTempogram(waveform, sampleRate)
	return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments)

aspectName = 'Tempo mean'
@registrationAudioAspect(aspectName)
def analyzeTempoMean(waveform: Audio, sampleRate: int, **keywordArguments: Any) -> float:
	"""Compute the mean of the tempo estimate.

	(AI generated docstring)

	The registered audio aspect name is `Tempo mean`.

	Returns
	-------
	tempoMean : float
		Mean value of the tempo estimate in beats per minute.

	See Also
	--------
	`analyzeTempo`
		Compute the full tempo estimate and describe the dominant-tempo model.
	"""
	return float(analyzeTempo(waveform, sampleRate, **keywordArguments).mean().item())

def analyzeZeroCrossingRate(waveform: Audio, **keywordArguments: Any) -> libturd:
	"""Compute the zero-crossing rate of the waveform.

	(AI generated docstring)

	You can use this function to measure how often `waveform` changes sign
	within each analysis frame. Higher values indicate more frequent sign
	alternations, which often accompany noisier or higher-frequency content
	[1][2].

	Parameters
	----------
	waveform : Audio
		Waveform whose sign changes are counted frame by frame.
	keywordArguments : Any
		Additional zero-crossing-analysis settings.

	Returns
	-------
	zeroCrossingRate : libturd
		Framewise zero-crossing rate.

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
		based on RMS and zero-crossings. IEEE Transactions on Multimedia,
		7(1), 155–166.
		https://www.csd.uoc.gr/~tziritas/papers/07tmm01-panagiotakis-proof.pdf
	[2] Zero-crossing rate. Introduction to Speech Processing.
		https://speechprocessingbook.aalto.fi/Representations/Zero-crossing_rate.html
	"""
	return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments)

aspectName = 'Zero-crossing rate mean'
@registrationAudioAspect(aspectName)
def analyzeZeroCrossingRateMean(waveform: Audio, **keywordArguments: Any) -> float:
	"""Compute the mean of the zero-crossing rate.

	(AI generated docstring)

	The registered audio aspect name is `Zero-crossing rate mean`.

	Returns
	-------
	zeroCrossingRateMean : float
		Mean value of the framewise zero-crossing rate.

	See Also
	--------
	`analyzeZeroCrossingRate`
		Compute the full zero-crossing-rate trajectory and describe the rate formula.
	"""
	return float(analyzeZeroCrossingRate(waveform, **keywordArguments).mean().item())
