# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio.registry import registrationAudioAspect
from numpy import dtype, ndarray
from typing import Any, TYPE_CHECKING
import librosa
import numpy

if TYPE_CHECKING:
	from analyzeAudio import ArrayAspect, ArrayAspectSpectrogramFramewise, SpectrogramMagnitude, SpectrogramPower
	from numpy import float32

def analyzeRMSSpectrogram(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute framewise root-mean-square magnitude from a spectrogram.

	(AI generated docstring)

	You can use this function to compute per-frame root-mean-square (RMS) magnitude from
	`spectrogramMagnitude`. This is a thin wrapper around `librosa.feature.rms` that uses the
	spectrogram input path instead of deriving a spectrogram from a waveform.

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose framewise RMS magnitude is measured.
	keywordArguments : Any
		Additional keyword arguments passed to `librosa.feature.rms`.

	Returns
	-------
	rootMeanSquare : ArrayAspectSpectrogramFramewise
		Framewise root-mean-square magnitude.

	"""
	return librosa.feature.rms(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('RMS Spectrogram mean')
def analyzeRMSSpectrogramMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'RMS Spectrogram mean': mean framewise RMS magnitude.

	Returns
	-------
	rmsMean : float
		Arithmetic mean of the framewise RMS magnitude.

	"""
	return float(analyzeRMSSpectrogram(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeRMSSpectrogram_dB(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute framewise RMS spectrogram magnitude in decibels.

	(AI generated docstring)

	You can use this function to convert framewise RMS spectrogram magnitudes into decibel units
	using a 20 * log10 mapping. The function obtains linear RMS values from
	`analyzeRMSSpectrogram` and applies a logarithmic mapping.

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose framewise RMS magnitude is measured.
	keywordArguments : Any
		Additional arguments forwarded to `analyzeRMSSpectrogram`.

	Returns
	-------
	rootMeanSquare_dB : ArrayAspectSpectrogramFramewise
		Framewise RMS spectrogram magnitude in decibels.

	"""
	rootMeanSquare: ArrayAspectSpectrogramFramewise = analyzeRMSSpectrogram(spectrogramMagnitude, **keywordArguments)
	return 20 * numpy.log10(rootMeanSquare, where=(rootMeanSquare != 0), out=None)

@registrationAudioAspect('RMS Spectrogram dB mean')
def analyzeRMSSpectrogram_dBMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'RMS Spectrogram dB mean': mean framewise RMS spectrogram magnitude in decibels.

	Returns
	-------
	rootMeanSquare_dBMean : float
		Mean value of the time-varying RMS spectrogram magnitude in decibels.

	"""
	return float(analyzeRMSSpectrogram_dB(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeChromagram(spectrogramPower: SpectrogramPower, sampleRate: int, **keywordArguments: Any) -> ndarray[tuple[int, int, int], dtype[float32]]:
	"""Compute octave-equivalent pitch-class energy over time.

	(AI generated docstring)

	You can use this function to summarize tonal content as one chroma vector per analysis frame. The
	function folds octave-equivalent spectral energy into a 12-class pitch representation so that
	notes separated by octaves contribute to the same chroma bin [1][2].

	Parameters
	----------
	spectrogramPower : SpectrogramPower
		Power-domain spectral representation whose energy is folded into pitch classes.
	sampleRate : int
		Sampling rate of the analyzed signal in hertz.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``librosa.feature.chroma_stft``.

	Returns
	-------
	chromagram : numpy.ndarray
		Time-varying chroma representation with shape (12, n_frames) and dtype float32. Each column is
		a pitch-class (chroma) vector for a single analysis frame.

	Mathematics
	-----------
	pitch-class folding : equation
	```
		Let C(p, t) ≜ chromagram at pitch class p and frame t
			Q(q, t) ≜ pitch-resolved spectral energy
			M ≜ number of octave copies

		C(p, t) = ∑_(m = 0)^(M - 1) Q(p + 12m, t)
		p ∈ {0, …, 11}
	```

	References
	----------
	[1] Fujishima, T. (1999). Realtime chord recognition of musical sound:
		A system using Common Lisp Music. Proceedings of the International Computer Music Conference,
		464–467. https://ccrma.stanford.edu/~jos/mus423h/Real_Time_Chord_Recognition_Musical.html
	[2] Lee, K., & Slaney, M. (2006). Automatic chord recognition from audio
		using a HMM with supervised learning. Proceedings of the International Society for Music
		Information Retrieval, 133–137. https://ccrma.stanford.edu/~kglee/pubs/klee-ismir06.pdf
	"""
	return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **keywordArguments)

@registrationAudioAspect('Chromagram mean')
def analyzeChromagramMean(spectrogramPower: SpectrogramPower, sampleRate: int, **keywordArguments: Any) -> float:
	"""Aspect 'Chromagram mean': mean of the framewise chromagram.

	Returns
	-------
	chromagramMean : float
		Mean value of the time-varying chromagram.

	"""
	return float(analyzeChromagram(spectrogramPower, sampleRate, **keywordArguments).mean().item())

def analyzeSpectralContrast(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspect:
	"""Compute octave-band peak-to-valley contrast.

	(AI generated docstring)

	You can use this function to measure how strongly spectral peaks stand above spectral valleys in
	each octave band. High values indicate narrow-band, harmonic structure, while lower values
	indicate flatter or noisier spectral content [1].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation to be analyzed in octave bands.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``librosa.feature.spectral_contrast``.

	Returns
	-------
	spectralContrast : numpy.ndarray
		Framewise contrast values for each octave band with shape (n_bands + 1, n_frames). Each row
		corresponds to one octave band.

	Mathematics
	-----------
	octave-band partition : equation
	```
		Let Bᵢ ≜ i-th octave band
			f_min ≜ lower cutoff of the first band

		Bᵢ = [2^(i - 1) f_min, 2^i f_min)
	```

	peak-valley contrast : equation
	```
		Let Xᵢ ≜ magnitudes in octave band Bᵢ
			α ∈ (0, 1) ≜ neighborhood factor
			Pᵢ ≜ mean of the largest α|Xᵢ| values in Xᵢ
			Vᵢ ≜ mean of the smallest α|Xᵢ| values in Xᵢ

		SCᵢ = log(Pᵢ) - log(Vᵢ)
	```

	References
	----------
	[1] Jiang, D.-N., Lu, L., Zhang, H.-J., Tao, J.-H., & Cai, L.-H. (2002).
		Music type classification by spectral contrast feature. Proceedings of the IEEE International
		Conference on Multimedia and Expo, 113–116.
		https://hcsi.cs.tsinghua.edu.cn/Paper/Paper02/200218.pdf
	"""
	return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Contrast mean')
def analyzeSpectralContrastMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Contrast mean': mean of the framewise spectral contrast.

	Returns
	-------
	spectralContrastMean : float
		Mean value of the time-varying spectral contrast.

	"""
	return float(analyzeSpectralContrast(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralBandwidth(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute spectral spread around the framewise centroid.

	(AI generated docstring)

	You can use this function to measure how widely spectral energy is dispersed around the center of
	mass of each analysis frame. The returned value is the p-order bandwidth, which reduces to the
	usual standard-deviation form when p = 2 [1].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose spread is measured.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``librosa.feature.spectral_bandwidth``.

	Returns
	-------
	spectralBandwidth : ArrayAspectSpectrogramFramewise
		Framewise spectral bandwidth values with shape (1, n_frames).

	Mathematics
	-----------
	normalized spectral weights : equation
	```
		Let S_k(t) ≜ spectral magnitude at frequency f_k and frame t
			w_k(t) ≜ normalized spectral weight
			c(t) ≜ spectral centroid

		w_k(t) = S_k(t) / ∑_j S_j(t)
	```

	p-order bandwidth : equation
	```
		BW_p(t) = (∑_k w_k(t) |f_k - c(t)|^p)^(1/p)
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S.
		(2011). The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	centroid: ArrayAspectSpectrogramFramewise = analyzeSpectralCentroid(spectrogramMagnitude, **keywordArguments)
	return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **keywordArguments)

@registrationAudioAspect('Spectral Bandwidth mean')
def analyzeSpectralBandwidthMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Bandwidth mean': mean of the framewise spectral bandwidth.

	Returns
	-------
	spectralBandwidthMean : float
		Mean value of the time-varying spectral bandwidth.

	"""
	return float(analyzeSpectralBandwidth(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralCentroid(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute the frequency center of mass of each analysis frame.

	(AI generated docstring)

	You can use this function to estimate the balance point of spectral energy in each frame. Higher
	centroid values are commonly associated with brighter timbres [1][2].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose center of mass is measured.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``librosa.feature.spectral_centroid``.

	Returns
	-------
	spectralCentroid : ArrayAspectSpectrogramFramewise
		Framewise spectral centroid values with shape (1, n_frames).

	Mathematics
	-----------
	normalized spectral weights : equation
	```
		Let S_k(t) ≜ spectral magnitude at frequency f_k and frame t
			p_k(t) ≜ normalized spectral weight

		p_k(t) = S_k(t) / ∑_j S_j(t)
	```

	spectral centroid : equation
	```
		c(t) = ∑_k f_k p_k(t)
	```

	References
	----------
	[1] Grey, J. M., & Gordon, J. W. (1978). Perceptual effects of spectral
		modifications on musical timbres. Journal of the Acoustical Society of America, 63(5),
		1493–1500.
	[2] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S.
		(2011). The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Centroid mean')
def analyzeSpectralCentroidMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Centroid mean': mean of the framewise spectral centroid.

	Returns
	-------
	spectralCentroidMean : float
		Mean value of the time-varying spectral centroid.

	"""
	return float(analyzeSpectralCentroid(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralFlatness(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute the spectral flatness ratio for each analysis frame.

	(AI generated docstring)

	You can use this function to quantify how noise-like or tone-like a spectrum is. The flatness
	ratio compares the geometric mean and the arithmetic mean of the magnitude spectrum and returns
	values in the interval [0, 1], where values near 1 are noise-like and values near 0 are tone-like
	[1]. Use ``analyzeSpectralFlatness_dB`` to convert the ratio to decibels.

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose flatness is measured.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``librosa.feature.spectral_flatness``.

	Returns
	-------
	spectralFlatness : ArrayAspectSpectrogramFramewise
		Framewise spectral flatness values in the range [0, 1].

	Mathematics
	-----------
	spectral-flatness ratio : equation
	```
		Let X(k) ≜ spectral magnitude and K ≜ number of frequency bins

		SF(X) = (∏_(k = 0)^(K - 1) X(k))^(1/K) /
				((1/K) ∑_(k = 0)^(K - 1) X(k))
	```

	decibel mapping : equation
	```
		SF_dB(X) = 20 log10(SF(X))
	```

	References
	----------
	[1] Gray, A. H., & Markel, J. D. (1974). A spectral-flatness measure for
		studying the autocorrelation method of linear prediction of speech analysis. IEEE Transactions
		on Acoustics, Speech, and Signal Processing, 22(3), 207–217.
	"""
	return librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Flatness mean')
def analyzeSpectralFlatnessMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Flatness mean': mean of the framewise spectral flatness.

	Returns
	-------
	spectralFlatnessMean : float
		Mean value of the time-varying spectral flatness ratio.

	"""
	return float(analyzeSpectralFlatness(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralFlatness_dB(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> ArrayAspectSpectrogramFramewise:
	"""Compute spectral flatness in decibels.

	(AI generated docstring)

	You can use this function to quantify how noise-like or tone-like a spectrum is. This function
	converts the framewise flatness ratio into decibels using 20·log10. Values are in decibels and
	follow the conversion from the ratio returned by ``analyzeSpectralFlatness``.

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose flatness is measured.
	keywordArguments : Any
		Additional keyword arguments forwarded to ``analyzeSpectralFlatness`` and the underlying
		``librosa`` call.

	Returns
	-------
	spectralFlatnessDecibels : ArrayAspectSpectrogramFramewise
		Framewise spectral-flatness values expressed in decibels.

	Mathematics
	-----------
	spectral-flatness ratio : equation
	```
		Let X(k) ≜ spectral magnitude and K ≜ number of frequency bins

		SF(X) = (∏_(k = 0)^(K - 1) X(k))^(1/K) /
				((1/K) ∑_(k = 0)^(K - 1) X(k))
	```

	decibel mapping : equation
	```
		SF_dB(X) = 20 log10(SF(X))
	```

	References
	----------
	[1] Gray, A. H., & Markel, J. D. (1974). A spectral-flatness measure for
		studying the autocorrelation method of linear prediction of speech analysis. IEEE Transactions
		on Acoustics, Speech, and Signal Processing, 22(3), 207–217.
	"""
	spectralFlatness: ArrayAspectSpectrogramFramewise = analyzeSpectralFlatness(spectrogramMagnitude, **keywordArguments)
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0), out=None)

@registrationAudioAspect('Spectral Flatness dB mean')
def analyzeSpectralFlatness_dBMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Flatness dB mean': mean of the framewise spectral flatness in decibels.

	Returns
	-------
	spectralFlatnessMean : float
		Mean value of the time-varying spectral flatness in decibels.

	"""
	return float(analyzeSpectralFlatness_dB(spectrogramMagnitude, **keywordArguments).mean().item())
