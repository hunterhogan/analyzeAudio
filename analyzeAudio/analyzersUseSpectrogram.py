# ruff: noqa: D100
from __future__ import annotations

from analyzeAudio import BleedFull, BleedFullArray, ParametersMelSpectrogram
from analyzeAudio.registry import registrationAudioAspect
from typing import Any, TYPE_CHECKING
from typing_extensions import Unpack
import librosa
import numpy

if TYPE_CHECKING:
	from analyzeAudio import libturd, SpectrogramMagnitude, SpectrogramPower

def analyzeChromagram(spectrogramPower: SpectrogramPower, sampleRate: int, **keywordArguments: Any) -> libturd:
	"""Compute octave-equivalent pitch-class energy over time.

	(AI generated docstring)

	You can use this function to summarize tonal content as one chroma vector per
	analysis frame. The function folds octave-equivalent spectral energy into a
	12-class pitch representation so that notes separated by octaves contribute to
	the same chroma bin [1][2].

	Parameters
	----------
	spectrogramPower : SpectrogramPower
		Power-domain spectral representation whose energy is folded into pitch
		classes.
	sampleRate : int
		Sampling rate of the analyzed signal in hertz.
	keywordArguments : Any
		Additional chroma-analysis settings.

	Returns
	-------
	chromagram : libturd
		Time-varying chroma representation with one octave-folded pitch-class vector
		per analysis frame.

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
		A system using Common Lisp Music. Proceedings of the International
		Computer Music Conference, 464–467.
		https://ccrma.stanford.edu/~jos/mus423h/Real_Time_Chord_Recognition_Musical.html
	[2] Lee, K., & Slaney, M. (2006). Automatic chord recognition from audio
		using a HMM with supervised learning. Proceedings of the International
		Society for Music Information Retrieval, 133–137.
		https://ccrma.stanford.edu/~kglee/pubs/klee-ismir06.pdf
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

def analyzeSpectralContrast(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> libturd:
	"""Compute octave-band peak-to-valley contrast.

	(AI generated docstring)

	You can use this function to measure how strongly spectral peaks stand above
	spectral valleys in each octave band. High values indicate narrow-band,
	harmonic structure, while lower values indicate flatter or noisier spectral
	content [1].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation to be analyzed in octave bands.
	keywordArguments : Any
		Additional spectral-contrast settings.

	Returns
	-------
	spectralContrast : libturd
		Framewise contrast values for each octave band.

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
		Music type classification by spectral contrast feature. Proceedings of
		the IEEE International Conference on Multimedia and Expo, 113–116.
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

def analyzeSpectralBandwidth(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> libturd:
	"""Compute spectral spread around the framewise centroid.

	(AI generated docstring)

	You can use this function to measure how widely spectral energy is dispersed
	around the center of mass of each analysis frame. The returned value is the
	p-order bandwidth, which reduces to the usual standard-deviation form when
	p = 2 [1].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose spread is measured.
	keywordArguments : Any
		Additional bandwidth-analysis settings.

	Returns
	-------
	spectralBandwidth : libturd
		Framewise spectral bandwidth values.

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
		(2011). The Timbre Toolbox: Extracting audio descriptors from musical
		signals. Journal of the Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	centroid: libturd = analyzeSpectralCentroid(spectrogramMagnitude, **keywordArguments)
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

def analyzeSpectralCentroid(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> libturd:
	"""Compute the frequency center of mass of each analysis frame.

	(AI generated docstring)

	You can use this function to estimate the balance point of spectral energy in
	each frame. Higher centroid values are commonly associated with brighter
	timbres [1][2].

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose center of mass is measured.
	keywordArguments : Any
		Additional centroid-analysis settings.

	Returns
	-------
	spectralCentroid : libturd
		Framewise spectral centroid values.

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
		modifications on musical timbres. Journal of the Acoustical Society of
		America, 63(5), 1493–1500.
	[2] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S.
		(2011). The Timbre Toolbox: Extracting audio descriptors from musical
		signals. Journal of the Acoustical Society of America, 130(5), 2902–2916.
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

def analyzeSpectralFlatness(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> libturd:
	"""Compute spectral flatness in decibels.

	(AI generated docstring)

	You can use this function to quantify how noise-like or tone-like a spectrum
	is. The underlying flatness ratio compares geometric and arithmetic means of
	the spectrum [1], and this function expresses that ratio on a decibel scale.

	Parameters
	----------
	spectrogramMagnitude : SpectrogramMagnitude
		Magnitude-domain spectral representation whose flatness is measured.
	keywordArguments : Any
		Additional flatness-analysis settings.

	Returns
	-------
	spectralFlatnessDecibels : libturd
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
		studying the autocorrelation method of linear prediction of speech
		analysis. IEEE Transactions on Acoustics, Speech, and Signal Processing,
		22(3), 207–217.
	"""
	spectralFlatness: libturd = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments)
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0), out=None)  # dB

@registrationAudioAspect('Spectral Flatness mean')
def analyzeSpectralFlatnessMean(spectrogramMagnitude: SpectrogramMagnitude, **keywordArguments: Any) -> float:
	"""Aspect 'Spectral Flatness mean': mean of the framewise spectral flatness in decibels.

	Returns
	-------
	spectralFlatnessMean : float
		Mean value of the time-varying spectral flatness in decibels.

	"""
	return float(analyzeSpectralFlatness(spectrogramMagnitude, **keywordArguments).mean().item())

#======== Contest =============================================================

def analyzeBleedFullMelDB(spectrogramMagnitudeAlfa: SpectrogramMagnitude, spectrogramMagnitudeBeta: SpectrogramMagnitude
						, **keywordArguments: Unpack[ParametersMelSpectrogram]) -> BleedFullArray:
	"""Separate mel-scaled dB excess and deficit between two spectrograms.

	You can use this function to compare two magnitude spectrograms as source-separation balance data.
	The function returns one array for places where `spectrogramMagnitudeBeta` has more mel-scaled dB
	energy than `spectrogramMagnitudeAlfa` and one array for places where `spectrogramMagnitudeBeta`
	has less mel-scaled dB energy than `spectrogramMagnitudeAlfa` [1][2].

	Parameters
	----------
	spectrogramMagnitudeAlfa : SpectrogramMagnitude
		Baseline magnitude spectrogram for the comparison.
	spectrogramMagnitudeBeta : SpectrogramMagnitude
		Second magnitude spectrogram whose added and missing mel-scaled dB content is measured against
		`spectrogramMagnitudeAlfa`.

	Returns
	-------
	bleedFullArray : BleedFullArray
		Named tuple containing `arrayBleed` with positive dB differences and `arrayFull` with negative
		dB differences. Zero dB differences are omitted.

	Mathematics
	-----------
	mel dB difference split : equation
	```
		Let Xₐ ≜ `spectrogramMagnitudeAlfa`,  Xᵦ ≜ `spectrogramMagnitudeBeta`
			F ≜ mel-scaled linear transformation matrix

		Mₐ = F · Xₐ
		Mᵦ = F · Xᵦ

		Aᵢ = 20 × log₁₀(Mₐᵢ)   ∀ Mₐᵢ ≠ 0, ∀ Aᵢ < |80.0|
		Bᵢ = 20 × log₁₀(Mᵦᵢ)   ∀ Mᵦᵢ ≠ 0, ∀ Bᵢ < |80.0|

		Δ = B − A

		Β = {Δᵢ ∈ Δ | Δᵢ > 0}
		Φ = {Δᵢ ∈ Δ | Δᵢ < 0}

		where  Β ≜ `arrayBleed`,  Φ ≜ `arrayFull`
	```

	Other Parameters
	----------------
	dtype : DTypeLike = numpy.float32
		Data type for the mel spectrograms and the returned difference arrays.
	fmax : float | None = None
		Maximum frequency (in hertz) included in the mel spectrograms. If `None`, the spectrograms are
		computed up to the Nyquist frequency.
	fmin : float = 0
		Minimum frequency (in hertz) included in the mel spectrograms.
	hop_length : int = 1024
		Number of samples between successive frames.
	htk : bool = False
		Whether to use HTK formula for mel scale.
	n_fft : int = 4096
		Length of the FFT window.
	n_mels : int = 512
		Number of Mel bands to generate.
	norm : float | Literal['slaney'] | None = "slaney"
		Normalization method for the mel spectrogram.
	power : float = 1.0
		Exponent for the magnitude spectrogram.
	sr : int = 44100
		Sample rate in hertz used to build the mel filter bank.
	top_db : float | None = 80.0
		Maximum decibel range retained after amplitude-to-decibel conversion.
	win_length : int = None
		Windowing function length for the FFT.
	window : str | tuple[Any, ...] | float | Callable[[int], ndarray] | ArrayLike = "hann"
		Windowing function for the FFT.

	References
	----------
	[1] csteinmetz1/auraloss issue #79. Enhancement ? New metric for source
		separation, measuring separately bleed and fullness in separated audio.
		https://github.com/csteinmetz1/auraloss/issues/79
	[2] ZFTurbo and jarredou. Music-Source-Separation-Training `bleed_full` metric.
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/c0197a0b2f1fffa8631779e1e92835a2e24d1c99/utils/metrics.py#L304-L385
	[3] librosa.feature.melspectrogram.
		https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html
	[4] librosa.amplitude_to_db.
		https://librosa.org/doc/latest/generated/librosa.amplitude_to_db.html
	"""
	parametersMelSpectrogram = ParametersMelSpectrogram(
		dtype=keywordArguments.get('dtype', numpy.float32)
		, fmax=keywordArguments.get('fmax')
		, fmin=keywordArguments.get('fmin', 0)
		, hop_length=keywordArguments.get('hop_length', 1024)
		, htk=keywordArguments.get('htk', False)
		, n_fft=keywordArguments.get('n_fft', 4096)
		, n_mels=keywordArguments.get('n_mels', 512)
		, norm=keywordArguments.get('norm', "slaney")
		, power=keywordArguments.get('power', 1.0)
		, sr=keywordArguments.get('sr', 44100)
		, win_length=keywordArguments.get('win_length', keywordArguments.get('n_fft', 4096))
		, window=keywordArguments.get('window', "hann")
	)

	top_db: float | None = keywordArguments.get('top_db', 80.0)

	spectrogramMagnitudeAlfa = librosa.feature.melspectrogram(S=spectrogramMagnitudeAlfa, **parametersMelSpectrogram)
	spectrogramMagnitudeBeta = librosa.feature.melspectrogram(S=spectrogramMagnitudeBeta, **parametersMelSpectrogram)
	spectrogramMagnitudeAlfa = librosa.amplitude_to_db(spectrogramMagnitudeAlfa, ref=1.0, top_db=top_db)  # pyright: ignore[reportUnknownMemberType]
	spectrogramMagnitudeBeta = librosa.amplitude_to_db(spectrogramMagnitudeBeta, ref=1.0, top_db=top_db)  # pyright: ignore[reportUnknownMemberType]

	return _bleedFullArrays(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta)

def _bleedFullArrays(spectrogramAlfa: SpectrogramMagnitude, spectrogramBeta: SpectrogramMagnitude) -> BleedFullArray:
	arrayDifferences = spectrogramBeta - spectrogramAlfa

	return BleedFullArray(arrayBleed=arrayDifferences[0 < arrayDifferences], arrayFull=arrayDifferences[arrayDifferences < 0])

def analyzeBleedFullMelDBMean(spectrogramMagnitudeAlfa: SpectrogramMagnitude, spectrogramMagnitudeBeta: SpectrogramMagnitude) -> BleedFull:
	"""Score mean mel-scaled dB-magnitude excess and deficit between two spectrograms.

	You can use this function to summarize source-separation balance as two higher-is-better scores.
	The function reports `bleed` from the mean positive dB-magnitude difference and `full` from the
	mean negative dB-magnitude difference, so added content and missing content remain separate
	[1][2].

	Parameters
	----------
	spectrogramMagnitudeAlfa : SpectrogramMagnitude
		Baseline magnitude spectrogram for the comparison.
	spectrogramMagnitudeBeta : SpectrogramMagnitude
		Second magnitude spectrogram whose added and missing mel-scaled dB-magnitude content is scored
		against `spectrogramMagnitudeAlfa`.

	Returns
	-------
	bleedFull : BleedFull
		Named tuple containing `bleed` and `full` reciprocal scores. Each score is `100 / (1 +
		|meanDifference|)` when the matching difference class exists and `0.0` when the matching
		difference class is empty.

	Mathematics
	-----------
	score mapping : equation
	```
		Let Δ⁺ ≜ positive mel-scaled dB-magnitude differences
			Δ⁻ ≜ negative mel-scaled dB-magnitude differences

		bleed = 100 / (1 + |mean(Δ⁺)|)
		full = 100 / (1 + |mean(Δ⁻)|)
	```

	Score Interpretation
	--------------------
	The reciprocal scale maps a zero mean magnitude to `100.0`. Because zero dB-magnitude differences
	are excluded from the selected difference class, nonempty difference classes usually score below
	`100.0`. Larger average dB-magnitude differences reduce the score toward `0.0`.

	See Also
	--------
	analyzeBleedFullMelDB
		Return the underlying positive and negative mel-scaled dB-magnitude difference arrays.

	References
	----------
	[1] csteinmetz1/auraloss issue #79. Enhancement ? New metric for source
		separation, measuring separately bleed and fullness in separated audio.
		https://github.com/csteinmetz1/auraloss/issues/79
	[2] ZFTurbo. Music-Source-Separation-Training `bleed_full` metric.
		https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/c0197a0b2f1fffa8631779e1e92835a2e24d1c99/utils/metrics.py#L304-L385
	[3] numpy.mean.
		https://numpy.org/doc/stable/reference/generated/numpy.mean.html
	"""
	bf: BleedFullArray = analyzeBleedFullMelDB(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta)

	if 0 < bf.arrayBleed.size:
		bleed: float = numpy.mean(bf.arrayBleed).item()
		bleed = 100 / (1 + abs(bleed))
	else:
		bleed = 0.0

	if 0 < bf.arrayFull.size:
		full: float = numpy.mean(bf.arrayFull).item()
		full = 100 / (1 + abs(full))
	else:
		full = 0.0

	return BleedFull(bleed=bleed, full=full)
