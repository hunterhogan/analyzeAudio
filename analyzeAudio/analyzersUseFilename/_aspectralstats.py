# ty:ignore[invalid-return-type]
# pyright: reportReturnType=false
"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio.analyzersUseFilename._wideRange import ffprobeAllInclusiveCache
from analyzeAudio.registry import registrationAudioAspect
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from analyzeAudio import ArrayChannelData, ArrayOverallData
	from collections.abc import Callable
	from os import PathLike

arrayChannelDataEmpty: ArrayChannelData = numpy.array([], dtype=numpy.float64).reshape(0, 0)
arrayOverallDataEmpty: ArrayOverallData = numpy.array([], dtype=numpy.float64).reshape(0)

def analyzeSpectralCentroid(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral centroid trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect where spectral energy is centered in each analyzed frame
	of one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralCentroid : ArrayChannelData
		Framewise spectral-centroid values across analyzed frames.

	Mathematics
	-----------
	centroid : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			f_k ≜ frequency at bin k

		Centroidᵢ = (∑ₖ f_k Xᵢ(k)) / (∑ₖ Xᵢ(k))
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('centroid', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral centroid mean')
def analyzeSpectralCentroidMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral centroid mean': mean framewise spectral centroid.

	Returns
	-------
	spectralCentroidMean : float
		Mean value of the framewise spectral centroid.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralCentroid
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralCrest(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral crest of an audio file.

	(AI generated docstring)

	You can use this function to analyze how strongly one spectral peak dominates the average
	spectral magnitude in analyzed frames. [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralCrest : arrayAspectData
		Spectral crest across analyzed frames.

	Mathematics
	-----------
	spectral crest : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			K ≜ number of frequency bins

		Crestᵢ = maxₖ Xᵢ(k) / ((1/K) ∑_(k = 1)^K Xᵢ(k))
		SpectralCrestMean = (1/T) ∑_(i = 1)^T Crestᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('crest', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral crest mean')
def analyzeSpectralCrestMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral crest mean': mean framewise spectral crest.

	Returns
	-------
	spectralCrestMean : float
		Mean value of the framewise spectral crest.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralCrest
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralDecrease(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral decrease trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect how strongly spectrum levels tend to drop from lower
	frequency bins toward higher frequency bins in each analyzed frame [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralDecrease : ArrayChannelData
		Framewise spectral-decrease values across analyzed frames.

	Mathematics
	-----------
	spectral decrease : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			K ≜ number of frequency bins

		Decreaseᵢ = (∑_(k = 2)^K (Xᵢ(k) - Xᵢ(1)) / (k - 1)) / ∑_(k = 2)^K Xᵢ(k)
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('decrease', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral decrease mean')
def analyzeSpectralDecreaseMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral decrease mean': mean framewise spectral decrease.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralDecreaseMean : float
		Mean spectral decrease across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralDecrease
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralEntropy(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral entropy trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise spectral uncertainty for one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralEntropy : ArrayChannelData
		Framewise spectral-entropy values across analyzed frames.

	Mathematics
	-----------
	entropy : equation
	```
		Let Eᵢ(k) ≜ spectral energy at subband k of frame i
			pᵢ(k) = Eᵢ(k) / ∑ⱼ Eᵢ(j)

		Hᵢ = −∑ₖ pᵢ(k) log(pᵢ(k))
	```

	References
	----------
	[1] Shen, J.-L., Hung, J.-W., & Lee, L.-S. (1998). Robust entropy-based endpoint detection
		for speech recognition in noisy environments.
		https://www.ee.columbia.edu/~dpwe/papers/ShenHL98-endpoint.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('entropy', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral entropy mean')
def analyzeSpectralEntropyMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral entropy mean': mean framewise spectral entropy.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralEntropyMean : float
		Mean spectral entropy across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralEntropy
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralFlatness(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral flatness trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect how noise-like or tone-like each analyzed frame is by
	comparing geometric and arithmetic spectral means [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralFlatness : ArrayChannelData
		Framewise spectral-flatness values across analyzed frames.

	Mathematics
	-----------
	spectral flatness : equation
	```
		Let Xᵢ(k) > 0 ≜ spectral magnitude at frequency bin k of frame i
			K ≜ number of frequency bins

		Flatnessᵢ = (∏_(k = 1)^K Xᵢ(k))^(1/K) / ((1/K) ∑_(k = 1)^K Xᵢ(k))
	```

	References
	----------
	[1] Gray, A. H., & Markel, J. D. (1974). Distance measures for speech processing.
		IEEE Transactions on Acoustics, Speech, and Signal Processing, 24(5), 380–391.
		https://ieeexplore.ieee.org/document/1162647
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('flatness', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral flatness mean')
def analyzeSpectralFlatnessMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral flatness mean': mean framewise spectral flatness.

	Returns
	-------
	spectralFlatnessMean : float
		Mean value of the framewise spectral flatness.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralFlatness
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralFlux(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral flux trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect how strongly the spectrum changes from frame to frame in
	one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralFlux : ArrayChannelData
		Framewise spectral-flux values across analyzed frames.

	Mathematics
	-----------
	framewise spectral flux : equation
	```
		Let A(i, k) ≜ magnitude spectrum at frame i and frequency bin k
			δ ≜ small positive constant

		Fluxᵢ = (1/K) ∑ₖ [log(A(i, k) + δ) - log(A(i - 1, k) + δ)]²
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('flux', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral flux mean')
def analyzeSpectralFluxMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral flux mean': mean framewise spectral flux.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralFluxMean : float
		Mean spectral flux across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralFlux
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralKurtosis(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral kurtosis trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise peakedness of the spectral distribution for one
	audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralKurtosis : ArrayChannelData
		Framewise spectral-kurtosis values across analyzed frames.

	Mathematics
	-----------
	kurtosis : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			pᵢ(k) = Xᵢ(k) / ∑ⱼ Xᵢ(j)
			cᵢ ≜ spectral centroid of frame i
			σᵢ ≜ spectral spread of frame i

		Kurtosisᵢ = ∑ₖ ((f_k - cᵢ) / σᵢ)⁴ pᵢ(k)
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('kurtosis', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral kurtosis mean')
def analyzeSpectralKurtosisMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral kurtosis mean': mean framewise spectral kurtosis.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralKurtosisMean : float
		Mean spectral kurtosis across analyzed frames.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralKurtosis
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralMean(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the power spectral density trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise mean power-spectral values for one audio file.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	powerSpectralDensity : ArrayChannelData
		Framewise power-spectral-density values across analyzed frames.

	Mathematics
	-----------
	mean PSD : equation
	```
		Let Pᵢ(k) ≜ power spectral density at frequency bin k of frame i
			K ≜ number of frequency bins

		PSDᵢ = (1/K) ∑_(k = 1)^K Pᵢ(k)
	```
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('mean', arrayChannelDataEmpty)

@registrationAudioAspect('Power spectral density mean')
def analyzeSpectralMeanMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Power spectral density mean': mean framewise power spectral density.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	powerSpectralDensityMean : float
		Mean power spectral density across all frequency bins and analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralMean
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralRolloff(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral rolloff trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect the framewise rolloff boundary where cumulative spectral
	energy reaches a configured proportion [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralRolloff : ArrayChannelData
		Framewise spectral-rolloff values across analyzed frames.

	Mathematics
	-----------
	rolloff frequency : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude or energy at frequency bin k of frame i
			η ∈ (0, 1) ≜ cumulative-energy proportion

		f_rolloff,ᵢ = min { f_m : ∑_(k = 1)^m Xᵢ(k) ≥ η ∑_(k = 1)^K Xᵢ(k) }
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('rolloff', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral rolloff mean')
def analyzeSpectralRolloffMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral rolloff mean': mean framewise spectral rolloff frequency.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralRolloffMean : float
		Mean spectral rolloff frequency across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralRolloff
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSkewness(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral skewness trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise spectral asymmetry for one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSkewness : ArrayChannelData
		Framewise spectral-skewness values across analyzed frames.

	Mathematics
	-----------
	skewness : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			pᵢ(k) = Xᵢ(k) / ∑ⱼ Xᵢ(j)
			cᵢ ≜ spectral centroid of frame i
			σᵢ ≜ spectral spread of frame i

		Skewnessᵢ = ∑ₖ ((f_k - cᵢ) / σᵢ)³ pᵢ(k)
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('skewness', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral skewness mean')
def analyzeSpectralSkewnessMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral skewness mean': mean framewise spectral skewness.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSkewnessMean : float
		Mean spectral skewness across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralSkewness
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSlope(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral slope trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect the framewise linear trend of spectrum level over
	frequency [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSlope : ArrayChannelData
		Framewise spectral-slope values across analyzed frames.

	Mathematics
	-----------
	linear spectral trend : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency f_k of frame i
			K ≜ number of frequency bins

		Slopeᵢ = (K ∑ₖ f_k Xᵢ(k) - (∑ₖ f_k)(∑ₖ Xᵢ(k))) /
				(K ∑ₖ f_k² - (∑ₖ f_k)²)
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('slope', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral slope mean')
def analyzeSpectralSlopeMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral slope mean': mean framewise spectral slope.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSlopeMean : float
		Mean spectral slope across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralSlope
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSpread(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral spread trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise bandwidth around the spectral centroid [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSpread : ArrayChannelData
		Framewise spectral-spread values across analyzed frames.

	Mathematics
	-----------
	spectral spread : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency f_k of frame i
			pᵢ(k) = Xᵢ(k) / ∑ⱼ Xᵢ(j)
			cᵢ ≜ spectral centroid of frame i

		Spreadᵢ = √(∑ₖ (f_k - cᵢ)² pᵢ(k))
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('spread', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral spread mean')
def analyzeSpectralSpreadMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral spread mean': mean framewise spectral spread.

	Returns
	-------
	spectralSpreadMean : float
		Mean value of the framewise spectral spread.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralSpread
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralVariance(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the spectral variance trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise dispersion of spectral power around each frame's
	mean spectral value [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralVariance : ArrayChannelData
		Framewise spectral-variance values across analyzed frames.

	Mathematics
	-----------
	framewise spectral variance : equation
	```
		Let Pᵢ(k) ≜ power spectral density at frequency bin k of frame i
			μᵢ ≜ mean PSD of frame i
			K ≜ number of frequency bins

		Varianceᵢ = (1/K) ∑_(k = 1)^K (Pᵢ(k) − μᵢ)²
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('variance', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral variance mean')
def analyzeSpectralVarianceMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral variance mean': mean framewise spectral variance.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralVarianceMean : float
		Mean spectral variance across analyzed frames.
	"""
	theArrayCallable: Callable[[str | PathLike[Any]], ArrayChannelData] = analyzeSpectralVariance
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)
