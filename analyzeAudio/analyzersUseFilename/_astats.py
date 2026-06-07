# ty:ignore[invalid-return-type]
# pyright: reportReturnType=false
"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio.analyzersUseFilename._wideRange import ffprobeAllInclusiveCache
from analyzeAudio.registry import registrationAudioAspect
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from analyzeAudio import arrayChannelData, arrayOverallData
	from os import PathLike

arrayChannelDataEmpty: arrayChannelData = numpy.array([], dtype=numpy.float64).reshape(0, 0)
arrayOverallDataEmpty: arrayOverallData = numpy.array([], dtype=numpy.float64).reshape(0)

def analyzeAbs_Peak_count(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Aspect 'Abs_Peak_count': number of samples at the absolute peak amplitude.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	absPeakCount : float | None
		Number of samples at the absolute peak amplitude.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Overall.Abs_Peak_count', arrayOverallDataEmpty)

@registrationAudioAspect('Abs_Peak_count total')
def analyzeAbs_Peak_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeAbs_Peak_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeBit_depth(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Bit_depth': number of bits per audio sample.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	bitDepth : float | None
		Number of bits per sample.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Bit_depth', arrayChannelDataEmpty)

@registrationAudioAspect('Bit_depth mean')
def analyzeBit_depthMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeBit_depth(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeCrest_factor(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Crest_factor': ratio of peak amplitude to RMS level [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	crestFactor : float | None
		Ratio of peak amplitude to RMS level.

	Mathematics
	-----------
	crest factor : equation
	```
		Let x[n] ≜ sample n of the audio signal
			N ≜ number of samples

		Peak = max_(n ∈ {0, …, N − 1}) |x[n]|
		RMS = √((1/N) ∑_(n = 0)^(N − 1) x[n]²)
		CrestFactor = Peak / RMS
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Crest_factor', arrayChannelDataEmpty)

@registrationAudioAspect('Crest_factor mean')
def analyzeCrest_factorMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeCrest_factor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeDC_offset(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'DC_offset': mean sample value as a proportion of full scale.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	DCoffset : float | None
		Mean sample value as a proportion of the full-scale range.

	Mathematics
	-----------
	DC offset : equation
	```
		Let x[n] ≜ sample n of the audio signal
			N ≜ number of samples

		DCoffset = (1/N) ∑_(n = 0)^(N - 1) x[n]
	```
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('DC_offset', arrayChannelDataEmpty)

@registrationAudioAspect('DC_offset mean')
def analyzeDC_offsetMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeDC_offset(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeDynamic_range(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Dynamic_range': difference between peak level and noise floor.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dynamicRange : float | None
		Difference between peak level and noise floor in decibels.

	Mathematics
	-----------
	dynamic range : equation
	```
		Let Peak_level ≜ peak amplitude in dB
			Noise_floor ≜ estimated noise floor in dB

		DynamicRange = Peak_level − Noise_floor
	```
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Dynamic_range', arrayChannelDataEmpty)

@registrationAudioAspect('Dynamic_range overall')
def analyzeDynamic_rangeOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeDynamic_range(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeEntropy(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Entropy': Shannon entropy of the amplitude distribution [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	signalEntropy : float | None
		Shannon entropy of the sample amplitude distribution.

	Mathematics
	-----------
	Shannon entropy : equation
	```
		Let x[n] ≜ sample n of the audio signal
			p(x) ≜ normalized amplitude probability distribution

		Entropy = −∑ₓ p(x) log₂(p(x))
	```

	References
	----------
	[1] Shen, J.-L., Hung, J.-W., & Lee, L.-S. (1998). Robust entropy-based endpoint detection
		for speech recognition in noisy environments.
		https://www.ee.columbia.edu/~dpwe/papers/ShenHL98-endpoint.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Entropy', arrayChannelDataEmpty)

@registrationAudioAspect('Entropy mean')
def analyzeEntropyMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeEntropy(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeFlat_factor(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Flat_factor': mean proportion of flat (identical consecutive) samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	flatFactor : float | None
		Mean proportion of flat (identical consecutive) samples across analysis frames.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Flat_factor', arrayChannelDataEmpty)

@registrationAudioAspect('Flat_factor mean')
def analyzeFlat_factorMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeFlat_factor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMax_difference(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Max_difference': largest absolute difference between consecutive samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxDifference : float | None
		Maximum absolute difference between consecutive samples.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Max_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Max_difference overall')
def analyzeMax_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeMax_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMax_level(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Max_level': maximum sample value.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxLevel : float | None
		Maximum sample value.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Max_level', arrayChannelDataEmpty)

@registrationAudioAspect('Max_level overall')
def analyzeMax_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeMax_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMean_difference(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Mean_difference': mean absolute difference between consecutive samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	meanDifference : float | None
		Mean absolute difference between consecutive samples.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Mean_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Mean_difference mean')
def analyzeMean_differenceMean(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeMean_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMin_difference(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Min_difference': smallest absolute difference between consecutive samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minDifference : float | None
		Minimum absolute difference between consecutive samples.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Min_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Min_difference overall')
def analyzeMin_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeMin_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMin_level(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Min_level': minimum sample value.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minLevel : float | None
		Minimum sample value.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Min_level', arrayChannelDataEmpty)

@registrationAudioAspect('Min_level overall')
def analyzeMin_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeMin_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNoise_floor(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Noise_floor': estimated background noise floor level in dBFS.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloor : float | None
		Estimated noise floor level in decibels relative to full scale.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Noise_floor', arrayChannelDataEmpty)

@registrationAudioAspect('Noise_floor overall')
def analyzeNoise_floorOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeNoise_floor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNoise_floor_count(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Noise_floor_count': number of samples at or below the noise floor.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloorCount : float | None
		Number of samples at or below the noise floor level.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Noise_floor_count', arrayChannelDataEmpty)

@registrationAudioAspect('Noise_floor_count total')
def analyzeNoise_floor_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeNoise_floor_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNumber_of_samples(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Aspect 'Number_of_samples': total number of audio samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	numberOfSamples : float | None
		Total number of audio samples.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Overall.Number_of_samples', arrayOverallDataEmpty)

@registrationAudioAspect('Number_of_samples total')
def analyzeNumber_of_samplesTotal(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeNumber_of_samples(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzePeak_count(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Peak_count': number of samples at or above the peak level.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakCount : float | None
		Number of samples at or above the peak level.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Peak_count', arrayChannelDataEmpty)

@registrationAudioAspect('Peak_count total')
def analyzePeak_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzePeak_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzePeak_level(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Peak_level': maximum absolute sample amplitude in dBFS [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakDB : float | None
		Peak amplitude in decibels relative to full scale.

	Mathematics
	-----------
	peak level : equation
	```
		Let x[n] ≜ sample n of the audio signal
			N ≜ number of samples

		Peak_level = 20 log₁₀(max_(n ∈ {0, …, N − 1}) |x[n]|)
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Peak_level', arrayChannelDataEmpty)

@registrationAudioAspect('Peak_level overall')
def analyzePeak_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzePeak_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_difference(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'RMS_difference': RMS of differences between consecutive samples.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSdifference : float | None
		RMS of differences between consecutive samples.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('RMS_difference', arrayChannelDataEmpty)

@registrationAudioAspect('RMS_difference overall')
def analyzeRMS_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeRMS_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_level(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Aspect 'RMS_level': overall RMS level in dBFS [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSlevel : float | None
		Overall RMS level in decibels relative to full scale.

	Mathematics
	-----------
	root mean square : equation
	```
		Let x[n] ≜ sample n of the audio signal
			N ≜ number of samples

		RMS = √((1/N) ∑_(n = 0)^(N - 1) x[n]²)
		RMS_level = 20 log₁₀(RMS)
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Overall.RMS_level', arrayOverallDataEmpty)

@registrationAudioAspect('RMS_level overall')
def analyzeRMS_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeRMS_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeRMS_peak(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'RMS_peak': highest short-term RMS level in dBFS.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSpeak : float | None
		Highest short-term RMS level in decibels relative to full scale.

	Mathematics
	-----------
	block RMS peak : equation
	```
		Let xᵢ[n] ≜ sample n of analysis block i
			Nᵢ ≜ number of samples in block i
			T ≜ number of analysis blocks

		RMSᵢ = √((1/Nᵢ) ∑_(n = 0)^(Nᵢ − 1) xᵢ[n]²)
		RMS_peak = 20 log₁₀(max_(i ∈ {1, …, T}) RMSᵢ)
	```
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('RMS_peak', arrayChannelDataEmpty)

@registrationAudioAspect('RMS_peak overall')
def analyzeRMS_peakOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeRMS_peak(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_trough(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'RMS_trough': lowest short-term RMS level in dBFS.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMStrough : float | None
		Lowest short-term RMS level in decibels relative to full scale.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('RMS_trough', arrayChannelDataEmpty)

@registrationAudioAspect('RMS_trough overall')
def analyzeRMS_troughOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeRMS_trough(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeZero_crossings(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Zero_crossings': mean number of sign changes per frame [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossings : float | None
		Mean number of sign changes per analyzed frame.

	Mathematics
	-----------
	framewise zero-crossing count : equation
	```
		Let xᵢ[n] ≜ sample n of frame i
			sgn(x) = 1,  x ≥ 0
			sgn(x) = −1, x < 0

		ZCᵢ = (1/2) ∑_(n = 1)^(Nᵢ - 1) |sgn(xᵢ[n]) - sgn(xᵢ[n - 1])|
	```

	mean aggregation : equation
	```
		Let T ≜ number of analyzed frames

		ZeroCrossings = (1/T) ∑_(i = 1)^T ZCᵢ
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Zero_crossings', arrayChannelDataEmpty)

@registrationAudioAspect('Zero_crossings total')
def analyzeZero_crossingsTotal(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeZero_crossings(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeZero_crossings_rate(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Aspect 'Zero_crossings_rate': mean normalized zero-crossing rate per frame [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossingsRate : float | None
		Mean normalized zero-crossing count per analyzed frame.

	Mathematics
	-----------
	framewise zero-crossing rate : equation
	```
		Let xᵢ[n] ≜ sample n of frame i
			sgn(x) = 1,  x ≥ 0
			sgn(x) = −1, x < 0

		ZCRᵢ = (1/(2Nᵢ)) ∑_(n = 1)^(Nᵢ - 1) |sgn(xᵢ[n]) - sgn(xᵢ[n - 1])|
	```

	mean aggregation : equation
	```
		Let T ≜ number of analyzed frames

		ZeroCrossingsRate = (1/T) ∑_(i = 1)^T ZCRᵢ
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Zero_crossings_rate', arrayChannelDataEmpty)

@registrationAudioAspect('Zero_crossings_rate overall')
def analyzeZero_crossings_rateOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeZero_crossings_rate(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect
