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
	from os import PathLike

arrayChannelDataEmpty: ArrayChannelData = numpy.array([], dtype=numpy.float64).reshape(0, 0)
arrayOverallDataEmpty: ArrayOverallData = numpy.array([], dtype=numpy.float64).reshape(0)

def analyzeAbs_Peak_count(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the number of samples at the absolute peak amplitude.

	You can use this function to obtain the per-file overall counts of samples that equal the absolute
	peak amplitude as reported by ffprobe for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	absPeakCount : ArrayOverallData
		NumPy array containing overall counts per file, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Overall.Abs_Peak_count', arrayOverallDataEmpty)

@registrationAudioAspect('Abs_Peak_count total')
def analyzeAbs_Peak_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	"""Return the total number of samples at the absolute peak amplitude.

	You can use this function to obtain the scalar overall count derived from `analyzeAbs_Peak_count`
	for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	absPeakCount : float | None
		Scalar total count, or None when unavailable.
	"""
	arrayAspect = analyzeAbs_Peak_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeBit_depth(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the bit depth per channel for an audio file.

	You can use this function to obtain the per-channel bit depth values reported by ffprobe
	for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	bitDepth : ArrayChannelData
		NumPy array of per-channel bit depth values, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Bit_depth', arrayChannelDataEmpty)

@registrationAudioAspect('Bit_depth mean')
def analyzeBit_depthMean(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean bit depth across channels.

	You can use this function to obtain the mean bit depth computed from per-channel values
	returned by `analyzeBit_depth` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	bitDepth : float | None
		Mean bit depth across channels, or None when unavailable.
	"""
	arrayAspect = analyzeBit_depth(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeCrest_factor(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the crest factor (ratio of peak amplitude to RMS level).

	You can use this function to obtain per-channel crest factor values for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	crestFactor : ArrayChannelData
		NumPy array of per-channel crest factor values, or an empty array when unavailable.

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
	"""Compute the mean crest factor across channels.

	You can use this function to obtain the mean crest factor computed from per-channel values
	returned by `analyzeCrest_factor` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	crestFactor : float | None
		Mean crest factor across channels, or None when unavailable.
	"""
	arrayAspect = analyzeCrest_factor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeDC_offset(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the DC offset (mean sample value) as a proportion of full scale.

	You can use this function to obtain per-channel DC offset values for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	DCoffset : ArrayChannelData
		NumPy array of per-channel DC offset values, or an empty array when unavailable.

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
	"""Compute the mean DC offset across channels.

	You can use this function to obtain the mean DC offset computed from per-channel values
	returned by `analyzeDC_offset` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	DCoffset : float | None
		Mean DC offset across channels, or None when unavailable.
	"""
	arrayAspect = analyzeDC_offset(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeDynamic_range(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the dynamic range: difference between peak level and noise floor.

	You can use this function to obtain per-channel dynamic range values (dB) for the audio file
	at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dynamicRange : ArrayChannelData
		NumPy array of per-channel dynamic range values in decibels, or an empty array when
		unavailable.

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
	"""Compute the overall dynamic range (mean across channels).

	You can use this function to obtain the mean dynamic range computed from per-channel values
	returned by `analyzeDynamic_range` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dynamicRange : float | None
		Mean dynamic range in decibels, or None when unavailable.
	"""
	arrayAspect = analyzeDynamic_range(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeEntropy(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the Shannon entropy of the amplitude distribution.

	You can use this function to obtain per-channel entropy values for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	signalEntropy : ArrayChannelData
		NumPy array of per-channel Shannon entropy values, or an empty array when unavailable.

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
	"""Compute the mean Shannon entropy across channels.

	You can use this function to obtain the mean entropy computed from per-channel values
	returned by `analyzeEntropy` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	signalEntropy : float | None
		Mean Shannon entropy across channels, or None when unavailable.
	"""
	arrayAspect = analyzeEntropy(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeFlat_factor(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the mean proportion of flat (identical consecutive) samples.

	You can use this function to obtain per-channel per-frame proportions of identical consecutive
	samples for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	flatFactor : ArrayChannelData
		NumPy array of per-channel flat-sample proportions, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Flat_factor', arrayChannelDataEmpty)

@registrationAudioAspect('Flat_factor mean')
def analyzeFlat_factorMean(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean flat-sample proportion across channels.

	You can use this function to obtain the mean flat-sample proportion computed from per-channel
	values returned by `analyzeFlat_factor` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	flatFactor : float | None
		Mean proportion across channels, or None when unavailable.
	"""
	arrayAspect = analyzeFlat_factor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMax_difference(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the largest absolute difference between consecutive samples.

	You can use this function to obtain per-channel maximum absolute sample-to-sample
	differences for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxDifference : ArrayChannelData
		NumPy array of per-channel maximum absolute differences, or an empty array when
		unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Max_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Max_difference overall')
def analyzeMax_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall maximum absolute difference (mean across channels).

	You can use this function to obtain the mean of per-channel maximum absolute differences
	returned by `analyzeMax_difference` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxDifference : float | None
		Mean maximum absolute difference across channels, or None when unavailable.
	"""
	arrayAspect = analyzeMax_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMax_level(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the maximum sample value per channel.

	You can use this function to obtain per-channel maximum sample values for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxLevel : ArrayChannelData
		NumPy array of per-channel maximum sample values, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Max_level', arrayChannelDataEmpty)

@registrationAudioAspect('Max_level overall')
def analyzeMax_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall maximum sample value (mean across channels).

	You can use this function to obtain the mean maximum sample value computed from per-channel
	values returned by `analyzeMax_level` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	maxLevel : float | None
		Mean maximum sample value across channels, or None when unavailable.
	"""
	arrayAspect = analyzeMax_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMean_difference(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the mean absolute difference between consecutive samples.

	You can use this function to obtain per-channel mean absolute differences between consecutive
	samples for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	meanDifference : ArrayChannelData
		NumPy array of per-channel mean absolute differences, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Mean_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Mean_difference mean')
def analyzeMean_differenceMean(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean of mean absolute differences across channels.

	You can use this function to obtain the scalar mean computed from per-channel values
	returned by `analyzeMean_difference` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	meanDifference : float | None
		Mean value across channels, or None when unavailable.
	"""
	arrayAspect = analyzeMean_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMin_difference(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the smallest absolute difference between consecutive samples.

	You can use this function to obtain per-channel minimum absolute differences between consecutive
	samples for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minDifference : ArrayChannelData
		NumPy array of per-channel minimum absolute differences, or an empty array when
		unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Min_difference', arrayChannelDataEmpty)

@registrationAudioAspect('Min_difference overall')
def analyzeMin_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall minimum absolute difference (mean across channels).

	You can use this function to obtain the mean of per-channel minimum differences returned by
	`analyzeMin_difference` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minDifference : float | None
		Mean minimum absolute difference across channels, or None when unavailable.
	"""
	arrayAspect = analyzeMin_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeMin_level(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the minimum sample value per channel.

	You can use this function to obtain per-channel minimum sample values for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minLevel : ArrayChannelData
		NumPy array of per-channel minimum sample values, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Min_level', arrayChannelDataEmpty)

@registrationAudioAspect('Min_level overall')
def analyzeMin_levelOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall minimum sample value (mean across channels).

	You can use this function to obtain the mean minimum sample value computed from per-channel
	values returned by `analyzeMin_level` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	minLevel : float | None
		Mean minimum sample value across channels, or None when unavailable.
	"""
	arrayAspect = analyzeMin_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNoise_floor(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the estimated background noise floor level in dBFS per channel.

	You can use this function to obtain per-channel noise-floor estimates (dBFS) for the audio file
	at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloor : ArrayChannelData
		NumPy array of per-channel noise-floor levels in dBFS, or an empty array when
		unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Noise_floor', arrayChannelDataEmpty)

@registrationAudioAspect('Noise_floor overall')
def analyzeNoise_floorOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall noise floor level (mean across channels).

	You can use this function to obtain the mean noise-floor level computed from per-channel
	values returned by `analyzeNoise_floor` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloor : float | None
		Mean noise-floor level in dBFS across channels, or None when unavailable.
	"""
	arrayAspect = analyzeNoise_floor(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNoise_floor_count(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the number of samples at or below the estimated noise floor per channel.

	You can use this function to obtain per-channel counts of samples deemed to be at or below
	the estimated noise floor for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloorCount : ArrayChannelData
		NumPy array of per-channel counts, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Noise_floor_count', arrayChannelDataEmpty)

@registrationAudioAspect('Noise_floor_count total')
def analyzeNoise_floor_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	"""Return the total count of samples at or below the noise floor.

	You can use this function to obtain the scalar total count derived from
	`analyzeNoise_floor_count` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	noiseFloorCount : float | None
		Scalar total count, or None when unavailable.
	"""
	arrayAspect = analyzeNoise_floor_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeNumber_of_samples(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the total number of audio samples.

	You can use this function to obtain per-file total sample counts reported by ffprobe for the
	audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	numberOfSamples : ArrayOverallData
		NumPy array containing total sample counts per file, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Overall.Number_of_samples', arrayOverallDataEmpty)

@registrationAudioAspect('Number_of_samples total')
def analyzeNumber_of_samplesTotal(pathFilename: str | PathLike[Any]) -> float | None:
	"""Return the total number of audio samples as a scalar.

	You can use this function to obtain the scalar total number of samples derived from
	`analyzeNumber_of_samples` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	numberOfSamples : float | None
		Scalar total number of samples, or None when unavailable.
	"""
	arrayAspect = analyzeNumber_of_samples(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzePeak_count(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the number of samples at or above the peak level per channel.

	You can use this function to obtain per-channel counts of samples that meet or exceed the
	peak level for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakCount : ArrayChannelData
		NumPy array of per-channel peak counts, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('Peak_count', arrayChannelDataEmpty)

@registrationAudioAspect('Peak_count total')
def analyzePeak_countTotal(pathFilename: str | PathLike[Any]) -> float | None:
	"""Return the total number of samples at or above the peak level.

	You can use this function to obtain the scalar total count derived from
	`analyzePeak_count` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakCount : float | None
		Scalar total count, or None when unavailable.
	"""
	arrayAspect = analyzePeak_count(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzePeak_level(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the peak level (maximum absolute sample amplitude) in dBFS.

	You can use this function to obtain per-channel peak level values (dBFS) for the audio file
	at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakDB : ArrayChannelData
		NumPy array of per-channel peak levels in dBFS, or an empty array when unavailable.

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
	"""Compute the overall peak level (mean across channels).

	You can use this function to obtain the mean peak level computed from per-channel values
	returned by `analyzePeak_level` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	peakDB : float | None
		Mean peak level in dBFS, or None when unavailable.
	"""
	arrayAspect = analyzePeak_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_difference(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the RMS of differences between consecutive samples per channel.

	You can use this function to obtain per-channel RMS values of sample-to-sample differences
	for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSdifference : ArrayChannelData
		NumPy array of per-channel RMS differences, or an empty array when unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('RMS_difference', arrayChannelDataEmpty)

@registrationAudioAspect('RMS_difference overall')
def analyzeRMS_differenceOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall RMS difference (mean across channels).

	You can use this function to obtain the mean RMS difference computed from per-channel values
	returned by `analyzeRMS_difference` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSdifference : float | None
		Mean RMS difference across channels, or None when unavailable.
	"""
	arrayAspect = analyzeRMS_difference(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_level(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the overall RMS level in dBFS for the audio file.

	You can use this function to obtain per-file overall RMS level values reported by ffprobe for
	the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSlevel : ArrayOverallData
		NumPy array containing overall RMS level values per file, or an empty array when
		unavailable.

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
	"""Return the scalar overall RMS level for the audio file.

	You can use this function to obtain the final scalar RMS level derived from
	`analyzeRMS_level` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSlevel : float | None
		Scalar RMS level in dBFS, or None when unavailable.
	"""
	arrayAspect = analyzeRMS_level(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeRMS_peak(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the highest short-term RMS level per channel in dBFS.

	You can use this function to obtain per-channel short-term RMS peak values for the audio
	file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSpeak : ArrayChannelData
		NumPy array of per-channel short-term RMS peak values in dBFS, or an empty array when
		unavailable.

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
	"""Compute the overall short-term RMS peak (mean across channels).

	You can use this function to obtain the mean short-term RMS peak computed from per-channel
	values returned by `analyzeRMS_peak` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMSpeak : float | None
		Mean short-term RMS peak in dBFS, or None when unavailable.
	"""
	arrayAspect = analyzeRMS_peak(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeRMS_trough(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the lowest short-term RMS level per channel in dBFS.

	You can use this function to obtain per-channel short-term RMS trough values for the audio
	file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMStrough : ArrayChannelData
		NumPy array of per-channel short-term RMS trough values in dBFS, or an empty array when
		unavailable.
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('RMS_trough', arrayChannelDataEmpty)

@registrationAudioAspect('RMS_trough overall')
def analyzeRMS_troughOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the overall short-term RMS trough (mean across channels).

	You can use this function to obtain the mean short-term RMS trough computed from per-channel
	values returned by `analyzeRMS_trough` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	RMStrough : float | None
		Mean short-term RMS trough in dBFS, or None when unavailable.
	"""
	arrayAspect = analyzeRMS_trough(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeZero_crossings(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the mean number of sign changes (zero crossings) per analysis frame.

	You can use this function to obtain per-channel zero-crossing counts (mean per frame) for the
	audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossings : ArrayChannelData
		NumPy array of per-channel mean zero-crossing counts per frame, or an empty array when
		unavailable.

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
	"""Return the mean zero-crossing count across channels as a scalar.

	You can use this function to obtain the scalar mean zero-crossing count derived from
	`analyzeZero_crossings` for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossings : float | None
		Mean zero-crossing count, or None when unavailable.
	"""
	arrayAspect = analyzeZero_crossings(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

def analyzeZero_crossings_rate(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	"""Compute the mean normalized zero-crossing rate per analysis frame.

	You can use this function to obtain per-channel zero-crossing rate values (normalized by
	frame length) for the audio file at `pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossingsRate : ArrayChannelData
		NumPy array of per-channel normalized zero-crossing rates per frame, or an empty array
		when unavailable.

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
	"""Compute the overall normalized zero-crossing rate (mean across channels).

	You can use this function to obtain the mean normalized zero-crossing rate computed from
	per-channel values returned by `analyzeZero_crossings_rate` for the audio file at
	`pathFilename`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	zeroCrossingsRate : float | None
		Mean normalized zero-crossing rate, or None when unavailable.
	"""
	arrayAspect = analyzeZero_crossings_rate(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect
