# pyright: reportArgumentType=false
# pyright: reportAssignmentType=false
# pyright: reportReturnType=false
# ty:ignore[invalid-argument-type]
# ty:ignore[invalid-assignment]
# ty:ignore[invalid-return-type]
"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio.analyzersUseFilename._wideRange import ffprobeAllInclusiveCache
from analyzeAudio.registry import registrationAudioAspect
from typing import Any, TYPE_CHECKING
import numpy

if TYPE_CHECKING:
	from analyzeAudio import ArrayChannelData, ArrayOverallData
	from os import PathLike

arrayOverallDataEmpty: ArrayOverallData = numpy.array([], dtype=numpy.float64).reshape(0)

# TODO one function to return one array with all LUFS aspects.

def analyzeTruePeakChannel(pathFilename: str | PathLike[Any]) -> ArrayChannelData:
	keepGoing: bool = True
	channel: int = 0
	listAspectChannels: list[ArrayOverallData] = [ffprobeAllInclusiveCache(pathFilename).get(f'true_peaks_ch{channel}', arrayOverallDataEmpty)]
	while keepGoing:
		channel += 1
		aspectChannel = ffprobeAllInclusiveCache(pathFilename).get(f'true_peaks_ch{channel}', None)
		if aspectChannel is not None:
			listAspectChannels.append(aspectChannel)
		else:
			keepGoing = False
	return numpy.stack(listAspectChannels, axis=0)

def analyzeTruePeak(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the true-peak trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise true-peak levels of one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	truePeak : ArrayOverallData
		Framewise true-peak values in dBTP.

	References
	----------
	[1] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('true_peak', arrayOverallDataEmpty)

@registrationAudioAspect('true_peak maximum')
def analyzeTruePeakOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'true_peak maximum': maximum framewise true-peak level in dBTP.

	Returns
	-------
	truePeakMaximum : float | None
		Maximum value of the framewise true-peak trajectory.

	"""
	arrayAspect = analyzeTruePeak(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect.max()
	else:
		aspect = None
	return aspect

def analyzeLUFSMomentary(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the LUFS momentary trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect short-window loudness values of one audio file [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	LUFSmomentary : ArrayOverallData
		Framewise momentary loudness values in LUFS.

	References
	----------
	[1] EBU Tech 3341. (2023). Loudness Metering: EBU Mode.
		https://tech.ebu.ch/docs/tech/tech3341.pdf
	[2] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('M', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS momentary maximum')
def analyzeLUFSMomentaryOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS momentary maximum': maximum momentary loudness.

	Returns
	-------
	LUFSmomentaryMaximum : float | None
		Maximum value of the framewise momentary-loudness trajectory.

	"""
	arrayAspect = analyzeLUFSMomentary(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect.max()
	else:
		aspect = None
	return aspect

def analyzeLUFSShortTerm(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the LUFS short-term trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect 3-second loudness values of one audio file [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	LUFSshortTerm : ArrayOverallData
		Framewise short-term loudness values in LUFS.

	References
	----------
	[1] EBU Tech 3342. (2023). Loudness range: A measure to supplement EBU R 128 loudness
		normalisation.
		https://tech.ebu.ch/docs/tech/tech3342.pdf
	[2] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('S', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS short-term maximum')
def analyzeLUFSShortTermOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS short-term maximum': maximum short-term loudness.

	Returns
	-------
	LUFSshortTermMaximum : float | None
		Maximum value of the framewise short-term-loudness trajectory.

	"""
	arrayAspect = analyzeLUFSShortTerm(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect.max()
	else:
		aspect = None
	return aspect

def analyzeLUFSIntegrated(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the integrated programme loudness of an audio file.

	(AI generated docstring)

	You can use this function to obtain one gated, K-weighted loudness value for a complete audio
	file. The aspect name is "LUFS integrated" [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	integratedLoudness : ArrayOverallData
		Framewise and cumulative integrated-loudness values in LUFS.

	Mathematics
	-----------
	gating-block loudness : equation
	```
		Let zᵢⱼ ≜ mean square of K-weighted channel i in gating block j
			Gᵢ ≜ channel weight
			lⱼ ≜ loudness of gating block j

		lⱼ = −0.691 + 10 log₁₀(∑ᵢ Gᵢ zᵢⱼ)
	```

	integrated loudness : equation
	```
		Let Γₐ = −70 LUFS
			Γᵣ ≜ absolute-gated loudness − 10 LU
			J_g = {j : lⱼ > Γₐ and lⱼ > Γᵣ}

		L_K = −0.691 + 10 log₁₀((1/|J_g|) ∑_(j ∈ J_g) ∑ᵢ Gᵢ zᵢⱼ)
	```

	References
	----------
	[1] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('I', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS integrated')
def analyzeLUFSIntegratedOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS integrated': gated K-weighted loudness of a complete file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	integratedLoudness : float | None
		Integrated loudness in LUFS.
	"""
	arrayAspect = analyzeLUFSIntegrated(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeLRA(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the loudness range of an audio file.

	(AI generated docstring)

	You can use this function to summarize the macroscopic spread of time-varying loudness in one
	audio file. The aspect name is "LUFS loudness range" [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessRange : ArrayOverallData
		Framewise and cumulative loudness-range values in loudness units.

	Mathematics
	-----------
	cascaded gating : equation
	```
		Let {Lⱼ} ≜ 3-second loudness values
			Γₐ = −70 LUFS
			Γᵣ = L_abs − 20 LU

		Keep only loudness values above Γₐ and Γᵣ.
	```

	percentile range : equation
	```
		L_low = Q₀.₁₀({Lⱼ})
		L_high = Q₀.₉₅({Lⱼ})
		LRA = L_high - L_low
	```

	References
	----------
	[1] EBU Tech 3342. (2023). Loudness range: A measure to supplement EBU R 128 loudness
		normalisation.
		https://tech.ebu.ch/docs/tech/tech3342.pdf
	[2] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('LRA', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS loudness range')
def analyzeLRAOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS loudness range': macroscopic spread of time-varying loudness [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessRange : float | None
		Loudness range in loudness units.
	"""
	arrayAspect = analyzeLRA(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeLUFSlow(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the lower loudness bound used in loudness-range measurement.

	(AI generated docstring)

	You can use this function to obtain the lower percentile boundary used when loudness range is
	computed for one audio file. The aspect name is "LUFS low" [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessLow : ArrayOverallData
		Framewise and cumulative lower-bound loudness values in LUFS.

	Mathematics
	-----------
	lower percentile bound : equation
	```
		Let {Lⱼ} ≜ gated loudness values used for loudness-range computation

		LUFS_low = Q₀.₁₀({Lⱼ})
	```

	References
	----------
	[1] EBU Tech 3342. (2023). Loudness range: A measure to supplement EBU R 128 loudness
		normalisation.
		https://tech.ebu.ch/docs/tech/tech3342.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('LRA.low', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS low')
def analyzeLUFSlowOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS low': lower percentile bound of the loudness range [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessLow : float | None
		Lower loudness bound in LUFS.
	"""
	arrayAspect = analyzeLUFSlow(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect

def analyzeLUFShigh(pathFilename: str | PathLike[Any]) -> ArrayOverallData:
	"""Compute the upper loudness bound used in loudness-range measurement.

	(AI generated docstring)

	You can use this function to obtain the upper percentile boundary used when loudness range is
	computed for one audio file. The aspect name is "LUFS high" [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessHigh : ArrayOverallData
		Framewise and cumulative upper-bound loudness values in LUFS.

	Mathematics
	-----------
	upper percentile bound : equation
	```
		Let {Lⱼ} ≜ gated loudness values used for loudness-range computation

		LUFS_high = Q₀.₉₅({Lⱼ})
	```

	References
	----------
	[1] EBU Tech 3342. (2023). Loudness range: A measure to supplement EBU R 128 loudness
		normalisation.
		https://tech.ebu.ch/docs/tech/tech3342.pdf
	"""
	return ffprobeAllInclusiveCache(pathFilename).get('LRA.high', arrayOverallDataEmpty)

@registrationAudioAspect('LUFS high')
def analyzeLUFShighOverall(pathFilename: str | PathLike[Any]) -> float | None:
	"""Aspect 'LUFS high': upper percentile bound of the loudness range [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessHigh : float | None
		Upper loudness bound in LUFS.
	"""
	arrayAspect = analyzeLUFShigh(pathFilename)
	if 0 < len(arrayAspect):
		aspect = arrayAspect[-1]
	else:
		aspect = None
	return aspect
