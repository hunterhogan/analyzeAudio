# ty:ignore[invalid-return-type]
# pyright: reportReturnType=false
# ruff: noqa: DOC201 ERA001 D103
"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio.pythonator import pythonizeFFprobe
from analyzeAudio.registry import registrationAudioAspect, registrationAudioContest
from functools import cache
from operator import getitem, neg
from statistics import mean
from typing import Any, cast, TYPE_CHECKING, TypeAlias
import math
import numpy
import pathlib
import re as regex
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from collections.abc import Callable
	from os import PathLike

arrayChannelData: TypeAlias = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]
arrayChannelDataEmpty: arrayChannelData = numpy.array([], dtype=numpy.float64).reshape(0, 0)
arrayOverallData: TypeAlias = numpy.ndarray[tuple[int], numpy.dtype[numpy.float64]]
arrayOverallDataEmpty: arrayOverallData = numpy.array([], dtype=numpy.float64).reshape(0)

#======== FFmpeg ==============================================================

def _meanDB(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], filterChain: str) -> float | None:
	"""I use this shared comparison function to average one decibel-valued aspect across analysis frames.

	(AI generated docstring)

	I use this function to compare two audio files with one framewise decibel-valued aspect and to
	return one scalar mean. The `filterChain` selects which registered comparison aspect is measured.

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.
	filterChain : str
		Identifier of the framewise comparison aspect to evaluate.

	Returns
	-------
	meanDecibels : float | None
		Arithmetic mean of the extracted decibel values.
	"""
	regexPattern = regex.compile(rf"^\[Parsed_{filterChain}_.* (.*) dB", regex.MULTILINE)
	commandLineFFmpeg = [
		'ffmpeg', '-hide_banner', '-loglevel', '32',
		'-i', f'{str(pathlib.Path(pathFilenameAlfa))}', '-i', f'{str(pathlib.Path(pathFilenameBeta))}',
		'-filter_complex', f'[0][1]{filterChain}', '-f', 'null', '-'
	]
	systemProcessFFmpeg = subprocess.run(commandLineFFmpeg, check=True, stderr=subprocess.PIPE)

	stderrFFmpeg: str = systemProcessFFmpeg.stderr.decode()

	return mean(map(float, regexPattern.findall(stderrFFmpeg)))

@registrationAudioContest('Peak Signal-to-Noise Ratio mean')
def analyzePSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Compute the mean peak signal-to-noise ratio between two audio files.

	(AI generated docstring)

	You can use this function to summarize how closely two audio files match with a peak-referenced
	decibel ratio. The registered audio aspect name is "Peak Signal-to-Noise Ratio mean" [1].

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.

	Returns
	-------
	PSNRmean : float | None
		Mean peak signal-to-noise ratio in decibels.

	Mathematics
	-----------
	framewise PSNR : equation
	```
		Let x[n] ≜ first audio signal sample
			y[n] ≜ second audio signal sample
			N ≜ number of samples in one analysis block
			x_peak ≜ peak sample magnitude of x

		MSE = (1/N) ∑_(n = 0)^(N - 1) (x[n] - y[n])²
		PSNR = 10 log₁₀(x_peak² / MSE)
	```

	mean aggregation : equation
	```
		Let P_t ≜ PSNR of analysis block t
			T ≜ number of analysis blocks

		PSNR_mean = (1/T) ∑_(t = 1)^T P_t
	```

	References
	----------
	[1] Hiary, H., Abu Dalhoum, A. L., Madain, A., Ortega, A., & Alfonseca, M. (2016).
		Blind audio watermarking technique based on two dimensional cellular automata.
		International Journal of Security and Its Applications, 10(9), 175–184.
		https://doi.org/10.14257/ijsia.2016.10.9.18
	"""
	filterChain: str = 'apsnr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

@registrationAudioContest('SDR mean')
def analyzeSDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Aspect 'SDR mean': mean signal-to-distortion ratio between two audio files [1].

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.

	Returns
	-------
	SDRmean : float | None
		Mean signal-to-distortion ratio in decibels.

	Mathematics
	-----------
	source-to-distortion ratio : equation
	```
		Let ŝ ≜ estimated signal
			s_target ≜ allowed target component
			e_interf ≜ interference component
			e_noise ≜ noise component
			e_artif ≜ artifact component

		ŝ = s_target + e_interf + e_noise + e_artif
		SDR = 10 log₁₀( ||s_target||₂² / ||e_interf + e_noise + e_artif||₂² )
	```

	mean aggregation : equation
	```
		Let D_t ≜ SDR of analysis block t
			T ≜ number of analysis blocks

		SDR_mean = (1/T) ∑_(t = 1)^T D_t
	```

	References
	----------
	[1] Vincent, E., Gribonval, R., & Févotte, C. (2006). Performance measurement in blind
		audio source separation. IEEE Transactions on Audio, Speech, and Language Processing,
		14(4), 1462–1469.
		https://www.irit.fr/~Cedric.Fevotte/publications/journals/ieee_asl_bsseval.pdf
	"""
	filterChain: str = 'asdr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

@registrationAudioContest('SI-SDR mean')
def analyzeSI_SDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Aspect 'SI-SDR mean': mean scale-invariant SDR between two audio files [1].

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.

	Returns
	-------
	SI_SDRmean : float | None
		Mean scale-invariant signal-to-distortion ratio in decibels.

	Mathematics
	-----------
	scale-invariant projection : equation
	```
		Let s ≜ first audio signal
			ŝ ≜ second audio signal
			α ≜ optimal scale factor

		α = <ŝ, s> / ||s||₂²
		s_target = αs
		e_res = ŝ - s_target
	```

	scale-invariant SDR : equation
	```
		SI-SDR = 10 log₁₀( ||s_target||₂² / ||e_res||₂² )
	```

	mean aggregation : equation
	```
		Let D_t ≜ SI-SDR of analysis block t
			T ≜ number of analysis blocks

		SI-SDR_mean = (1/T) ∑_(t = 1)^T D_t
	```

	References
	----------
	[1] Le Roux, J., Wisdom, S., Erdogan, H., & Hershey, J. R. (2019). SDR – half-baked or
		well done? Proceedings of the IEEE International Conference on Acoustics, Speech, and
		Signal Processing, 626–630.
		https://www.jonathanleroux.org/pdf/LeRoux2019ICASSP05sdr.pdf
	"""
	filterChain: str = 'asisdr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

#-------- K values ------------------------------------------------------------

def _KValue(unscaled: float, K: float = 10.0) -> float:
	unscaled = max(min(unscaled, K), neg(K) + 1e-6)
	return 100.0 * math.log1p(unscaled + K) / math.log1p(2 * K)

@registrationAudioContest('K Peak Signal-to-Noise Ratio mean')
def analyzeKPSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize PSNR value using bounded logarithmic scaling."""
	psnrMean = analyzePSNRmean(pathFilenameAlfa, pathFilenameBeta)
	if psnrMean:
		psnrMean = _KValue(psnrMean, K)
	return psnrMean

@registrationAudioContest('K SDR mean')
def analyzeKSDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize SDR value using bounded logarithmic scaling."""
	sdrMean = analyzeSDRmean(pathFilenameAlfa, pathFilenameBeta)
	if sdrMean:
		sdrMean = _KValue(sdrMean, K)
	return sdrMean

@registrationAudioContest('K SI-SDR mean')
def analyzeKSI_SDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize SI-SDR value using bounded logarithmic scaling."""
	si_sdrMean = analyzeSI_SDRmean(pathFilenameAlfa, pathFilenameBeta)
	if si_sdrMean:
		si_sdrMean = _KValue(si_sdrMean, K)
	return si_sdrMean

#======== FFprobe =============================================================

@cache
def _ffprobeShotgunAndCache(pathFilename: str | PathLike[Any]) -> dict[str, arrayChannelData | arrayOverallData]:
	"""I use this shared extractor to collect scalar audio aspects from one analysis pass.

	(AI generated docstring)

	I use this function to convert one structured analysis result into a dictionary of scalar and
	array audio aspects.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dictionaryAspects : dict[str, arrayChannelData | arrayOverallData]
		Dictionary mapping aspect identifiers to array numeric values.
	"""
	# for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
	# TODO Investigate, why `PureWindowsPath`?
	# `as_posix` because using lavfi bypasses the CLI sanitation/standardization functions, AND lavfi
	# either never works with NT paths or doesn't always work with NT paths, but POSIX is always safe
	# IF escaped properly. Does this work in POSIX filesystems? IDK. The "contest" aspects, like
	# SI-SDR use a different FFmpeg call that treats the filenames with
	# `str(pathlib.Path(pathFilenameBeta))`.
	pFn = pathlib.PureWindowsPath(pathFilename)
	lavfiPathFilename = pFn.drive.replace(":", "\\\\:") + pathlib.PureWindowsPath(pFn.root, pFn.relative_to(pFn.anchor)).as_posix()

	filterChain: list[str] = []
	filterChain += ["aspectralstats"]
	# by default length=0.05, 50ms. Set to 0.1, 100ms to match ebur128.
	# filterChain += ["astats=metadata=1:length=0.1:measure_perchannel=Crest_factor+Zero_crossings_rate+Zero_crossings+Dynamic_range:measure_overall=all"]
	# TODO FFmpeg might have a bug. per-channel `Abs_Peak_count` is not inserted in the metadata, but it is in the parsed_stats summary.
	filterChain += ["astats=metadata=1:length=0.1:measure_perchannel=all:measure_overall=Number_of_samples+RMS_level+Abs_Peak_count"]
	filterChain += ["ebur128=metadata=1:dualmono=true:framelog=verbose:peak=true"]

	entriesFFprobe: list[str] = ["frame_tags"]

	commandLineFFprobe: list[str] = [
		"ffprobe"
		, "-hide_banner"
		, "-f"
		, "lavfi"
		, f"amovie={lavfiPathFilename},{','.join(filterChain)}"
		, "-show_entries"
		, ':'.join(entriesFFprobe)
		, "-output_format"
		, "json=compact=1"
	]

	systemProcessFFprobe = subprocess.Popen(commandLineFFprobe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdoutFFprobe, _DISCARDstderr = systemProcessFFprobe.communicate()
	FFprobeStructured = getitem(pythonizeFFprobe(stdoutFFprobe.decode('utf-8')), -1)

	dictionaryAspects: dict[str, arrayChannelData | arrayOverallData] = {}
	if 'aspectralstats' in FFprobeStructured:
		"""No matter how many channels, each keyName is `numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]`
		where `tuple[int, int]` is (channel, frame)
		"""
		dictionaryAspects.update(FFprobeStructured['aspectralstats'])
	if 'r128' in FFprobeStructured:
		dictionaryAspects.update(FFprobeStructured['r128'])
		# index -1 is the cumulative value for LUFS I, low, and high and peak; plus, the array has 3 significant digits instead of the summary's 1.
	if 'astats' in FFprobeStructured:
		dictionaryAspects.update(FFprobeStructured['astats'])
		# TODO Crest_factor "standard ratio of peak to RMS level (note: not in dB)"

		# TODO Bit_depth: 'Bit_depth', 'Bit_depth2', 'Bit_depth3', 'Bit_depth4',

		# If I ran two passes of the filter, I could force per-frame values for all aspects with `reset=1`. astats is pretty fast because there are no transformations.

	return dictionaryAspects

#-------- aspectralstats ----------------------------------

def analyzeSpectralCentroid(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralCentroid : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('centroid', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral centroid mean')
def analyzeSpectralCentroidMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral centroid mean': mean framewise spectral centroid.

	Returns
	-------
	spectralCentroidMean : float
		Mean value of the framewise spectral centroid.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralCentroid
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralCrest(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	return _ffprobeShotgunAndCache(pathFilename).get('crest', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral crest mean')
def analyzeSpectralCrestMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral crest mean': mean framewise spectral crest.

	Returns
	-------
	spectralCrestMean : float
		Mean value of the framewise spectral crest.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralCrest
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralDecrease(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralDecrease : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('decrease', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralDecrease
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralEntropy(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Compute the spectral entropy trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise spectral uncertainty for one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralEntropy : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('entropy', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralEntropy
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralFlatness(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralFlatness : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('flatness', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral flatness mean')
def analyzeSpectralFlatnessMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral flatness mean': mean framewise spectral flatness.

	Returns
	-------
	spectralFlatnessMean : float
		Mean value of the framewise spectral flatness.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralFlatness
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralFlux(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralFlux : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('flux', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralFlux
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralKurtosis(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralKurtosis : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('kurtosis', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralKurtosis
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralMean(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Compute the power spectral density trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise mean power-spectral values for one audio file.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	powerSpectralDensity : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('mean', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralMean
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralRolloff(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralRolloff : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('rolloff', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralRolloff
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSkewness(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Compute the spectral skewness trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise spectral asymmetry for one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSkewness : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('skewness', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralSkewness
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSlope(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralSlope : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('slope', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralSlope
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralSpread(pathFilename: str | PathLike[Any]) -> arrayChannelData:
	"""Compute the spectral spread trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise bandwidth around the spectral centroid [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSpread : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('spread', arrayChannelDataEmpty)

@registrationAudioAspect('Spectral spread mean')
def analyzeSpectralSpreadMean(pathFilename: str | PathLike[Any]) -> float:
	"""Aspect 'Spectral spread mean': mean framewise spectral spread.

	Returns
	-------
	spectralSpreadMean : float
		Mean value of the framewise spectral spread.

	"""
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralSpread
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

def analyzeSpectralVariance(pathFilename: str | PathLike[Any]) -> arrayChannelData:
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
	spectralVariance : arrayAspectData
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
	return _ffprobeShotgunAndCache(pathFilename).get('variance', arrayChannelDataEmpty)

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
	theArrayCallable: Callable[[str | PathLike[Any]], arrayChannelData] = analyzeSpectralVariance
	return numpy.mean(theArrayCallable(pathFilename)).astype(float)

#-------- astats -------------------------------------------------------------------------------------

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
	return _ffprobeShotgunAndCache(pathFilename).get('Overall.Abs_Peak_count', arrayOverallDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Bit_depth', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Crest_factor', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('DC_offset', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Dynamic_range', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Entropy', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Flat_factor', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Max_difference', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Max_level', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Mean_difference', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Min_difference', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Min_level', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Noise_floor', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Noise_floor_count', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Overall.Number_of_samples', arrayOverallDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Peak_count', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Peak_level', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('RMS_difference', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Overall.RMS_level', arrayOverallDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('RMS_peak', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('RMS_trough', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Zero_crossings', arrayChannelDataEmpty)

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
	return _ffprobeShotgunAndCache(pathFilename).get('Zero_crossings_rate', arrayChannelDataEmpty)

@registrationAudioAspect('Zero_crossings_rate overall')
def analyzeZero_crossings_rateOverall(pathFilename: str | PathLike[Any]) -> float | None:
	arrayAspect = analyzeZero_crossings_rate(pathFilename)
	if 0 < len(arrayAspect):
		aspect: float | None = numpy.mean(arrayAspect[..., -1:None]).astype(float)
	else:
		aspect = None
	return aspect

#-------- ebur128 ------------------------------------------------------------------------------------

# TODO 'true_peaks_ch0', 'true_peaks_ch1', etc.
# TODO one function ro return one array with all LUFS aspects.

def analyzeTruePeak(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Compute the true-peak trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect framewise true-peak levels of one audio file [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	truePeak : arrayLUFSData
		Framewise true-peak values in dBTP.

	References
	----------
	[1] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return _ffprobeShotgunAndCache(pathFilename).get('true_peak', arrayOverallDataEmpty)

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

def analyzeLUFSMomentary(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Compute the LUFS momentary trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect short-window loudness values of one audio file [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	LUFSmomentary : arrayLUFSData
		Framewise momentary loudness values in LUFS.

	References
	----------
	[1] EBU Tech 3341. (2023). Loudness Metering: EBU Mode.
		https://tech.ebu.ch/docs/tech/tech3341.pdf
	[2] ITU-R BS.1770-5. (2023). Algorithms to measure audio programme loudness and
		true-peak audio level.
		https://www.itu.int/dms_pubrec/itu-r/rec/bs/R-REC-BS.1770-5-202311-I!!PDF-E.pdf
	"""
	return _ffprobeShotgunAndCache(pathFilename).get('M', arrayOverallDataEmpty)

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

def analyzeLUFSShortTerm(pathFilename: str | PathLike[Any]) -> arrayOverallData:
	"""Compute the LUFS short-term trajectory of an audio file.

	(AI generated docstring)

	You can use this function to inspect 3-second loudness values of one audio file [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	LUFSshortTerm : arrayLUFSData
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
	return _ffprobeShotgunAndCache(pathFilename).get('S', arrayOverallDataEmpty)

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

def analyzeLUFSIntegrated(pathFilename: str | PathLike[Any]) -> arrayOverallData:
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
	integratedLoudness : arrayLUFSData
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
	return _ffprobeShotgunAndCache(pathFilename).get('I', arrayOverallDataEmpty)

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

def analyzeLRA(pathFilename: str | PathLike[Any]) -> arrayOverallData:
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
	loudnessRange : arrayLUFSData
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
	return _ffprobeShotgunAndCache(pathFilename).get('LRA', arrayOverallDataEmpty)

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

def analyzeLUFSlow(pathFilename: str | PathLike[Any]) -> arrayOverallData:
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
	loudnessLow : arrayLUFSData
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
	return _ffprobeShotgunAndCache(pathFilename).get('LRA.low', arrayOverallDataEmpty)

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

def analyzeLUFShigh(pathFilename: str | PathLike[Any]) -> arrayOverallData:
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
	loudnessHigh : arrayLUFSData
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
	return _ffprobeShotgunAndCache(pathFilename).get('LRA.high', arrayOverallDataEmpty)

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
