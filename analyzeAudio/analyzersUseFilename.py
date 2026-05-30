"""Analyzers that use the filename of an audio file to analyze its audio data."""
# ruff: noqa: D103
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import registrationAudioAspect
from analyzeAudio.pythonator import pythonizeFFprobe
from functools import cache
from operator import getitem
from statistics import mean
from typing import Any, cast, TYPE_CHECKING
import numpy
import pathlib
import re as regex
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from os import PathLike

def _meanDB(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], filterChain: str) -> float | None:
	"""I use this shared comparison helper to average one decibel-valued aspect across analysis frames.

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

aspectName = 'Peak Signal-to-Noise Ratio mean'
@registrationAudioAspect(aspectName)
def getPSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Compute the mean peak signal-to-noise ratio between two audio files.

	(AI generated docstring)

	You can use this function to summarize how closely two audio files match with a peak-referenced
	decibel ratio. The registered audio aspect name is `Peak Signal-to-Noise Ratio mean` [1].

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
		http://dx.doi.org/10.14257/ijsia.2016.10.9.18
	"""
	filterChain: str = 'apsnr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

aspectName = 'SDR mean'
@registrationAudioAspect(aspectName)
def getSDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Compute the mean signal-to-distortion ratio between two audio files.

	(AI generated docstring)

	You can use this function to summarize the ratio between a target component and the remaining
	distortion energy when two audio files are compared. The registered audio aspect name is
	`SDR mean` [1].

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

aspectName = 'SI-SDR mean'
@registrationAudioAspect(aspectName)
def getSI_SDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Compute the mean scale-invariant signal-to-distortion ratio between two audio files.

	You can use this function to compare two audio files while forgiving only a single global scaling
	factor between them. The registered audio aspect name is `SI-SDR mean` [1].

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

@cache
def ffprobeShotgunAndCache(pathFilename: str | PathLike[Any]) -> dict[str, float]:
	"""I use this shared extractor to collect scalar audio aspects from one analysis pass.

	(AI generated docstring)

	I use this function to convert one structured analysis result into a dictionary of scalar audio
	aspects. Other wrappers read one key from the returned dictionary and expose it as one registered
	audio aspect.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	dictionaryAspectsAnalyzed : dict[str, float]
		Dictionary mapping aspect identifiers to scalar numeric values.
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
	filterChain += ["astats=metadata=1:measure_perchannel=Crest_factor+Zero_crossings_rate+Zero_crossings+Dynamic_range:measure_overall=all"]
	filterChain += ["aspectralstats"]
	filterChain += ["ebur128=metadata=1:framelog=quiet"]

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

	dictionaryAspectsAnalyzed: dict[str, float] = {}
	if 'aspectralstats' in FFprobeStructured:
		for keyName in FFprobeStructured['aspectralstats']:
			"""No matter how many channels, each keyName is `numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]`
			where `tuple[int, int]` is (channel, frame)
			NOTE (as of this writing) `registrar` can only understand the generic class `numpy.ndarray` and not more specific typing
			dictionaryAspectsAnalyzed[keyName] = FFprobeStructured['aspectralstats'][keyName]"""
			dictionaryAspectsAnalyzed[keyName] = numpy.mean(FFprobeStructured['aspectralstats'][keyName]).astype(float)
	if 'r128' in FFprobeStructured:
		for keyName in FFprobeStructured['r128']:
			dictionaryAspectsAnalyzed[keyName] = FFprobeStructured['r128'][keyName][-1]
	if 'astats' in FFprobeStructured:
		for keyName, arrayFeatureValues in cast('dict[str, numpy.ndarray[Any, Any]]', FFprobeStructured['astats']).items():
			dictionaryAspectsAnalyzed[keyName.split('.')[-1]] = numpy.mean(arrayFeatureValues[..., -1:None]).astype(float)

	return dictionaryAspectsAnalyzed

aspectName = 'Zero crossings'
@registrationAudioAspect(aspectName)
def analyzeZero_crossings(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean number of zero crossings in an audio file.

	(AI generated docstring)

	You can use this function to summarize how often the waveform changes sign across analyzed
	frames of one audio file. The registered audio aspect name is `Zero crossings` [1].

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
	return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings')

aspectName = 'Zero-crossings rate'
@registrationAudioAspect(aspectName)
def analyzeZero_crossings_rate(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean zero-crossing rate of an audio file.

	(AI generated docstring)

	You can use this function to summarize how densely sign changes occur within analyzed frames of
	one audio file. The registered audio aspect name is `Zero-crossings rate` [1].

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
	return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings_rate')

aspectName = 'DC offset'
@registrationAudioAspect(aspectName)
def analyzeDCoffset(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('DC_offset')

aspectName = 'Dynamic range'
@registrationAudioAspect(aspectName)
def analyzeDynamicRange(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Dynamic_range')

aspectName = 'Signal entropy'
@registrationAudioAspect(aspectName)
def analyzeSignalEntropy(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Entropy')

aspectName = 'Duration-samples'
@registrationAudioAspect(aspectName)
def analyzeNumber_of_samples(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Number_of_samples')

aspectName = 'Peak dB'
@registrationAudioAspect(aspectName)
def analyzePeak_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Peak_level')

aspectName = 'RMS total'
@registrationAudioAspect(aspectName)
def analyzeRMS_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_level')

aspectName = 'Crest factor'
@registrationAudioAspect(aspectName)
def analyzeCrest_factor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Crest_factor')

aspectName = 'RMS peak'
@registrationAudioAspect(aspectName)
def analyzeRMS_peak(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_peak')

aspectName = 'LUFS integrated'
@registrationAudioAspect(aspectName)
def analyzeLUFSintegrated(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the integrated programme loudness of an audio file.

	(AI generated docstring)

	You can use this function to obtain one gated, K-weighted loudness value for a complete audio
	file. The registered audio aspect name is `LUFS integrated` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	integratedLoudness : float | None
		Integrated loudness in LUFS.

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
	return ffprobeShotgunAndCache(pathFilename).get('I')

aspectName = 'LUFS loudness range'
@registrationAudioAspect(aspectName)
def analyzeLRA(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the loudness range of an audio file.

	(AI generated docstring)

	You can use this function to summarize the macroscopic spread of time-varying loudness in one
	audio file. The registered audio aspect name is `LUFS loudness range` [1][2].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessRange : float | None
		Loudness range in loudness units.

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
	return ffprobeShotgunAndCache(pathFilename).get('LRA')

aspectName = 'LUFS low'
@registrationAudioAspect(aspectName)
def analyzeLUFSlow(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the lower loudness bound used in loudness-range measurement.

	(AI generated docstring)

	You can use this function to obtain the lower percentile boundary used when loudness range is
	computed for one audio file. The registered audio aspect name is `LUFS low` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessLow : float | None
		Lower loudness bound in LUFS.

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
	return ffprobeShotgunAndCache(pathFilename).get('LRA.low')

aspectName = 'LUFS high'
@registrationAudioAspect(aspectName)
def analyzeLUFShigh(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the upper loudness bound used in loudness-range measurement.

	(AI generated docstring)

	You can use this function to obtain the upper percentile boundary used when loudness range is
	computed for one audio file. The registered audio aspect name is `LUFS high` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	loudnessHigh : float | None
		Upper loudness bound in LUFS.

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
	return ffprobeShotgunAndCache(pathFilename).get('LRA.high')

aspectName = 'Power spectral density mean'
@registrationAudioAspect(aspectName)
def analyzeMean(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('mean')

aspectName = 'Spectral variance mean'
@registrationAudioAspect(aspectName)
def analyzeVariance(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('variance')

aspectName = 'Spectral centroid mean'
@registrationAudioAspect(aspectName)
def analyzeCentroid(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral centroid of an audio file.

	(AI generated docstring)

	You can use this function to obtain one scalar summary of framewise spectral centroid from an
	audio file. The registered audio aspect name is `Spectral centroid mean`.

	Returns
	-------
	spectralCentroidMean : float | None
		Mean value of the framewise spectral centroid.

	See Also
	--------
	`analyzeAudio.analyzersUseSpectrogram.analyzeSpectralCentroid`
		Compute the full spectral-centroid trajectory and describe the center-of-mass formula.
	"""
	return ffprobeShotgunAndCache(pathFilename).get('centroid')

aspectName = 'Spectral spread mean'
@registrationAudioAspect(aspectName)
def analyzeSpread(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral spread of an audio file.

	(AI generated docstring)

	You can use this function to obtain one scalar summary of framewise spectral spread from an audio
	file. The registered audio aspect name is `Spectral spread mean`.

	Returns
	-------
	spectralSpreadMean : float | None
		Mean value of the framewise spectral spread.

	See Also
	--------
	`analyzeAudio.analyzersUseSpectrogram.analyzeSpectralBandwidth`
		Compute the full spectral-spread trajectory and describe the p-order definition.
	"""
	return ffprobeShotgunAndCache(pathFilename).get('spread')

aspectName = 'Spectral skewness mean'
@registrationAudioAspect(aspectName)
def analyzeSkewness(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral skewness of an audio file.

	(AI generated docstring)

	You can use this function to summarize the asymmetry of the framewise spectral distribution of one
	audio file. The registered audio aspect name is `Spectral skewness mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSkewnessMean : float | None
		Mean spectral skewness across analyzed frames.

	Mathematics
	-----------
	normalized spectral weights : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			pᵢ(k) = Xᵢ(k) / ∑ⱼ Xᵢ(j)
			cᵢ ≜ spectral centroid of frame i
			σᵢ ≜ spectral spread of frame i
	```

	skewness : equation
	```
		Skewnessᵢ = ∑ₖ ((f_k - cᵢ) / σᵢ)³ pᵢ(k)
		SpectralSkewnessMean = (1/T) ∑_(i = 1)^T Skewnessᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('skewness')

aspectName = 'Spectral kurtosis mean'
@registrationAudioAspect(aspectName)
def analyzeKurtosis(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral kurtosis of an audio file.

	(AI generated docstring)

	You can use this function to summarize how peaked the framewise spectral distribution is for one
	audio file. The registered audio aspect name is `Spectral kurtosis mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralKurtosisMean : float | None
		Mean spectral kurtosis across analyzed frames.

	Mathematics
	-----------
	normalized spectral weights : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			pᵢ(k) = Xᵢ(k) / ∑ⱼ Xᵢ(j)
			cᵢ ≜ spectral centroid of frame i
			σᵢ ≜ spectral spread of frame i
	```

	kurtosis : equation
	```
		Kurtosisᵢ = ∑ₖ ((f_k - cᵢ) / σᵢ)⁴ pᵢ(k)
		SpectralKurtosisMean = (1/T) ∑_(i = 1)^T Kurtosisᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

aspectName = 'Spectral entropy mean'
@registrationAudioAspect(aspectName)
def analyzeSpectralEntropy(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral entropy of an audio file.

	(AI generated docstring)

	You can use this function to summarize how concentrated or diffuse the framewise spectral energy
	distribution is for one audio file. The registered audio aspect name is `Spectral entropy mean`
	[1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralEntropyMean : float | None
		Mean spectral entropy across analyzed frames.

	Mathematics
	-----------
	spectral probabilities : equation
	```
		Let Eᵢ(k) ≜ spectral energy at subband k of frame i
			pᵢ(k) = Eᵢ(k) / ∑ⱼ Eᵢ(j)
	```

	entropy : equation
	```
		Hᵢ = −∑ₖ pᵢ(k) log(pᵢ(k))
		SpectralEntropyMean = (1/T) ∑_(i = 1)^T Hᵢ
	```

	References
	----------
	[1] Shen, J.-L., Hung, J.-W., & Lee, L.-S. (1998). Robust entropy-based endpoint detection
		for speech recognition in noisy environments.
		https://www.ee.columbia.edu/~dpwe/papers/ShenHL98-endpoint.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('entropy')

aspectName = 'Spectral flatness mean'
@registrationAudioAspect(aspectName)
def analyzeFlatness(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral flatness of an audio file.

	(AI generated docstring)

	You can use this function to obtain one scalar summary of framewise spectral flatness from an
	audio file. The registered audio aspect name is `Spectral flatness mean`.

	Returns
	-------
	spectralFlatnessMean : float | None
		Mean value of the framewise spectral flatness.

	See Also
	--------
	`analyzeAudio.analyzersUseSpectrogram.analyzeSpectralFlatness`
		Compute the full spectral-flatness trajectory and describe the geometric-to-arithmetic ratio.
	"""
	return ffprobeShotgunAndCache(pathFilename).get('flatness')

aspectName = 'Spectral crest mean'
@registrationAudioAspect(aspectName)
def analyzeCrest(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral crest of an audio file.

	(AI generated docstring)

	You can use this function to summarize how strongly one spectral peak dominates the average
	spectral magnitude in analyzed frames. The registered audio aspect name is `Spectral crest mean`
	[1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralCrestMean : float | None
		Mean spectral crest across analyzed frames.

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
	return ffprobeShotgunAndCache(pathFilename).get('crest')

aspectName = 'Spectral flux mean'
@registrationAudioAspect(aspectName)
def analyzeFlux(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral flux of an audio file.

	(AI generated docstring)

	You can use this function to summarize how strongly the spectrum changes from one frame to the
	next in one audio file. The registered audio aspect name is `Spectral flux mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralFluxMean : float | None
		Mean spectral flux across analyzed frames.

	Mathematics
	-----------
	framewise spectral flux : equation
	```
		Let A(i, k) ≜ magnitude spectrum at frame i and frequency bin k
			δ ≜ small positive constant

		Fluxᵢ = (1/K) ∑ₖ [log(A(i, k) + δ) - log(A(i - 1, k) + δ)]²
		SpectralFluxMean = (1/T) ∑_(i = 1)^T Fluxᵢ
	```

	References
	----------
	[1] Lu, L., Jiang, H., & Zhang, H. J. (2001). A robust audio classification and segmentation
		method. Microsoft Research Technical Report MSR-TR-2001-79.
		https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2001-79.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('flux')

aspectName = 'Spectral slope mean'
@registrationAudioAspect(aspectName)
def analyzeSlope(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral slope of an audio file.

	(AI generated docstring)

	You can use this function to summarize the linear trend of spectral magnitude over frequency in
	analyzed frames. The registered audio aspect name is `Spectral slope mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralSlopeMean : float | None
		Mean spectral slope across analyzed frames.

	Mathematics
	-----------
	linear spectral trend : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency f_k of frame i
			K ≜ number of frequency bins

		Slopeᵢ = (K ∑ₖ f_k Xᵢ(k) - (∑ₖ f_k)(∑ₖ Xᵢ(k))) /
				(K ∑ₖ f_k² - (∑ₖ f_k)²)
		SpectralSlopeMean = (1/T) ∑_(i = 1)^T Slopeᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('slope')

aspectName = 'Spectral decrease mean'
@registrationAudioAspect(aspectName)
def analyzeDecrease(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral decrease of an audio file.

	(AI generated docstring)

	You can use this function to summarize how fast spectral magnitude tends to fall after the first
	bins of each analyzed frame. The registered audio aspect name is `Spectral decrease mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralDecreaseMean : float | None
		Mean spectral decrease across analyzed frames.

	Mathematics
	-----------
	spectral decrease : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude at frequency bin k of frame i
			K ≜ number of frequency bins

		Decreaseᵢ = (∑_(k = 2)^K (Xᵢ(k) - Xᵢ(1)) / (k - 1)) / ∑_(k = 2)^K Xᵢ(k)
		SpectralDecreaseMean = (1/T) ∑_(i = 1)^T Decreaseᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('decrease')

aspectName = 'Spectral rolloff mean'
@registrationAudioAspect(aspectName)
def analyzeRolloff(pathFilename: str | PathLike[Any]) -> float | None:
	"""Compute the mean spectral rolloff frequency of an audio file.

	(AI generated docstring)

	You can use this function to summarize the frequency below which a chosen proportion of spectral
	energy is concentrated in analyzed frames. The registered audio aspect name is
	`Spectral rolloff mean` [1].

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path of the audio file to analyze.

	Returns
	-------
	spectralRolloffMean : float | None
		Mean spectral rolloff frequency across analyzed frames.

	Mathematics
	-----------
	rolloff frequency : equation
	```
		Let Xᵢ(k) ≜ spectral magnitude or energy at frequency bin k of frame i
			η ∈ (0, 1) ≜ cumulative-energy proportion

		f_rolloff,ᵢ = min { f_m : ∑_(k = 1)^m Xᵢ(k) ≥ η ∑_(k = 1)^K Xᵢ(k) }
		SpectralRolloffMean = (1/T) ∑_(i = 1)^T f_rolloff,ᵢ
	```

	References
	----------
	[1] Peeters, G., Giordano, B. L., Susini, P., Misdariis, N., & McAdams, S. (2011).
		The Timbre Toolbox: Extracting audio descriptors from musical signals. Journal of the
		Acoustical Society of America, 130(5), 2902–2916.
		https://www.mcgill.ca/mpcl/files/mpcl/peeters_2011_jasa.pdf
	"""
	return ffprobeShotgunAndCache(pathFilename).get('rolloff')

aspectName = 'Abs_Peak_count'
@registrationAudioAspect(aspectName)
def analyzeAbs_Peak_count(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Abs_Peak_count')

aspectName = 'Bit_depth'
@registrationAudioAspect(aspectName)
def analyzeBit_depth(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Bit_depth')

aspectName = 'Flat_factor'
@registrationAudioAspect(aspectName)
def analyzeFlat_factor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Flat_factor')

aspectName = 'Max_difference'
@registrationAudioAspect(aspectName)
def analyzeMax_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Max_difference')

aspectName = 'Max_level'
@registrationAudioAspect(aspectName)
def analyzeMax_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Max_level')

aspectName = 'Mean_difference'
@registrationAudioAspect(aspectName)
def analyzeMean_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Mean_difference')

aspectName = 'Min_difference'
@registrationAudioAspect(aspectName)
def analyzeMin_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Min_difference')

aspectName = 'Min_level'
@registrationAudioAspect(aspectName)
def analyzeMin_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Min_level')

aspectName = 'Noise_floor'
@registrationAudioAspect(aspectName)
def analyzeNoise_floor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Noise_floor')

aspectName = 'Noise_floor_count'
@registrationAudioAspect(aspectName)
def analyzeNoise_floor_count(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Noise_floor_count')

aspectName = 'Peak_count'
@registrationAudioAspect(aspectName)
def analyzePeak_count(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Peak_count')

aspectName = 'RMS_difference'
@registrationAudioAspect(aspectName)
def analyzeRMS_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_difference')

aspectName = 'RMS_trough'
@registrationAudioAspect(aspectName)
def analyzeRMS_trough(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_trough')
