# ruff: noqa: DOC201
"""Analyzers that use the filename of an audio file to analyze its audio data."""
from __future__ import annotations

from analyzeAudio._beDRY import KValue
from analyzeAudio.registry import registrationAudioContest
from functools import cache
from statistics import mean
from typing import Any, TYPE_CHECKING
import pathlib
import re as regex
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from os import PathLike

#======== FFmpeg ==============================================================
# Why am I doing it this way?
# asdr, apsnr, and asisdr.
# `AVFILTER_FLAG_METADATA_ONLY`, so results are only available from stderr.
# apsnr probably has a bug.
# aap, anlmf, anlms, arls, axcorrelate
# Output is a stream of values, not metadata.
# ffmpeg -hide_banner -i "/apps/analyzeAudio/tests/dataSamples/SpeakSoftly_BrokenMan60sec/comparand_vocals_bad.wav" -i "/apps/analyzeAudio/tests/dataSamples/SpeakSoftly_BrokenMan60sec/reference_vocals.wav" -filter_complex "[0][1]axcorrelate" -codec pcm_f32le zz.wav

@cache
def _meanDB(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], filterChain: str) -> float:
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
	meanDecibels : float
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

# @registrationAudioContest('Peak Signal-to-Noise Ratio mean')
def analyzePSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float:
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

# @registrationAudioContest('SDR mean')
def analyzeSDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float:
	"""Aspect 'SDR mean': mean signal-to-distortion ratio between two audio files [1].

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.

	Returns
	-------
	SDRmean : float
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

# @registrationAudioContest('SI-SDR mean')
def analyzeSI_SDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float:
	"""Aspect 'SI-SDR mean': mean scale-invariant SDR between two audio files [1].

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path of the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path of the second audio file.

	Returns
	-------
	SI_SDRmean : float
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

# @registrationAudioContest('Peak Signal-to-Noise Ratio mean K')
def analyzePSNRmeanK(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize PSNR value using bounded logarithmic scaling."""
	return KValue(analyzePSNRmean(pathFilenameAlfa, pathFilenameBeta), K)

# @registrationAudioContest('SDR mean K')
def analyzeSDRmeanK(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize SDR value using bounded logarithmic scaling."""
	return KValue(analyzeSDRmean(pathFilenameAlfa, pathFilenameBeta), K)

# @registrationAudioContest('SI-SDR mean K')
def analyzeSI_SDRmeanK(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], K: float = 10.0) -> float:
	"""Normalize SI-SDR value using bounded logarithmic scaling."""
	return KValue(analyzeSI_SDRmean(pathFilenameAlfa, pathFilenameBeta), K)
