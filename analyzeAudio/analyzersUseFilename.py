from .pythonator import pythonizeFFprobe
from analyzeAudio import registrationAudioAspect, cacheAudioAnalyzers
from os import PathLike
from statistics import mean
from typing import Any, cast
import cachetools
import numpy
import pathlib
import re as regex
import subprocess

@registrationAudioAspect('SI-SDR mean')
def getSI_SDRmean(pathFilenameAlpha: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""
	Calculate the mean Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between two audio files.
	This function uses FFmpeg to compute the SI-SDR between two audio files specified by their paths.
	The SI-SDR values are extracted from the FFmpeg output and their mean is calculated.
	Parameters:
		pathFilenameAlpha: Path to the first audio file.
		pathFilenameBeta: Path to the second audio file.
	Returns:
		SI_SDRmean: The mean SI-SDR value in decibels (dB).
	Raises:
		subprocess.CalledProcessError: If the FFmpeg command fails.
		ValueError: If no SI-SDR values are found in the FFmpeg output.
	"""
	commandLineFFmpeg = [
		'ffmpeg', '-hide_banner', '-loglevel', '32',
		'-i', f'{str(pathlib.Path(pathFilenameAlpha))}', '-i', f'{str(pathlib.Path(pathFilenameBeta))}',
		'-filter_complex', '[0][1]asisdr', '-f', 'null', '-'
	]
	systemProcessFFmpeg = subprocess.run(commandLineFFmpeg, check=True, stderr=subprocess.PIPE)

	stderrFFmpeg = systemProcessFFmpeg.stderr.decode()

	regexSI_SDR = regex.compile(r"^\[Parsed_asisdr_.* (.*) dB", regex.MULTILINE)

	listMatchesSI_SDR = regexSI_SDR.findall(stderrFFmpeg)
	SI_SDRmean = mean(float(match) for match in listMatchesSI_SDR)
	return SI_SDRmean

@cachetools.cached(cache=cacheAudioAnalyzers)
def ffprobeShotgunAndCache(pathFilename: str | PathLike[Any]) -> dict[str, float | numpy.ndarray]:
	# for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
	pFn = pathlib.PureWindowsPath(pathFilename)
	lavfiPathFilename = pFn.drive.replace(":", "\\\\:")+pathlib.PureWindowsPath(pFn.root,pFn.relative_to(pFn.anchor)).as_posix()

	filterChain: list[str] = []
	filterChain += ["astats=metadata=1:measure_perchannel=Crest_factor+Zero_crossings_rate+Dynamic_range:measure_overall=all"]
	filterChain += ["aspectralstats"]
	filterChain += ["ebur128=metadata=1:framelog=quiet"]

	entriesFFprobe = ["frame_tags"]

	commandLineFFprobe = [
		"ffprobe", "-hide_banner",
		"-f", "lavfi", f"amovie={lavfiPathFilename},{','.join(filterChain)}",
		"-show_entries", ':'.join(entriesFFprobe),
		"-output_format", "json=compact=1",
	]

	systemProcessFFprobe = subprocess.Popen(commandLineFFprobe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdoutFFprobe, _DISCARDstderr = systemProcessFFprobe.communicate()
	FFprobeStructured = pythonizeFFprobe(stdoutFFprobe.decode('utf-8'))[-1]

	dictionaryAspectsAnalyzed: dict[str, float | numpy.ndarray] = {}
	if 'aspectralstats' in FFprobeStructured:
		for keyName in FFprobeStructured['aspectralstats']:
			# No matter how many channels, each keyName is `numpy.ndarray[tuple[int, int], numpy.dtype[numpy.float64]]`
			# where `tuple[int, int]` is (channel, frame)
			# NOTE (as of this writing) `registrar` can only understand the generic class `numpy.ndarray` and not more specific typing
			dictionaryAspectsAnalyzed[keyName] = FFprobeStructured['aspectralstats'][keyName]
			# dictionaryAspectsAnalyzed[keyName] = numpy.mean(FFprobeStructured['aspectralstats'][keyName]).astype(float)
	if 'r128' in FFprobeStructured:
		for keyName in FFprobeStructured['r128']:
			dictionaryAspectsAnalyzed[keyName] = FFprobeStructured['r128'][keyName][-1]
	if 'astats' in FFprobeStructured:
		for keyName, arrayFeatureValues in cast(dict[str, numpy.ndarray[Any, Any]], FFprobeStructured['astats']).items():
			dictionaryAspectsAnalyzed[keyName.split('.')[-1]] = numpy.mean(arrayFeatureValues[..., -1:None]).astype(float)

	return dictionaryAspectsAnalyzed

@registrationAudioAspect('Zero-crossings rate')
def analyzeZero_crossings_rate(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings_rate')

@registrationAudioAspect('DC offset')
def analyzeDCoffset(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('DC_offset')

@registrationAudioAspect('Dynamic range')
def analyzeDynamicRange(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Dynamic_range')

@registrationAudioAspect('Signal entropy')
def analyzeSignalEntropy(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Entropy')

@registrationAudioAspect('Duration-samples')
def analyzeNumber_of_samples(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Number_of_samples')

@registrationAudioAspect('Peak dB')
def analyzePeak_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Peak_level')

@registrationAudioAspect('RMS total')
def analyzeRMS_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_level')

@registrationAudioAspect('Crest factor')
def analyzeCrest_factor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Crest_factor')

@registrationAudioAspect('RMS peak')
def analyzeRMS_peak(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_peak')

@registrationAudioAspect('LUFS integrated')
def analyzeLUFSintegrated(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('I')

@registrationAudioAspect('LUFS loudness range')
def analyzeLRA(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('LRA')

@registrationAudioAspect('LUFS low')
def analyzeLUFSlow(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('LRA.low')

@registrationAudioAspect('LUFS high')
def analyzeLUFShigh(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('LRA.high')

@registrationAudioAspect('Power spectral density')
def analyzeMean(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('mean')

@registrationAudioAspect('Spectral variance')
def analyzeVariance(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('variance')

@registrationAudioAspect('Spectral centroid')
def analyzeCentroid(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('centroid')

@registrationAudioAspect('Spectral spread')
def analyzeSpread(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('spread')

@registrationAudioAspect('Spectral skewness')
def analyzeSkewness(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('skewness')

@registrationAudioAspect('Spectral kurtosis')
def analyzeKurtosis(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

@registrationAudioAspect('Spectral entropy')
def analyzeSpectralEntropy(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('entropy')

@registrationAudioAspect('Spectral flatness')
def analyzeFlatness(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('flatness')

@registrationAudioAspect('Spectral crest')
def analyzeCrest(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('crest')

@registrationAudioAspect('Spectral flux')
def analyzeFlux(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('flux')

@registrationAudioAspect('Spectral slope')
def analyzeSlope(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('slope')

@registrationAudioAspect('Spectral decrease')
def analyzeDecrease(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('decrease')

@registrationAudioAspect('Spectral rolloff')
def analyzeRolloff(pathFilename: str | PathLike[Any]) -> numpy.ndarray:
	return ffprobeShotgunAndCache(pathFilename).get('rolloff')

@registrationAudioAspect('Abs_Peak_count')
def analyzeAbs_Peak_count(pathFilename: str | PathLike[Any]) -> float | None:
	print('Abs_Peak_count', pathFilename)
	return ffprobeShotgunAndCache(pathFilename).get('Abs_Peak_count')

@registrationAudioAspect('Bit_depth')
def analyzeBit_depth(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Bit_depth')

@registrationAudioAspect('Flat_factor')
def analyzeFlat_factor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Flat_factor')

@registrationAudioAspect('Max_difference')
def analyzeMax_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Max_difference')

@registrationAudioAspect('Max_level')
def analyzeMax_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Max_level')

@registrationAudioAspect('Mean_difference')
def analyzeMean_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Mean_difference')

@registrationAudioAspect('Min_difference')
def analyzeMin_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Min_difference')

@registrationAudioAspect('Min_level')
def analyzeMin_level(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Min_level')

@registrationAudioAspect('Noise_floor')
def analyzeNoise_floor(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Noise_floor')

@registrationAudioAspect('Noise_floor_count')
def analyzeNoise_floor_count(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Noise_floor_count')

@registrationAudioAspect('Peak_count')
def analyzePeak_count(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Peak_count')

@registrationAudioAspect('RMS_difference')
def analyzeRMS_difference(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_difference')

@registrationAudioAspect('RMS_trough')
def analyzeRMS_trough(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('RMS_trough')
