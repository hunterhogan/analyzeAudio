"""Analyzers that use the filename of an audio file to analyze its audio data."""
# ruff: noqa: D103
from __future__ import annotations

from analyzeAudio import registrationAudioAspect
from analyzeAudio.pythonator import pythonizeFFprobe
from statistics import mean
from typing import Any, cast, TYPE_CHECKING
import numpy
import pathlib
import re as regex
import subprocess  # noqa: S404

if TYPE_CHECKING:
	from os import PathLike

def _meanDB(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any], filterChain: str) -> float | None:
	regexPattern = regex.compile(rf"^\[Parsed_{filterChain}_.* (.*) dB", regex.MULTILINE)
	commandLineFFmpeg = [
		'ffmpeg', '-hide_banner', '-loglevel', '32',
		'-i', f'{str(pathlib.Path(pathFilenameAlfa))}', '-i', f'{str(pathlib.Path(pathFilenameBeta))}',
		'-filter_complex', f'[0][1]{filterChain}', '-f', 'null', '-'
	]
	systemProcessFFmpeg = subprocess.run(commandLineFFmpeg, check=True, stderr=subprocess.PIPE)

	stderrFFmpeg: str = systemProcessFFmpeg.stderr.decode()

	return mean(map(float, regexPattern.findall(stderrFFmpeg)))

@registrationAudioAspect('Peak Signal-to-Noise Ratio mean')
def getPSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	filterChain: str = 'apsnr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

@registrationAudioAspect('SDR mean')
def getSDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	filterChain: str = 'asdr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

@registrationAudioAspect('SI-SDR mean')
def getSI_SDRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
	"""Calculate the mean Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between two audio files.

	Parameters
	----------
	pathFilenameAlfa : str | PathLike[Any]
		Path to the first audio file.
	pathFilenameBeta : str | PathLike[Any]
		Path to the second audio file.

	Returns
	-------
	SI_SDRmean : float | None
		The mean SI-SDR value in decibels (dB).
	"""
	filterChain: str = 'asisdr'
	return _meanDB(pathFilenameAlfa, pathFilenameBeta, filterChain)

def ffprobeShotgunAndCache(pathFilename: str | PathLike[Any]) -> dict[str, float]:
	# for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
	pFn = pathlib.PureWindowsPath(pathFilename)
	lavfiPathFilename = pFn.drive.replace(":", "\\\\:") + pathlib.PureWindowsPath(pFn.root, pFn.relative_to(pFn.anchor)).as_posix()

	filterChain: list[str] = []
	filterChain += ["astats=metadata=1:measure_perchannel=Crest_factor+Zero_crossings_rate+Zero_crossings+Dynamic_range:measure_overall=all"]
	filterChain += ["aspectralstats"]
	filterChain += ["ebur128=metadata=1:framelog=quiet"]

	entriesFFprobe: list[str] = ["frame_tags"]

	commandLineFFprobe: list[str] = [
		"ffprobe", "-hide_banner",
		"-f", "lavfi", f"amovie={lavfiPathFilename},{','.join(filterChain)}",
		"-show_entries", ':'.join(entriesFFprobe),
		"-output_format", "json=compact=1",
	]

	systemProcessFFprobe = subprocess.Popen(commandLineFFprobe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	stdoutFFprobe, _DISCARDstderr = systemProcessFFprobe.communicate()
	FFprobeStructured = pythonizeFFprobe(stdoutFFprobe.decode('utf-8'))[-1]

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

@registrationAudioAspect('Zero crossings')
def analyzeZero_crossings(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings')

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

@registrationAudioAspect('Power spectral density mean')
def analyzeMean(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('mean')

@registrationAudioAspect('Spectral variance mean')
def analyzeVariance(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('variance')

@registrationAudioAspect('Spectral centroid mean')
def analyzeCentroid(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('centroid')

@registrationAudioAspect('Spectral spread mean')
def analyzeSpread(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('spread')

@registrationAudioAspect('Spectral skewness mean')
def analyzeSkewness(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('skewness')

@registrationAudioAspect('Spectral kurtosis mean')
def analyzeKurtosis(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

@registrationAudioAspect('Spectral entropy mean')
def analyzeSpectralEntropy(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('entropy')

@registrationAudioAspect('Spectral flatness mean')
def analyzeFlatness(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('flatness')

@registrationAudioAspect('Spectral crest mean')
def analyzeCrest(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('crest')

@registrationAudioAspect('Spectral flux mean')
def analyzeFlux(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('flux')

@registrationAudioAspect('Spectral slope mean')
def analyzeSlope(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('slope')

@registrationAudioAspect('Spectral decrease mean')
def analyzeDecrease(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('decrease')

@registrationAudioAspect('Spectral rolloff mean')
def analyzeRolloff(pathFilename: str | PathLike[Any]) -> float | None:
	return ffprobeShotgunAndCache(pathFilename).get('rolloff')

@registrationAudioAspect('Abs_Peak_count')
def analyzeAbs_Peak_count(pathFilename: str | PathLike[Any]) -> float | None:
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
