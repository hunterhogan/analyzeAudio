from .pythonator import pythonizeFFprobe
from analyzeAudio import registrationAudioAspect, cacheAudioAnalyzers
from statistics import mean
from typing import Any, Dict, List, Optional, Union, cast
import cachetools
import numpy
import os
import pathlib
import re as regex
import subprocess

@registrationAudioAspect('SI-SDR mean')
def getSI_SDRmean(pathFilenameAlpha: Union[str, os.PathLike[Any]], pathFilenameBeta: Union[str, os.PathLike[Any]]) -> Optional[float]:
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
def ffprobeShotgunAndCache(pathFilename: Union[str, os.PathLike[Any]]) -> Dict[str, float]:
    # for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
    pFn = pathlib.PureWindowsPath(pathFilename)
    lavfiPathFilename = pFn.drive.replace(":", "\\\\:")+pathlib.PureWindowsPath(pFn.root,pFn.relative_to(pFn.anchor)).as_posix()

    filterChain: List[str] = []
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
    stdoutFFprobe, DISCARDstderr = systemProcessFFprobe.communicate()
    FFprobeStructured = pythonizeFFprobe(stdoutFFprobe.decode('utf-8'))[-1]

    dictionaryAspectsAnalyzed = {}
    if 'aspectralstats' in FFprobeStructured:
        for keyName in FFprobeStructured['aspectralstats']:
            dictionaryAspectsAnalyzed[keyName] = numpy.mean(FFprobeStructured['aspectralstats'][keyName])
    if 'r128' in FFprobeStructured:
        for keyName in FFprobeStructured['r128']:
            dictionaryAspectsAnalyzed[keyName] = FFprobeStructured['r128'][keyName][-1]
    if 'astats' in FFprobeStructured:
        for keyName, arrayFeatureValues in cast(dict, FFprobeStructured['astats']).items():
            dictionaryAspectsAnalyzed[keyName.split('.')[-1]] = numpy.mean(arrayFeatureValues[..., -1:])

    return dictionaryAspectsAnalyzed

@registrationAudioAspect('Zero-crossings rate')
def analyzeZero_crossings_rate(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings_rate')

@registrationAudioAspect('DC offset')
def analyzeDCoffset(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('DC_offset')

@registrationAudioAspect('Dynamic range')
def analyzeDynamicRange(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Dynamic_range')

@registrationAudioAspect('Signal entropy')
def analyzeSignalEntropy(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Entropy')

@registrationAudioAspect('Duration-samples')
def analyzeNumber_of_samples(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Number_of_samples')

@registrationAudioAspect('Peak dB')
def analyzePeak_level(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Peak_level')

@registrationAudioAspect('RMS total')
def analyzeRMS_level(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_level')

@registrationAudioAspect('Crest factor')
def analyzeCrest_factor(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Crest_factor')

@registrationAudioAspect('RMS peak')
def analyzeRMS_peak(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_peak')

@registrationAudioAspect('LUFS integrated')
def analyzeLUFSintegrated(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('I')

@registrationAudioAspect('LUFS loudness range')
def analyzeLRA(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('LRA')

@registrationAudioAspect('LUFS low')
def analyzeLUFSlow(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.low')

@registrationAudioAspect('LUFS high')
def analyzeLUFShigh(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.high')

@registrationAudioAspect('Spectral mean')
def analyzeMean(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('mean')

@registrationAudioAspect('Spectral variance')
def analyzeVariance(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('variance')

@registrationAudioAspect('Spectral centroid')
def analyzeCentroid(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('centroid')

@registrationAudioAspect('Spectral spread')
def analyzeSpread(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('spread')

@registrationAudioAspect('Spectral skewness')
def analyzeSkewness(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('skewness')

@registrationAudioAspect('Spectral kurtosis')
def analyzeKurtosis(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

@registrationAudioAspect('Spectral entropy')
def analyzeSpectralEntropy(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('entropy')

@registrationAudioAspect('Spectral flatness')
def analyzeFlatness(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('flatness')

@registrationAudioAspect('Spectral crest')
def analyzeCrest(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('crest')

@registrationAudioAspect('Spectral flux')
def analyzeFlux(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('flux')

@registrationAudioAspect('Spectral slope')
def analyzeSlope(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('slope')

@registrationAudioAspect('Spectral decrease')
def analyzeDecrease(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('decrease')

@registrationAudioAspect('Spectral rolloff')
def analyzeRolloff(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('rolloff')

@registrationAudioAspect('Abs_Peak_count')
def analyzeAbs_Peak_count(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    print('Abs_Peak_count', pathFilename)
    return ffprobeShotgunAndCache(pathFilename).get('Abs_Peak_count')

@registrationAudioAspect('Bit_depth')
def analyzeBit_depth(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Bit_depth')

@registrationAudioAspect('Flat_factor')
def analyzeFlat_factor(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Flat_factor')

@registrationAudioAspect('Max_difference')
def analyzeMax_difference(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Max_difference')

@registrationAudioAspect('Max_level')
def analyzeMax_level(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Max_level')

@registrationAudioAspect('Mean_difference')
def analyzeMean_difference(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Mean_difference')

@registrationAudioAspect('Min_difference')
def analyzeMin_difference(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Min_difference')

@registrationAudioAspect('Min_level')
def analyzeMin_level(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Min_level')

@registrationAudioAspect('Noise_floor')
def analyzeNoise_floor(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Noise_floor')

@registrationAudioAspect('Noise_floor_count')
def analyzeNoise_floor_count(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Noise_floor_count')

@registrationAudioAspect('Peak_count')
def analyzePeak_count(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('Peak_count')

@registrationAudioAspect('RMS_difference')
def analyzeRMS_difference(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_difference')

@registrationAudioAspect('RMS_trough')
def analyzeRMS_trough(pathFilename: Union[str, os.PathLike[Any]]) -> Optional[float]:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_trough')
