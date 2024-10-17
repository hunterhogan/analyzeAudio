from .pythonator import pythonizeFFprobe
from analyzeAudio import registrationAudioAspect
from cachetools import cached
from os import PathLike
from pathlib import PureWindowsPath, Path
from statistics import mean
from typing import Dict, List
import numpy
import re as regex
import subprocess

@registrationAudioAspect('SI-SDR mean')
def getSI_SDRmean(pathFilenameAlpha: PathLike, pathFilenameBeta: PathLike) -> float:
    """
    Calculate the mean Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) between two audio files.
    This function uses FFmpeg to compute the SI-SDR between two audio files specified by their paths.
    The SI-SDR values are extracted from the FFmpeg output and their mean is calculated.
    Args:
        pathFilenameAlpha (PathLike): Path to the first audio file.
        pathFilenameBeta (PathLike): Path to the second audio file.
    Returns:
        float: The mean SI-SDR value in decibels (dB).
    Raises:
        subprocess.CalledProcessError: If the FFmpeg command fails.
        ValueError: If no SI-SDR values are found in the FFmpeg output.
    """
    commandLineFFmpeg = ['ffmpeg', '-hide_banner', '-loglevel', '32', 
                         '-i', f'{str(Path(pathFilenameAlpha))}', '-i', f'{str(Path(pathFilenameBeta))}', 
                         '-filter_complex', '[0][1]asisdr', '-f', 'null', '-']
    systemProcessFFmpeg = subprocess.run(commandLineFFmpeg, check=True, stderr=subprocess.PIPE)

    stderrFFmpeg = systemProcessFFmpeg.stderr.decode()

    regexSI_SDR = regex.compile(r"^\[Parsed_asisdr_.* (.*) dB", regex.MULTILINE)

    listMatchesSI_SDR = regexSI_SDR.findall(stderrFFmpeg)
    SI_SDRmean = mean([float(match) for match in listMatchesSI_SDR])
    return SI_SDRmean

@cached(cache={})
def ffprobeShotgunAndCache(pathFilename: PathLike) -> Dict[str, float]:
    
    # for lavfi amovie/movie, the colons after driveLetter letters need to be escaped twice.
    pFn = Path(pathFilename)
    lavfiPathFilename = pFn.drive.replace(":", "\\\\:")+pFn.with_segments(pFn.root,pFn.relative_to(pFn.anchor)).as_posix()
    
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
        for keyName, arrayFeatureValues in FFprobeStructured['astats'].items():
            dictionaryAspectsAnalyzed[keyName.split('.')[-1]] = numpy.mean(arrayFeatureValues[..., -1:])

    return dictionaryAspectsAnalyzed

@registrationAudioAspect('Zero-crossings rate')
def analyzeZero_crossings_rate(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Zero_crossings_rate')

@registrationAudioAspect('DC offset')
def analyzeDCoffset(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('DC_offset')

@registrationAudioAspect('Dynamic range')
def analyzeDynamicRange(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Dynamic_range')

@registrationAudioAspect('Signal entropy')
def analyzeEntropy(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Entropy')

@registrationAudioAspect('Duration-samples')
def analyzeNumber_of_samples(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Number_of_samples')

@registrationAudioAspect('Peak dB')
def analyzePeak_level(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Peak_level')

@registrationAudioAspect('RMS total')
def analyzeRMS_level(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_level')

@registrationAudioAspect('Crest factor')
def analyzeCrest_factor(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Crest_factor')

@registrationAudioAspect('RMS peak')
def analyzeRMS_peak(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_peak')

@registrationAudioAspect('LUFS integrated')
def analyzeLUFSintegrated(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('I')

@registrationAudioAspect('LUFS loudness range')
def analyzeLRA(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA')

@registrationAudioAspect('LUFS low')
def analyzeLUFSlow(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.low')

@registrationAudioAspect('LUFS high')
def analyzeLUFShigh(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.high')

@registrationAudioAspect('Spectral mean')
def analyzeMean(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('mean')

@registrationAudioAspect('Spectral variance')
def analyzeVariance(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('variance')

@registrationAudioAspect('Spectral centroid')
def analyzeCentroid(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('centroid')

@registrationAudioAspect('Spectral spread')
def analyzeSpread(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('spread')

@registrationAudioAspect('Spectral skewness')
def analyzeSkewness(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('skewness')

@registrationAudioAspect('Spectral kurtosis')
def analyzeKurtosis(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

@registrationAudioAspect('Spectral entropy')
def analyzeEntropy(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('entropy')

@registrationAudioAspect('Spectral flatness')
def analyzeFlatness(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('flatness')

@registrationAudioAspect('Spectral crest')
def analyzeCrest(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('crest')

@registrationAudioAspect('Spectral flux')
def analyzeFlux(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('flux')

@registrationAudioAspect('Spectral slope')
def analyzeSlope(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('slope')

@registrationAudioAspect('Spectral decrease')
def analyzeDecrease(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('decrease')

@registrationAudioAspect('Spectral rolloff')
def analyzeRolloff(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('rolloff')

@registrationAudioAspect('Abs_Peak_count')
def analyzeAbs_Peak_count(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Abs_Peak_count')

@registrationAudioAspect('Bit_depth')
def analyzeBit_depth(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Bit_depth')

@registrationAudioAspect('Flat_factor')
def analyzeFlat_factor(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Flat_factor')

@registrationAudioAspect('Max_difference')
def analyzeMax_difference(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Max_difference')

@registrationAudioAspect('Max_level')
def analyzeMax_level(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Max_level')

@registrationAudioAspect('Mean_difference')
def analyzeMean_difference(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Mean_difference')

@registrationAudioAspect('Min_difference')
def analyzeMin_difference(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Min_difference')

@registrationAudioAspect('Min_level')
def analyzeMin_level(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Min_level')

@registrationAudioAspect('Noise_floor')
def analyzeNoise_floor(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Noise_floor')

@registrationAudioAspect('Noise_floor_count')
def analyzeNoise_floor_count(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Noise_floor_count')

@registrationAudioAspect('Peak_count')
def analyzePeak_count(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('Peak_count')

@registrationAudioAspect('RMS_difference')
def analyzeRMS_difference(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_difference')

@registrationAudioAspect('RMS_trough')
def analyzeRMS_trough(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('RMS_trough')
