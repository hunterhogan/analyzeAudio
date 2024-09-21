from analyzeAudio import registrationAudioAspect
from cachetools import cached
from .pythonator import pythonizeFFprobe
from typing import Dict, List
import numpy
import subprocess
from os import PathLike
import re
import pathlib

@cached(cache={})
def ffprobeShotgunAndCache(pathFilename: PathLike) -> Dict[str, float]:
    # the colons after driveLetter letters can be tricky because they need to be escaped twice.
    
    """
    https://docs.python.org/3/library/pathlib.html
Pure paths are useful in some special cases; for example:
... the pure classes simply don't have any OS-accessing operations.
    pathlib.PurePosixPath(pathFilename)<--this
    that will help ensure that FFprobe receives the pathFilename in a POSIX-compliant format.
    furthermore, it is "semantically" accurate because Python doesn't use the path: it's just a datapoint
    """
    
    pathFilename = pathlib.Path(pathFilename).as_posix()
    escapedColonPathFilename = re.sub(r'(?<!\\):', r'\\:', pathFilename)

    filterchain: List[str] = []
    filterchain += ["astats=metadata=1:measure_perchannel=Crest_factor+Zero_crossings_rate+Dynamic_range:measure_overall=all"]
    filterchain += ["aspectralstats"]
    filterchain += ["ebur128=metadata=1:framelog=quiet"]

    entriesFFprobe = ["frame_tags"]

    commandLineFFprobe = [
        "ffprobe", "-hide_banner",
        "-f", "lavfi", f"amovie={escapedColonPathFilename},{','.join(filterchain)}",
        "-show_entries", ':'.join(entriesFFprobe),
        "-output_format", "json=compact=1",
    ]

    systemprocessFFprobe = subprocess.Popen(commandLineFFprobe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutFFprobe, DISCARDstderr = systemprocessFFprobe.communicate()
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
def analyzeI(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('I')

@registrationAudioAspect('LUFS loudness range')
def analyzeI(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA')

@registrationAudioAspect('LUFS low')
def analyzeI(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.low')

@registrationAudioAspect('LUFS high')
def analyzeI(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('LRA.high')

@registrationAudioAspect('Spectral mean')
def analyzemean(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('mean')

@registrationAudioAspect('Spectral variance')
def analyzevariance(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('variance')

@registrationAudioAspect('Spectral centroid')
def analyzecentroid(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('centroid')

@registrationAudioAspect('Spectral spread')
def analyzespread(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('spread')

@registrationAudioAspect('Spectral skewness')
def analyzeskewness(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('skewness')

@registrationAudioAspect('Spectral kurtosis')
def analyzekurtosis(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('kurtosis')

@registrationAudioAspect('Spectral entropy')
def analyzeentropy(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('entropy')

@registrationAudioAspect('Spectral flatness')
def analyzeflatness(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('flatness')

@registrationAudioAspect('Spectral crest')
def analyzecrest(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('crest')

@registrationAudioAspect('Spectral flux')
def analyzeflux(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('flux')

@registrationAudioAspect('Spectral slope')
def analyzeslope(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('slope')

@registrationAudioAspect('Spectral decrease')
def analyzedecrease(pathFilename: PathLike) -> float:
    return ffprobeShotgunAndCache(pathFilename).get('decrease')

@registrationAudioAspect('Spectral rolloff')
def analyzerolloff(pathFilename: PathLike) -> float:
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
