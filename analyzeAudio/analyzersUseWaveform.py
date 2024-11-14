from analyzeAudio import registrationAudioAspect, audioAspects, cacheAudioAnalyzers
from typing import Any
import librosa
import numpy
import cachetools

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Tempogram')
def analyzeTempogram(waveform: numpy.ndarray, sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
    return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
@registrationAudioAspect('RMS from waveform')
def analyzeRMS(waveform: numpy.ndarray, **keywordArguments: Any) -> numpy.ndarray:
    arrayRMS = librosa.feature.rms(y=waveform, **keywordArguments)
    return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0)) # dB

@registrationAudioAspect('Tempo')
def analyzeTempo(waveform: numpy.ndarray, sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
    tempogram = audioAspects['Tempogram']['analyzer'](waveform, sampleRate)
    return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments)

@registrationAudioAspect('Zero-crossing rate') # This is distinct from 'Zero-crossings rate'
def analyzeZeroCrossingRate(waveform: numpy.ndarray, **keywordArguments: Any) -> numpy.ndarray:
    return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments)
