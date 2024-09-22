from analyzeAudio import audioAspects, registrationAudioAspect
from functools import cache
from typing import Any
import librosa
import numpy

@cache
@registrationAudioAspect('Tempogram')
def analyzeTempogram(waveform: numpy.ndarray, sampleRate: int, **kwargs: Any) -> numpy.ndarray:
    return librosa.feature.tempogram(y=waveform, sr=sampleRate, **kwargs)

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
@registrationAudioAspect('RMS')
def analyzeRMS(waveform: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    arrayRMS = librosa.feature.rms(y=waveform, **kwargs)
    return 20 * numpy.log10(arrayRMS, where=arrayRMS != 0) # dB

@registrationAudioAspect('Tempo')
def analyzeTempo(waveform: numpy.ndarray, sampleRate: int, **kwargs: Any) -> numpy.ndarray:
    tempogram = audioAspects['Tempogram']['analyzer'](waveform, sampleRate)
    return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **kwargs)

@registrationAudioAspect('Zero-crossing rate') # This is distinct from 'Zero-crossings rate'
def analyzeZeroCrossingRate(waveform: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    return librosa.feature.zero_crossing_rate(y=waveform, **kwargs)
