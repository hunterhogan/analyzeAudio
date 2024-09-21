from analyzeAudio import audioAspects, registrationAudioAspect
from functools import cache
from numpy.typing import NDArray
from typing import Any
import librosa
import numpy

@cache
@registrationAudioAspect('Tempogram')
def analyzeTempogram(waveform: NDArray, sampleRate: int, **kwargs: Any) -> NDArray:
    return librosa.feature.tempogram(y=waveform, sr=sampleRate, **kwargs)

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
@registrationAudioAspect('RMS')
def analyzeRMS(waveform: NDArray, **kwargs: Any) -> NDArray:
    arrayRMS = librosa.feature.rms(y=waveform, **kwargs)
    return 20 * numpy.log10(arrayRMS, where=arrayRMS != 0) # dB

@registrationAudioAspect('Tempo')
def analyzeTempo(waveform: NDArray, sampleRate: int, **kwargs: Any) -> NDArray:
    tempogram = audioAspects['Tempogram']['analyzer'](waveform, sampleRate)
    return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **kwargs)

@registrationAudioAspect('Zero-crossing rate') # This is distinct from 'Zero-crossings rate'
def analyzeZeroCrossingRate(waveform: NDArray, **kwargs: Any) -> NDArray:
    return librosa.feature.zero_crossing_rate(y=waveform, **kwargs)
