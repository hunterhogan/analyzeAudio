from analyzeAudio import registrationAudioAspect, audioAspects
from functools import cache
from typing import Any
import librosa
import numpy

@registrationAudioAspect('Chromagram')
def analyzeChromagram(spectrogramPower: numpy.ndarray, sampleRate: int, **kwargs: Any) -> numpy.ndarray:
    return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **kwargs)

@registrationAudioAspect('Spectral Contrast')
def analyzeSpectralContrast(spectrogramMagnitude: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **kwargs)

@registrationAudioAspect('Spectral Bandwidth')
def analyzeSpectralBandwidth(spectrogramMagnitude: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    centroid = audioAspects['Spectral Centroid']['analyzer'](spectrogramMagnitude)
    return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **kwargs)

@cache
@registrationAudioAspect('Spectral Centroid')
def analyzeSpectralCentroid(spectrogramMagnitude: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **kwargs)

@registrationAudioAspect('Spectral Flatness')
def analyzeSpectralFlatness(spectrogramMagnitude: numpy.ndarray, **kwargs: Any) -> numpy.ndarray:
    spectralFlatness = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **kwargs)
    return 20 * numpy.log10(spectralFlatness, where=spectralFlatness != 0) # dB
