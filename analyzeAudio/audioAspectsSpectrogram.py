from analyzeAudio import registrationAudioAspect, audioAspects
from functools import cache
from numpy.typing import NDArray
from typing import Any
import librosa
import numpy

@registrationAudioAspect('Chromagram')
def analyzeChromagram(spectrogramPower: NDArray, sampleRate: int, **kwargs: Any) -> NDArray:
    return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **kwargs)

@registrationAudioAspect('Spectral Contrast')
def analyzeSpectralContrast(spectrogramMagnitude: NDArray, **kwargs: Any) -> NDArray:
    return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **kwargs)

@registrationAudioAspect('Spectral Bandwidth')
def analyzeSpectralBandwidth(spectrogramMagnitude: NDArray, **kwargs: Any) -> NDArray:
    centroid = audioAspects['Spectral Centroid']['analyzer'](spectrogramMagnitude)
    return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **kwargs)

@cache
@registrationAudioAspect('Spectral Centroid')
def analyzeSpectralCentroid(spectrogramMagnitude: NDArray, **kwargs: Any) -> NDArray:
    return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **kwargs)

@registrationAudioAspect('Spectral Flatness')
def analyzeSpectralFlatness(spectrogramMagnitude: NDArray, **kwargs: Any) -> NDArray:
    spectralFlatness = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **kwargs)
    return 20 * numpy.log10(spectralFlatness, where=spectralFlatness != 0) # dB
