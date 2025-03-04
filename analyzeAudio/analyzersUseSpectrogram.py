from analyzeAudio import registrationAudioAspect, audioAspects, cacheAudioAnalyzers
from typing import Any
import cachetools
import librosa
import numpy
from optype.numpy import AnyFloatingDType, ToArray3D, ToFloat3D
from numpy import dtype, floating

@registrationAudioAspect('Chromagram')
def analyzeChromagram(spectrogramPower: numpy.ndarray[Any, dtype[floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **keywordArguments)

@registrationAudioAspect('Spectral Contrast')
def analyzeSpectralContrast(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Bandwidth')
def analyzeSpectralBandwidth(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	centroid = audioAspects['Spectral Centroid']['analyzer'](spectrogramMagnitude)
	return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **keywordArguments)

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Spectral Centroid')
def analyzeSpectralCentroid(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Flatness')
def analyzeSpectralFlatness(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	spectralFlatness = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments)
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0)) # dB
