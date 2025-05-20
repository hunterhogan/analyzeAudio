"""Analyzers that use the spectrogram to analyze audio data."""
from analyzeAudio import registrationAudioAspect, audioAspects, cacheAudioAnalyzers
from typing import Any
import cachetools
import librosa
import numpy
from numpy import dtype, floating

@registrationAudioAspect('Chromagram')
def analyzeChromagram(spectrogramPower: numpy.ndarray[Any, dtype[floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **keywordArguments) # type: ignore

@registrationAudioAspect('Spectral Contrast')
def analyzeSpectralContrast(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **keywordArguments) # type: ignore

@registrationAudioAspect('Spectral Bandwidth')
def analyzeSpectralBandwidth(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	centroid = audioAspects['Spectral Centroid']['analyzer'](spectrogramMagnitude)
	return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **keywordArguments) # type: ignore

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Spectral Centroid')
def analyzeSpectralCentroid(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **keywordArguments) # type: ignore

@registrationAudioAspect('Spectral Flatness')
def analyzeSpectralFlatness(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	spectralFlatness: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]] = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments) # type: ignore
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0)) # dB
