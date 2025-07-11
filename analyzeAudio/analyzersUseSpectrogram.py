"""Analyzers that use the spectrogram to analyze audio data."""
# ruff: noqa: D103
from analyzeAudio import audioAspects, cacheAudioAnalyzers, registrationAudioAspect
from numpy import dtype, floating
from typing import Any
import cachetools
import librosa
import numpy

@registrationAudioAspect('Chromagram')
def analyzeChromagram(spectrogramPower: numpy.ndarray[Any, dtype[floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

@registrationAudioAspect('Spectral Contrast')
def analyzeSpectralContrast(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

@registrationAudioAspect('Spectral Bandwidth')
def analyzeSpectralBandwidth(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	centroid = audioAspects['Spectral Centroid']['analyzer'](spectrogramMagnitude)
	return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Spectral Centroid')
def analyzeSpectralCentroid(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

@registrationAudioAspect('Spectral Flatness')
def analyzeSpectralFlatness(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	spectralFlatness: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]] = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments) # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0)) # dB
