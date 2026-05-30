# ruff: noqa: D103
"""Analyzers that use the spectrogram to analyze audio data."""
from __future__ import annotations

from analyzeAudio import registrationAudioAspect
from numpy import dtype, floating
from typing import Any
import librosa
import numpy

def analyzeChromagram(spectrogramPower: numpy.ndarray[Any, dtype[floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.chroma_stft(S=spectrogramPower, sr=sampleRate, **keywordArguments)

@registrationAudioAspect('Chromagram mean')
def analyzeChromagramMean(spectrogramPower: numpy.ndarray[Any, dtype[floating[Any]]], sampleRate: int, **keywordArguments: Any) -> float:
	return float(analyzeChromagram(spectrogramPower, sampleRate, **keywordArguments).mean().item())

def analyzeSpectralContrast(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.spectral_contrast(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Contrast mean')
def analyzeSpectralContrastMean(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeSpectralContrast(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralBandwidth(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	centroid = analyzeSpectralCentroid(spectrogramMagnitude, **keywordArguments)
	return librosa.feature.spectral_bandwidth(S=spectrogramMagnitude, centroid=centroid, **keywordArguments)

@registrationAudioAspect('Spectral Bandwidth mean')
def analyzeSpectralBandwidthMean(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeSpectralBandwidth(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralCentroid(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.spectral_centroid(S=spectrogramMagnitude, **keywordArguments)

@registrationAudioAspect('Spectral Centroid mean')
def analyzeSpectralCentroidMean(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeSpectralCentroid(spectrogramMagnitude, **keywordArguments).mean().item())

def analyzeSpectralFlatness(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	spectralFlatness: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]] = librosa.feature.spectral_flatness(S=spectrogramMagnitude, **keywordArguments)
	return 20 * numpy.log10(spectralFlatness, where=(spectralFlatness != 0), out=None)  # dB

@registrationAudioAspect('Spectral Flatness mean')
def analyzeSpectralFlatnessMean(spectrogramMagnitude: numpy.ndarray[Any, dtype[floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeSpectralFlatness(spectrogramMagnitude, **keywordArguments).mean().item())
