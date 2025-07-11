"""Analyzers that use the waveform of audio data."""
# ruff: noqa: D103
from analyzeAudio import audioAspects, cacheAudioAnalyzers, registrationAudioAspect
from typing import Any
import cachetools
import librosa
import numpy

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Tempogram')
def analyzeTempogram(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
@registrationAudioAspect('RMS from waveform')
def analyzeRMS(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	arrayRMS: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]] = librosa.feature.rms(y=waveform, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
	return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0)) # dB

@registrationAudioAspect('Tempo')
def analyzeTempo(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	tempogram = audioAspects['Tempogram']['analyzer'](waveform, sampleRate)
	return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]

@registrationAudioAspect('Zero-crossing rate') # This is distinct from 'Zero-crossings rate'
def analyzeZeroCrossingRate(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> numpy.ndarray: # pyright: ignore [reportMissingTypeArgument, reportUnknownParameterType]
	return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments) # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
