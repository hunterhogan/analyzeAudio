"""Analyzers that use the waveform of audio data."""
# ruff: noqa: D103
from __future__ import annotations

from analyzeAudio import registrationAudioAspect
from typing import Any
import librosa
import numpy

def analyzeTempogram(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)

@registrationAudioAspect('Tempogram mean')
def analyzeTempogramMean(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> float:
	return float(analyzeTempogram(waveform, sampleRate, **keywordArguments).mean().item())

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
def analyzeRMS(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	arrayRMS: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]] = librosa.feature.rms(y=waveform, **keywordArguments)
	return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0), out=None)  # dB

@registrationAudioAspect('RMS from waveform mean')
def analyzeRMSMean(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeRMS(waveform, **keywordArguments).mean().item())

def analyzeTempo(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	tempogram = analyzeTempogram(waveform, sampleRate)
	return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments)

@registrationAudioAspect('Tempo mean')
def analyzeTempoMean(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], sampleRate: int, **keywordArguments: Any) -> float:
	return float(analyzeTempo(waveform, sampleRate, **keywordArguments).mean().item())

def analyzeZeroCrossingRate(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments)

@registrationAudioAspect('Zero-crossing rate mean')  # This is distinct from 'Zero-crossings rate mean'
def analyzeZeroCrossingRateMean(waveform: numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]], **keywordArguments: Any) -> float:
	return float(analyzeZeroCrossingRate(waveform, **keywordArguments).mean().item())
