"""Docstring?! Ain't nobody got time for that!."""
from .audioAspectsRegistry import (
	analyzeAudioFile, analyzeAudioListPathFilenames, audioAspects, cacheAudioAnalyzers, getListAvailableAudioAspects,
	registrationAudioAspect)

__all__ = [
	'analyzeAudioFile',
	'analyzeAudioListPathFilenames',
	'audioAspects',
	'getListAvailableAudioAspects',
]

# isort: split
from . import analyzersUseFilename, analyzersUseSpectrogram, analyzersUseTensor, analyzersUseWaveform
