# pyright: reportUnusedImport=false
"""Docstring?! Ain't nobody got time for that!."""
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import (
	analyzeAudioFile as analyzeAudioFile, analyzeAudioListPathFilenames as analyzeAudioListPathFilenames, audioAspects as audioAspects,
	cacheAudioAnalyzers, getListAvailableAudioAspects as getListAvailableAudioAspects, registrationAudioAspect)

__all__ = [
	'analyzeAudioFile',
	'analyzeAudioListPathFilenames',
	'audioAspects',
	'getListAvailableAudioAspects',
]

# isort: split
from . import analyzersUseFilename, analyzersUseSpectrogram, analyzersUseTensor, analyzersUseWaveform
