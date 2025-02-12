from .audioAspectsRegistry import registrationAudioAspect, cacheAudioAnalyzers, analyzeAudioFile, \
	analyzeAudioListPathFilenames, getListAvailableAudioAspects, audioAspects

__all__ = [
	'analyzeAudioFile',
	'analyzeAudioListPathFilenames',
	'audioAspects',
	'getListAvailableAudioAspects',
]

from . import analyzersUseFilename
from . import analyzersUseSpectrogram
from . import analyzersUseTensor
from . import analyzersUseWaveform
