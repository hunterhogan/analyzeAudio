from .audioAspectsRegistry import registrationAudioAspect, cacheAudioAnalyzers, analyzeAudioFile, \
    analyzeAudioListPathFilenames, getListAvailableAudioAspects, audioAspects

__version__ = "0.0.11"
__author__ = "Hunter Hogan"

__all__ = [
    '__author__',
    '__version__',
    'analyzeAudioFile',
    'analyzeAudioListPathFilenames',
    'audioAspects',
    'getListAvailableAudioAspects',
]

from . import analyzersUseFilename
from . import analyzersUseSpectrogram
from . import analyzersUseTensor
from . import analyzersUseWaveform
