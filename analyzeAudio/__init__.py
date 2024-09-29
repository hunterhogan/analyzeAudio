from .audioAspectsRegistry import registrationAudioAspect, analyzeAudioFile, analyzeAudioListPathFilenames, getListAvailableAudioAspects
import configparser
from pathlib import Path

parse_setupDOTcfg = configparser.ConfigParser()

parse_setupDOTcfg.read(Path(__file__).resolve().parent.parent / 'setup.cfg')

__version__ = parse_setupDOTcfg.get('metadata', 'version', fallback='0.0.0')
__author__ = parse_setupDOTcfg.get('metadata', 'author', fallback='Unknown')

__all__ = [
    '__author__', 
    '__version__',
    analyzeAudioFile, 
    analyzeAudioListPathFilenames, 
    getListAvailableAudioAspects,
    ]

from . import analyzersUseFilename
from . import analyzersUseSpectrogram
from . import analyzersUseTensor
from . import analyzersUseWaveform