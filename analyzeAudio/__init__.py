from .audioAspectsRegistry import registrationAudioAspect, cacheAudioAnalyzers, analyzeAudioFile, \
    analyzeAudioListPathFilenames, getListAvailableAudioAspects, audioAspects
from pathlib import Path
import configparser
import toml

parse_setupDOTcfg = configparser.ConfigParser()

parse_setupDOTcfg.read(Path(__file__).resolve().parent.parent / 'setup.cfg')

parsePyproject = toml.load(Path(__file__).resolve().parent.parent / 'pyproject.toml')
dictionaryProjectMetadata = parsePyproject.get('project', {})
__version__ = dictionaryProjectMetadata.get('version', '0.0.0')
__author__ = dictionaryProjectMetadata.get('authors', [{'name': 'Unknown'}])[0]['name']

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
