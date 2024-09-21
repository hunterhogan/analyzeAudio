from .audioAspectsRegistry import analyzeAudio, registrationAudioAspect, audioAspects, analyzeListPathFilenamesAudio
import configparser
from pathlib import Path

parse_setupDOTcfg = configparser.ConfigParser()

parse_setupDOTcfg.read(Path(__file__).resolve().parent.parent / 'setup.cfg')

__version__ = parse_setupDOTcfg.get('metadata', 'version', fallback='0.0.0')
__author__ = parse_setupDOTcfg.get('metadata', 'author', fallback='Unknown')

__all__ = ['analyzeAudio', '__version__', '__author__', 'registrationAudioAspect', 'audioAspects', 'analyzeListPathFilenamesAudio']


from . import audioAspectsTensor
from . import audioAspectsSpectrogram
from . import audioAspectsFilename
from . import audioAspectsWaveform