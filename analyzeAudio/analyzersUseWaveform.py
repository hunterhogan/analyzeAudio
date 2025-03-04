from analyzeAudio import registrationAudioAspect, audioAspects, cacheAudioAnalyzers
from typing import Any
import librosa
import numpy
from optype.numpy import ToArray2D, AnyFloatingDType
import cachetools

@cachetools.cached(cache=cacheAudioAnalyzers)
@registrationAudioAspect('Tempogram')
def analyzeTempogram(waveform: ToArray2D[AnyFloatingDType], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.tempogram(y=waveform, sr=sampleRate, **keywordArguments)

# "RMS value from audio samples is faster ... However, ... spectrogram ... more accurate ... because ... windowed"
@registrationAudioAspect('RMS from waveform')
def analyzeRMS(waveform: ToArray2D[AnyFloatingDType], **keywordArguments: Any) -> numpy.ndarray:
	arrayRMS = librosa.feature.rms(y=waveform, **keywordArguments)
	return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0)) # dB

@registrationAudioAspect('Tempo')
def analyzeTempo(waveform: ToArray2D[AnyFloatingDType], sampleRate: int, **keywordArguments: Any) -> numpy.ndarray:
	tempogram = audioAspects['Tempogram']['analyzer'](waveform, sampleRate)
	return librosa.feature.tempo(y=waveform, sr=sampleRate, tg=tempogram, **keywordArguments)

@registrationAudioAspect('Zero-crossing rate') # This is distinct from 'Zero-crossings rate'
def analyzeZeroCrossingRate(waveform: ToArray2D[AnyFloatingDType], **keywordArguments: Any) -> numpy.ndarray:
	return librosa.feature.zero_crossing_rate(y=waveform, **keywordArguments)
