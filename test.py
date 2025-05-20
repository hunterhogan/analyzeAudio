import pathlib
from analyzeAudio import analyzeAudioListPathFilenames
from Z0Z_tools import dataTabularTOpathFilenameDelimited

if __name__ == "__main__":

	lPFn=(list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").rglob("test*.wav")))
	aa=['Abs_Peak_count', 'Bit_depth', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak dB', 'Peak_count', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid', 'Spectral crest', 'Spectral decrease', 'Spectral entropy', 'Spectral flatness', 'Spectral flux', 'Spectral kurtosis', 'Spectral mean', 'Spectral rolloff', 'Spectral skewness', 'Spectral slope', 'Spectral spread', 'Spectral variance', 'Tempo mean', 'Tempogram mean', 'Zero-crossing rate mean', ]
	singleTargetFloats=[ 'Peak dB', 'RMS from waveform mean', 'RMS total', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid', 'Spectral crest', 'Spectral decrease', 'Spectral entropy', 'Spectral flatness', 'Spectral flux', 'Spectral kurtosis', 'Spectral mean', 'Spectral rolloff', 'Spectral skewness', 'Spectral slope', 'Spectral spread', 'Spectral variance', ]
	rows=analyzeAudioListPathFilenames(lPFn,singleTargetFloats)

	dataTabularTOpathFilenameDelimited(lPFn[0].parent/'l073.tab', rows, ['pathFilename']+singleTargetFloats)
	singleTestsA=['SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Contrast mean', 'Spectral flatness', 'Spectral rolloff',]
