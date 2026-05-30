# ruff: noqa: ERA001
from __future__ import annotations

from analyzeAudio import dataTabularTOpathFilenameDelimited
from analyzeAudio.analyze import analyzeAudioListPathFilenames
import pathlib

if __name__ == "__main__":

	lPFn: list[pathlib.Path] = (list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").rglob("test*.wav")))
	aa: list[str] = ['Abs_Peak_count', 'Bit_depth', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'L1SNR mean', 'L1SNRDB mean', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'LogWMSE mean', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'MultiL1SNRDB mean', 'Noise_floor', 'Noise_floor_count', 'Peak Signal-to-Noise Ratio mean', 'Peak dB', 'Peak_count', 'Power spectral density mean', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SDR mean', 'SI-SDR mean', 'SRMR mean', 'STFTL1SNRDB mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid mean', 'Spectral crest mean', 'Spectral decrease mean', 'Spectral entropy mean', 'Spectral flatness mean', 'Spectral flux mean', 'Spectral kurtosis mean', 'Spectral rolloff mean', 'Spectral skewness mean', 'Spectral slope mean', 'Spectral spread mean', 'Spectral variance mean', 'Tempo mean', 'Tempogram mean', 'Zero crossings', 'Zero-crossing rate mean', 'Zero-crossings rate']
	singleTargetFloats: list[str] = ['Abs_Peak_count', 'Bit_depth', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak dB', 'Peak_count', 'Power spectral density mean', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid mean', 'Spectral crest mean', 'Spectral decrease mean', 'Spectral entropy mean', 'Spectral flatness mean', 'Spectral flux mean', 'Spectral kurtosis mean', 'Spectral rolloff mean', 'Spectral skewness mean', 'Spectral slope mean', 'Spectral spread mean', 'Spectral variance mean', 'Tempo mean', 'Tempogram mean', 'Zero crossings', 'Zero-crossing rate mean', 'Zero-crossings rate']
	rows: list[list[str | float]] = analyzeAudioListPathFilenames(lPFn, singleTargetFloats)

	dataTabularTOpathFilenameDelimited(lPFn[0].parent.parent.parent / 'l073.tab', rows, ['pathFilename', *singleTargetFloats])
	singleTargetFloats: list[str] = ['Crest factor', 'Spectral kurtosis', 'Signal entropy', 'Spectral flatness']

	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[0], lPFn[1])
	# print(PSNR_channelsMean)
	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[1], lPFn[0])
	# print(PSNR_channelsMean)
