# ruff: noqa: ERA001
from __future__ import annotations

from analyzeAudio import dataTabularTOpathFilenameDelimited, getListAvailableAudioAspects
from analyzeAudio.analyze import analyzeAudioListPathFilenames
import pathlib

if __name__ == "__main__":

	lPFn: list[pathlib.Path] = (list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").rglob("ch2_44100_83s_LUFS23_VoiceAndMusic*.wav")))
	aa: list[str] = ['LUFS short-term maximum', 'true_peak maximum', 'LUFS momentary maximum', 'Abs_Peak_count', 'Bit_depth', 'ChromaSTFTLoss', 'Chromagram mean', 'Crest factor', 'DC offset', 'DCLoss', 'Duration-samples', 'Dynamic range', 'ESRLoss', 'Flat_factor', 'L1FrequencyLoss', 'L1SNR', 'L1SNRDB', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'LogCoshLoss', 'LogWMSE', 'Max_difference', 'Max_level', 'Mean_difference', 'MelSTFTLoss', 'Min_difference', 'Min_level', 'MultiL1SNRDB', 'MultiResolutionSTFTLoss', 'Noise_floor', 'Noise_floor_count', 'Peak Signal-to-Noise Ratio mean', 'Peak dB', 'Peak_count', 'Power spectral density mean', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'RandomResolutionSTFTLoss', 'SDR mean', 'SDSDRLoss', 'SI-SDR mean', 'SISDRLoss', 'SNRLoss', 'SRMR mean', 'STFTL1SNRDB', 'STFTLoss', 'STFTMagnitudeLoss', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid mean', 'Spectral crest mean', 'Spectral decrease mean', 'Spectral entropy mean', 'Spectral flatness mean', 'Spectral flux mean', 'Spectral kurtosis mean', 'Spectral rolloff mean', 'Spectral skewness mean', 'Spectral slope mean', 'Spectral spread mean', 'Spectral variance mean', 'SpectralConvergenceLoss', 'SumAndDifferenceSTFTLoss', 'Tempo mean', 'Tempogram mean', 'Zero crossings', 'Zero-crossing rate mean', 'Zero-crossings rate']
	singleTargetFloats: list[str] = getListAvailableAudioAspects()
	rows: list[list[str | float]] = analyzeAudioListPathFilenames(lPFn, singleTargetFloats)

	dataTabularTOpathFilenameDelimited(lPFn[0].parent.parent.parent / 'l073.tab', rows, ['pathFilename', *singleTargetFloats])
	singleTargetFloats: list[str] = ['Crest factor', 'Spectral kurtosis mean', 'Signal entropy', 'Spectral flatness mean']

	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[0], lPFn[1])
	# print(PSNR_channelsMean)
	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[1], lPFn[0])
	# print(PSNR_channelsMean)
