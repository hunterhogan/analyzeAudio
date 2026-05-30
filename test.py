# ruff: noqa: ERA001
from __future__ import annotations

from analyzeAudio import audioAspects  # pyright: ignore[reportUnusedImport]
from analyzeAudio.analyze import analyzeAudioListPathFilenames
from typing import Any, TYPE_CHECKING
from Z0Z_tools import dataTabularTOpathFilenameDelimited
import pathlib

if TYPE_CHECKING:
	from numpy import dtype, ndarray

if __name__ == "__main__":

	lPFn: list[pathlib.Path] = (list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").rglob("test*.wav")))
	aa: list[str] = ['Abs_Peak_count', 'Bit_depth', 'Chromagram mean', 'Crest factor', 'DC offset', 'Duration-samples', 'Dynamic range', 'Flat_factor', 'LUFS high', 'LUFS integrated', 'LUFS loudness range', 'LUFS low', 'Max_difference', 'Max_level', 'Mean_difference', 'Min_difference', 'Min_level', 'Noise_floor', 'Noise_floor_count', 'Peak dB', 'Peak_count', 'RMS from waveform mean', 'RMS peak', 'RMS total', 'RMS_difference', 'RMS_trough', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid', 'Spectral crest', 'Spectral decrease', 'Spectral entropy', 'Spectral flatness', 'Spectral flux', 'Spectral kurtosis', 'Spectral mean', 'Spectral rolloff', 'Spectral skewness', 'Spectral slope', 'Spectral spread', 'Spectral variance', 'Tempo mean', 'Tempogram mean', 'Zero-crossing rate mean']
	singleTargetFloats: list[str] = ['Peak dB', 'RMS from waveform mean', 'RMS total', 'SRMR mean', 'Signal entropy', 'Spectral Bandwidth mean', 'Spectral Centroid mean', 'Spectral Contrast mean', 'Spectral Flatness mean', 'Spectral centroid mean', 'Spectral crest mean', 'Spectral decrease mean', 'Spectral entropy mean', 'Spectral flatness mean', 'Spectral flux mean', 'Spectral kurtosis mean', 'Power spectral density mean', 'Spectral rolloff mean', 'Spectral skewness mean', 'Spectral slope mean', 'Spectral spread mean', 'Spectral variance mean']
	rows: list[list[str | float | ndarray[tuple[Any, ...], dtype[Any]]]] = analyzeAudioListPathFilenames(lPFn, singleTargetFloats)

	dataTabularTOpathFilenameDelimited(lPFn[0].parent / 'l073.tab', rows, ['pathFilename', *singleTargetFloats])
	singleTargetFloats: list[str] = ['Crest factor', 'Spectral kurtosis', 'Signal entropy', 'Spectral flatness']

	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[0], lPFn[1])
	# print(PSNR_channelsMean)
	# PSNR_channelsMean = audioAspects['Peak Signal-to-Noise Ratio mean']['analyzer'](lPFn[1], lPFn[0])
	# print(PSNR_channelsMean)
