# ruff: noqa: T201 F811
from __future__ import annotations

from analyzeAudio import audioContests, dataTabularTOpathFilenameDelimited, getListAvailableAudioAspects, settingsPackage
from analyzeAudio.analyze import analyzeAudioListPathFilenames
import pathlib

if __name__ == "__main__":

	listPathFilenames: list[pathlib.Path] = (list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").glob("*Pink.wav")))
	listAspectNames: list[str] = ['Crest factor', 'Spectral kurtosis mean', 'Signal entropy', 'Spectral flatness mean']
	listAspectNames: list[str] = getListAvailableAudioAspects()
	listAspectNames: list[str] = ['Tempogram mean']
	rows: list[list[str | float]] = analyzeAudioListPathFilenames(listPathFilenames, listAspectNames)

	dataTabularTOpathFilenameDelimited(settingsPackage.pathPackage.parent / 'aspects.tab', rows, ['pathFilename', *listAspectNames])

	if False:
		PSNR_channelsMean = audioContests['Peak Signal-to-Noise Ratio mean']['analyzer'](listPathFilenames[0], listPathFilenames[1])
		print(PSNR_channelsMean)
		PSNR_channelsMean = audioContests['Peak Signal-to-Noise Ratio mean']['analyzer'](listPathFilenames[1], listPathFilenames[0])
		print(PSNR_channelsMean)
