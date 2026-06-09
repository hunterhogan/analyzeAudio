# ruff: noqa: T201 F811
from __future__ import annotations

from analyzeAudio import audioContests, dataTabularTOpathFilenameDelimited, getListAvailableAudioAspects, settingsPackage
from analyzeAudio.analyze import analyzeAudioListPathFilenames
from pathlib import Path
import pathlib

if __name__ == "__main__":

	listPathFilenames: list[pathlib.Path] = (list(pathlib.Path("/apps/analyzeAudio/tests/dataSamples").glob("ch2_44*.wav")))

	if True:
		listAspectNames: list[str] = ['Crest factor', 'Spectral kurtosis mean', 'Signal entropy', 'Spectral flatness mean']
		listAspectNames: list[str] = ['Tempogram mean']
		listAspectNames: list[str] = getListAvailableAudioAspects()
		rows: list[list[str | float]] = analyzeAudioListPathFilenames(listPathFilenames, listAspectNames, CPUlimit=.5)

		dataTabularTOpathFilenameDelimited(settingsPackage.pathPackage.parent / 'aspects.tab', rows, ['pathFilename', *listAspectNames])

	if False:
		alfa = Path('/apps/analyzeAudio/tests/dataSamples/SpeakSoftly_BrokenMan60sec/reference_vocals.wav')
		beta = Path('/apps/analyzeAudio/tests/dataSamples/SpeakSoftly_BrokenMan60sec/comparand_vocals_bad.wav')
		contest = 'Peak Signal-to-Noise Ratio mean'
		contest = 'SI-SDR mean'
		PSNR_channelsMean = audioContests[contest]['analyzer'](alfa, beta)
		print(PSNR_channelsMean)
		PSNR_channelsMean = audioContests[contest]['analyzer'](beta, alfa)
		print(PSNR_channelsMean)
