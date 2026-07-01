# ruff: noqa: T201 F811 D100
from __future__ import annotations

from analyzeAudio import (
	audioContests, dataTabularTOpathFilenameDelimited, getListAvailableAudioAspects, getListAvailableAudioMetadata, getMetadataPathFilenames,
	settingsPackage)
from analyzeAudio.analyze import analyzeAudioListPathFilenames
from pathlib import Path
from time import perf_counter

if __name__ == "__main__":

	listPathFilenames: list[Path] = (list(Path("/apps/analyzeAudio/tests/dataSamples").glob("ch2_44*.wav")))
	listPathFilenames: list[Path] = (list(Path("/apps/analyzeAudio/tests/dataSamples").glob("*.wav")))
	listPathFilenames: list[Path] = (list(Path("/data/MusicDemixingBenchmarks/synthetic").glob("*.wav")))

	timeStart = perf_counter()

	if True:
		listMetadataNames: list[str] = ['channels', 'sample_rate']
		listMetadataNames: list[str] = getListAvailableAudioMetadata()
		rows: list[list[str]] = getMetadataPathFilenames(listPathFilenames, listMetadataNames, CPUlimit=-2)

		dataTabularTOpathFilenameDelimited(settingsPackage.pathPackage.parent / 'metadata.tab', rows, ['pathFilename', *listMetadataNames])

	if False:
		listAspectNames: list[str] = ['NISQA mean']
		listAspectNames: list[str] = ['Crest_factor mean', 'Spectral kurtosis mean', 'Entropy mean', 'Spectral flatness mean']
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

	print(perf_counter() - timeStart)
