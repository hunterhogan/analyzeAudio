from __future__ import annotations

from analyzeAudio import settingsPackage
from hunterHearsPy import readAudioFile, stft
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'

listFilenames = [
	'pink-20RMS60sec.wav',
	'pink-40RMS60sec.wav',
	'pink-60RMS60sec.wav',
	'testParkMono96kHz32float12.1sec.wav',
	'testPink2ch7.1sec.wav',
	'testSine2ch5sec.wav',
	'testSine2ch5secCopy1.wav',
	'testTrain2ch48kHz6.3sec.wav',
	'testVideo11sec.mkv',
	'testWooWooMono16kHz32integerClipping9sec.wav',
]
