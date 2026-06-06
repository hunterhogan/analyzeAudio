from __future__ import annotations

from analyzeAudio import settingsPackage
from hunterHearsPy import readAudioFile, stft
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'

listDataSamplesFilenames = [
	'ch1_16000_09s_s32le_Clipping.wav',
	'ch1_44100_01s_LUFS03_1kHz.wav',
	'ch1_44100_120s_LUFS18_FrequencySweep.wav',
	'ch1_44100_83s_LUFS23_VoiceAndMusic.wav',
	'ch1_48000_120s_LUFS18_FrequencySweep.wav',
	'ch1_48000_300s_s24le.wav',
	'ch1_48000_83s_LUFS24_VoiceAndMusic.wav',
	'ch1_96000_12.1s_f32le.wav',
	'ch2_44100_04s_LUFS10_RelGate.wav',
	'ch2_44100_04s_LUFS69.5_AbsGate.wav',
	'ch2_44100_05s_s16le.wav',
	'ch2_44100_09s_LUFS20_birdsPink.wav',
	'ch2_44100_29s_LUFS23_10000Hz.wav',
	'ch2_44100_29s_LUFS23_1000Hz.wav',
	'ch2_44100_29s_LUFS23_100Hz.wav',
	'ch2_44100_29s_LUFS23_2000Hz.wav',
	'ch2_44100_29s_LUFS23_25Hz.wav',
	'ch2_44100_29s_LUFS23_500Hz.wav',
	'ch2_44100_60s_f32le_01RMS.wav',
	'ch2_44100_60s_f32le_20RMS.wav',
	'ch2_44100_60s_f32le_40RMS.wav',
	'ch2_44100_60s_f32le_60RMS.wav',
	'ch2_44100_7.1s_s16le.wav',
	'ch2_44100_83s_LUFS23_VoiceAndMusic.wav',
	'ch2_44100_83s_LUFS24_VoiceAndMusic.wav',
	'ch2_48000_04s_LUFS10_RelGate.wav',
	'ch2_48000_04s_LUFS69.5_AbsGate.wav',
	'ch2_48000_6.3s_s16le.wav',
	'ch2_48000_83s_LUFS23_VoiceAndMusic.wav',
	'ch2_48000_83s_LUFS24_VoiceAndMusic.wav',
]

listDataSamplesPathFilenames: tuple[Path, ...] = tuple(map(pathDataSamples.joinpath, listDataSamplesFilenames))
