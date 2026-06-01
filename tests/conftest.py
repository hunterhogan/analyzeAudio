"""Zero-configuration conftest for src-layout projects."""
from __future__ import annotations

import pathlib
import numpy
import pytest
import torch

pathDataSamples = pathlib.Path(__file__).parent / "dataSamples"

listOfFiles = ["pink-20RMS60sec.wav",
"pink-40RMS60sec.wav",
"pink-60RMS60sec.wav",
"testParkMono96kHz32float12.1sec.wav",
"testPink2ch7.1sec.wav",
"testSine2ch5sec.wav",
"testSine2ch5secCopy1.wav",
"testTrain2ch48kHz6.3sec.wav",
"testVideo11sec.mkv",
"testWooWooMono16kHz32integerClipping9sec.wav",
]

@pytest.fixture
def tensorAudioAuralossCase() -> tuple[torch.Tensor, torch.Tensor, int]:
	intSampleCount = 33011
	tensorPhase = torch.linspace(0.17, 23.17, steps=intSampleCount, dtype=torch.float32)
	tensorAudioAlfa = torch.stack((
		torch.sin(tensorPhase) * 0.73,
		torch.cos(tensorPhase * 0.61) * 0.57,
	), dim=0).unsqueeze(0)
	tensorAudioBeta = torch.stack((
		torch.sin(tensorPhase * 0.97 + 0.29) * 0.69,
		torch.cos(tensorPhase * 0.59 + 0.41) * 0.53,
	), dim=0).unsqueeze(0)
	return tensorAudioAlfa, tensorAudioBeta, 44100

@pytest.fixture
def tensorSpectrogramMagnitudeAuralossCase() -> tuple[torch.Tensor, torch.Tensor]:
	tensorSpectrogramMagnitudeAlfa = torch.linspace(0.17, 19.17, steps=257 * 137, dtype=torch.float32).reshape(1, 257, 137)
	tensorSpectrogramMagnitudeBeta = (tensorSpectrogramMagnitudeAlfa * 0.83) + 0.19
	return tensorSpectrogramMagnitudeAlfa, tensorSpectrogramMagnitudeBeta

@pytest.fixture
def spectrogramFeatureCase() -> tuple[numpy.ndarray, numpy.ndarray, int]:
	arraySpectrogramMagnitude = numpy.linspace(0.17, 23.17, num=257 * 149, dtype=numpy.float32).reshape(257, 149)
	arraySpectrogramPower = numpy.square(arraySpectrogramMagnitude)
	return arraySpectrogramMagnitude, arraySpectrogramPower, 44100

@pytest.fixture
def waveformLibrosaCase() -> tuple[numpy.ndarray, int]:
	intSampleCount = 131071
	arrayPhase = numpy.linspace(0.17, 97.17, num=intSampleCount, dtype=numpy.float32)
	arrayWaveform = (
		numpy.sin(arrayPhase) * 0.73
		+ numpy.cos((arrayPhase * 0.61) + 0.29) * 0.17
		+ numpy.sin((arrayPhase * 0.13) + 0.41) * 0.11
	).astype(numpy.float32)
	return arrayWaveform, 44100
