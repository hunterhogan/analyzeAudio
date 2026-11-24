"""Expected values are the accepted recordings in `tests.dataSamples.SpeakSoftly_BrokenMan60sec.expected.expectedSpectrogram`.

The recordings floor each mel-scaled decibel spectrogram at `top_db` below the peak of the whole array, as
`librosa.amplitude_to_db` did before librosa 1.0. From librosa 1.0, `librosa.amplitude_to_db` floors each
channel at `top_db` below the peak of that channel.
"""
from __future__ import annotations

from analyzeAudio.contestsSpectrogram import analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean
from tests import assert_approx
from tests.dataSamples.audioFiles import sampleRateSpeakSoftly
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from analyzeAudio import SpectrogramMagnitude
	from hunterMakesPy import CallableFunction
	from pathlib import Path

@pytest.mark.parametrize('analyzer', (analyzeBleedlessMelDBMean, analyzeFullnessMelDBMean))
def test_contestSpectrogramMagnitude(analyzer: CallableFunction[..., float], pathFilenamesContest: tuple[Path, Path], spectrogramMagnitudeAlfa: SpectrogramMagnitude
	, spectrogramMagnitudeBeta: SpectrogramMagnitude, expectedContest: float) -> None:
	assert_approx(analyzer(spectrogramMagnitudeAlfa, spectrogramMagnitudeBeta, sr=sampleRateSpeakSoftly), expectedContest, 1e-6, 1e-12
		, analyzer.__name__, pathFilenamesContest[0].name, pathFilenamesContest[1].name, sr=sampleRateSpeakSoftly)
