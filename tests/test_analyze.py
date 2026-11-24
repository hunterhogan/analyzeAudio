"""Each expected aspect value is the accepted recording of the analyzer registered under the aspect name, in `tests.dataSamples.expected`."""
from __future__ import annotations

from analyzeAudio.analyze import analyzeAudioFile
from analyzeAudio.analyzeSpectrogram import analyzeChromagramMean, analyzeSpectralCentroidMean
from analyzeAudio.analyzeWaveform import analyzeRMSWaveformMean, analyzeTempoMean
from tests import assert_approx, assertEqualTo
from typing import TYPE_CHECKING
import pytest

if TYPE_CHECKING:
	from hunterMakesPy import CallableFunction
	from pathlib import Path

@pytest.mark.parametrize('aspectName, analyzer', (
	pytest.param('Chromagram mean', analyzeChromagramMean, id='ChromagramMean'),
	pytest.param('RMS Waveform mean', analyzeRMSWaveformMean, id='RMSWaveformMean'),
	pytest.param('Spectral Centroid mean', analyzeSpectralCentroidMean, id='SpectralCentroidMean'),
	pytest.param('Tempo mean', analyzeTempoMean, id='TempoMean'),
))
def test_analyzeAudioFile(pathFilename: Path, aspectName: str, analyzer: CallableFunction[..., float], expectedAspect: float) -> None:
	"""`analyzeAudioFile` reads the file at `pathFilename` and passes the analyzer registered as `aspectName` the arguments that `analyzer` receives in its own test."""
	listAspectNames: tuple[str, ...] = ('Unregistered aspect', aspectName)
	aspectNotFound, aspectValue = analyzeAudioFile(pathFilename, listAspectNames)
	assertEqualTo(aspectNotFound, 'not found', analyzeAudioFile.__name__, pathFilename.name, listAspectNames)
	assert_approx(aspectValue, expectedAspect, 1e-6, 1e-12, analyzeAudioFile.__name__, pathFilename.name, listAspectNames)
