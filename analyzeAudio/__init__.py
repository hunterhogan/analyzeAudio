# noqa: D104
from __future__ import annotations

# isort: split
from analyzeAudio._theTypes import (
	analyzersAudioAspects as analyzersAudioAspects, Audio as Audio, libturd as libturd, parameterSpecifications as parameterSpecifications,
	Spectrogram as Spectrogram, SpectrogramMagnitude as SpectrogramMagnitude, SpectrogramPower as SpectrogramPower,
	typeReturned as typeReturned)

# isort: split
from analyzeAudio._beDRY import truncateTensors as truncateTensors

# isort: split
from analyzeAudio.audioAspectsRegistry import audioAspects as audioAspects, getListAvailableAudioAspects as getListAvailableAudioAspects

# isort: split
# pyright: reportUnusedImport=false
# NOTE Importing the modules triggers the registration of analyzer functions.
from analyzeAudio import (
	analyzersUseFilename, analyzersUseSpectrogram, analyzersUseTensor, analyzersUseTensorSpectrogram, analyzersUseWaveform)

# isort: split
from analyzeAudio.analyze import analyzeAudioFile as analyzeAudioFile, analyzeAudioListPathFilenames as analyzeAudioListPathFilenames

# isort: split
from analyzeAudio._misfit import dataTabularTOpathFilenameDelimited as dataTabularTOpathFilenameDelimited
