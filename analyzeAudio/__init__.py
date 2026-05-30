# noqa: D104
# pyright: reportUnusedImport=false
from __future__ import annotations

# isort: split
from analyzeAudio._theTypes import (
	analyzersAudioAspects as analyzersAudioAspects, parameterSpecifications as parameterSpecifications, typeReturned as typeReturned)

# isort: split
from analyzeAudio.audioAspectsRegistry import (
	audioAspects as audioAspects, getListAvailableAudioAspects as getListAvailableAudioAspects, registrationAudioAspect)

# isort: split
from analyzeAudio import analyzersUseFilename, analyzersUseSpectrogram, analyzersUseTensor, analyzersUseWaveform

# isort: split
from analyzeAudio.analyze import analyzeAudioFile as analyzeAudioFile, analyzeAudioListPathFilenames as analyzeAudioListPathFilenames
