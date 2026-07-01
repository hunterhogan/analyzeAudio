# noqa: D104
from __future__ import annotations

# isort: split
from analyzeAudio._theSSOT import settingsPackage  # pyright: ignore[reportUnusedImport]

# isort: split
from analyzeAudio._theTypes import (
	AnalyzerAudioAspects as AnalyzerAudioAspects, ArrayAspect as ArrayAspect,
	ArrayAspectSpectrogramFramewise as ArrayAspectSpectrogramFramewise, ArrayAspectWaveformFramewise as ArrayAspectWaveformFramewise,
	ArrayChannelData as ArrayChannelData, ArrayOverallData as ArrayOverallData, Audio as Audio,
	AuralossChromaSTFTLoss as AuralossChromaSTFTLoss, ParametersMelSpectrogram as ParametersMelSpectrogram,
	SpectrogramMagnitude as SpectrogramMagnitude, SpectrogramPower as SpectrogramPower, 个 as 个, 归个 as 归个, 形 as 形)

# isort: split
from analyzeAudio._dataBaskets import BleedFull as BleedFull, BleedFullArray as BleedFullArray

# isort: split
from analyzeAudio._beDRY import KValue as KValue

# isort: split
from analyzeAudio.registry import (
	audioAspects as audioAspects, audioContests as audioContests, getListAvailableAudioAspects as getListAvailableAudioAspects,
	getListAvailableAudioContests as getListAvailableAudioContests, getListAvailableAudioMetadata as getListAvailableAudioMetadata)

# NOTE Importing the modules triggers the registration of functions.
# isort: split
from analyzeAudio import analyzeFilename, analyzeSpectrogram, analyzeTensor, analyzeWaveform  # pyright: ignore[reportUnusedImport]

# isort: split
from analyzeAudio import contestsSpectrogram, contestsTensor, contestsTensorSpectrogram  # pyright: ignore[reportUnusedImport]

# isort: split
from analyzeAudio.analyze import analyzeAudioFile as analyzeAudioFile, analyzeAudioListPathFilenames as analyzeAudioListPathFilenames

# isort: split
from analyzeAudio.metadata import getMetadata as getMetadata, getMetadataPathFilenames as getMetadataPathFilenames

# isort: split
from analyzeAudio._misfit import dataTabularTOpathFilenameDelimited as dataTabularTOpathFilenameDelimited
