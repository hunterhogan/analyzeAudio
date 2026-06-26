from __future__ import annotations

from analyzeAudio import settingsPackage
from datetime import date
from tests import ContestPathFilenames
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'
listPathFilenamesDataSamples: tuple[Path, ...] = tuple(pathDataSamples.glob('*.wav'))

pathFilenameMixture: Path = pathAudioFractions / 'reference_mixture_Hz44100.flac'
listPathFilenamesContests: list[ContestPathFilenames] = []
for pathFilenameReference in pathAudioFractions.glob('reference_*.flac'):
	for pathFilenameComparand in pathAudioFractions.glob(f'{pathFilenameReference.stem.replace("reference_", "comparand_").replace("_Hz", "_*_Hz")}*.flac'):
		listPathFilenamesContests.append(ContestPathFilenames(pathFilenameReference, pathFilenameComparand))  # noqa: PERF401

randomSeed: int = date.today().toordinal()
