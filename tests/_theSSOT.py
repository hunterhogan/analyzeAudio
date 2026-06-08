from __future__ import annotations

from analyzeAudio import settingsPackage
from datetime import date
from tests import ContestFilenames
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from pathlib import Path

pathDataSamples: Path = settingsPackage.pathPackage.parent / 'tests' / 'dataSamples'
pathAudioFractions: Path = pathDataSamples / 'SpeakSoftly_BrokenMan60sec'
listPathFilenamesDataSamples: tuple[Path, ...] = tuple(pathDataSamples.glob('*.wav'))

listPathFilenamesContests: list[ContestFilenames] = []
for pathFilenameReference in pathAudioFractions.glob('reference_*.wav'):
	for pathFilenameComparand in pathAudioFractions.glob(f'{pathFilenameReference.stem.replace("reference_", "comparand_")}*.wav'):
		listPathFilenamesContests.append(ContestFilenames(pathFilenameReference, pathFilenameComparand))  # noqa: PERF401

randomSeed: int = date.today().toordinal()
