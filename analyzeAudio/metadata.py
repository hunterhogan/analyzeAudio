# ruff: noqa: D100 D103
from __future__ import annotations

from analyzeAudio.registry import audioMetadata
from concurrent.futures import as_completed, ProcessPoolExecutor
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from pathlib import PurePath
from tqdm.auto import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Callable, Sequence
	from concurrent.futures import Future
	from os import PathLike
	from typing import Any

def getMetadata(pathFilename: str | PathLike[Any], listMetadataNames: Sequence[str]) -> tuple[str, ...]:
	dictionaryMetadata: dict[str, str] = dict.fromkeys(listMetadataNames, 'not found')

	for aspectName in filter(audioMetadata.__contains__, listMetadataNames):
		analyzer: Callable[[str | PathLike[Any]], str] = audioMetadata[aspectName]['analyzer']
		dictionaryMetadata[aspectName] = analyzer(pathFilename)

	return tuple(map(dictionaryMetadata.__getitem__, listMetadataNames))

def getMetadataPathFilenames(listPathFilenames: Sequence[str | PathLike[Any]], listMetadataNames: Sequence[str], *, CPUlimit: bool | float | int | None = None) -> list[list[str]]:
	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	with ProcessPoolExecutor(max_workers) as concurrencyManager:
		dictionaryConcurrency: dict[Future[tuple[str, ...]], str | PathLike[Any]] = {
			concurrencyManager.submit(getMetadata, pathFilename, listMetadataNames): pathFilename
				for pathFilename in listPathFilenames}

		disabled: bool = True
		if (3 < len(listPathFilenames) and (5 < (max(len(listPathFilenames) / max_workers, 1) * len(listMetadataNames)))):
			disabled = False

		rowsListFilenameMetadata: list[list[str]] = [
			[PurePath(dictionaryConcurrency[claimTicket]).as_posix(), *claimTicket.result()] for claimTicket
				in tqdm(as_completed(dictionaryConcurrency), total=len(dictionaryConcurrency), unit='files', desc='Get metadata'
					, leave=False, disable=disabled)
		]

	return rowsListFilenameMetadata
