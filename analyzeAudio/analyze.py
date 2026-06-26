"""Analyze audio files with registered aspect analyzers.

(AI generated docstring)

You can use this module to compute named audio aspect values for one file or many files. The
module returns one value per requested aspect name and preserves the requested aspect order.

Contents
--------
Functions
	analyzeAudioFile
		Compute requested aspect values for one audio file.
	analyzeAudioListPathFilenames
		Compute requested aspect values for many audio files.

References
----------
[1] `analyzeAudio.audioAspectsRegistry.audioAspects`

"""
from __future__ import annotations

from analyzeAudio.registry import audioAspects
from concurrent.futures import as_completed, ProcessPoolExecutor
from hunterHearsPy import stft
from hunterMakesPy.parseParameters import defineConcurrencyLimit
from pathlib import PurePath
from tqdm.auto import tqdm
from typing import TYPE_CHECKING
import numpy
import soundfile
import torch

if TYPE_CHECKING:
	from analyzeAudio import Audio, SpectrogramMagnitude, SpectrogramPower
	from collections.abc import Callable, Sequence
	from concurrent.futures import Future
	from hunterHearsPy.theTypes import Spectrogram
	from os import PathLike
	from torch import Tensor
	from typing import Any

def analyzeAudioFile(pathFilename: str | PathLike[Any], listAspectNames: Sequence[str]) -> tuple[str | float, ...]:
	"""
	Compute requested aspect values for one audio file.

	You can use this function to evaluate each name in `listAspectNames` against
	`pathFilename`. The function returns one value for each requested aspect name. If a
	name from `listAspectNames` is absent from `audioAspects` [1], the matching return entry is
	`'not found'`.

	Parameters
	----------
	pathFilename : str | PathLike[Any]
		Path to the audio file that the function reads.
	listAspectNames : Sequence[str]
		Audio aspect name sequence to evaluate. The function preserves the order of
		`listAspectNames` in the returned list.

	Returns
	-------
	listAspectValues : list[str | float]
		One result for each entry in `listAspectNames`. Each result is either the analyzer
		value or `'not found'` when no analyzer is registered for the matching aspect name.

	References
	----------
	[1] `analyzeAudio.audioAspectsRegistry.audioAspects`

	"""  # noqa: DOC501
	dictionaryAspectsAnalyzed: dict[str, str | float] = dict.fromkeys(listAspectNames, 'not found')
	"""Despite returning a list, use a dictionary to preserve the order of the listAspectNames.
	Similarly, 'not found' ensures the returned list length == len(listAspectNames)"""

	# TODO I don't use `hunterHearsPy.readAudioFile` here because the sample rate is set by the
	# function instead of being read from the file.
	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate  # pyright: ignore[reportUnusedVariable]
		waveform: Audio = readSoundFile.read(dtype='float32', always_2d=True).astype(numpy.float32)
		waveform = waveform.T

	tryAgain: bool = True
	while tryAgain:
		try:
			# memory-sharing
			tensorAudio: Tensor = torch.from_numpy(waveform)  # pyright: ignore[reportUnusedVariable, reportUnknownMemberType] # noqa: F841
			tryAgain = False
		except (RuntimeError, ValueError) as ERRORmessage:  # noqa: PERF203
			if 'negative stride' in str(ERRORmessage):
				waveform = waveform.copy()  # not memory-sharing
				tryAgain = True
			else:
				raise RuntimeError from ERRORmessage

	spectrogram: Spectrogram = stft(waveform, sampleRate=sampleRate)
	spectrogramMagnitude: SpectrogramMagnitude = numpy.absolute(spectrogram)
	spectrogramPower: SpectrogramPower = spectrogramMagnitude ** 2  # pyright: ignore[reportUnusedVariable] # noqa: F841

	pytorchOnCPU: bool = not torch.cuda.is_available()  # pyright: ignore[reportUnusedVariable] # False if GPU available, True if not  # noqa: F841

	for aspectName in filter(audioAspects.__contains__, listAspectNames):
		analyzer: Callable[..., Any] = audioAspects[aspectName]['analyzer']
		analyzerParameters: list[str] = audioAspects[aspectName]['analyzerParameters']
		dictionaryAspectsAnalyzed[aspectName] = analyzer(*map(vars().get, analyzerParameters))

	return tuple(map(dictionaryAspectsAnalyzed.__getitem__, listAspectNames))

def analyzeAudioListPathFilenames(listPathFilenames: Sequence[str | PathLike[Any]], listAspectNames: Sequence[str], *, CPUlimit: bool | float | int | None = None) -> list[list[str | float]]:
	"""
	Compute requested aspect values for many audio files.

	You can use this function to evaluate the same `listAspectNames` against each path in
	`listPathFilenames`. The function returns one row per analyzed file. Each row begins with
	the file path normalized to POSIX text, followed by the aspect values aligned with
	`listAspectNames`. You can write the returned rows directly with
	`analyzeAudio.dataTabularTOpathFilenameDelimited` [2].

	Parameters
	----------
	listPathFilenames : Sequence[str] | Sequence[PathLike[Any]]
		Path sequence of audio files to analyze.
	listAspectNames : Sequence[str]
		Audio aspect name sequence to evaluate for each file.
	CPUlimit : bool | float | int | None = None
		Worker-count value for the process pool. Use `None` for the default worker count, or
		use a positive integer for an explicit worker count. The function forwards `CPUlimit`
		directly to the worker pool without additional normalization, so values less than `1`
		fail in the worker pool.

	Returns
	-------
	rowsListFilenameAspectValues : list[list[str | float]]
		Row sequence of analyzed output. Each row contains the POSIX text form of one
		`pathFilename` in column `0`, followed by the aspect values aligned with
		`listAspectNames`.

	Result ordering
	---------------
	row order : completion order
		`rowsListFilenameAspectValues` follows worker completion order rather than the input
		order of `listPathFilenames`.

	Examples
	--------
	Use the returned rows with `analyzeAudio.dataTabularTOpathFilenameDelimited` [2].

	```python
	from analyzeAudio import dataTabularTOpathFilenameDelimited
	from analyzeAudio.analyze import analyzeAudioListPathFilenames
	import pathlib

	lPFn = list(pathlib.Path('/apps/analyzeAudio/tests/dataSamples').rglob('test*.wav'))
	singleTargetFloats = ['Crest factor', 'Spectral flatness']
	rows = analyzeAudioListPathFilenames(lPFn, singleTargetFloats)

	dataTabularTOpathFilenameDelimited(
		lPFn[0].parent.parent.parent / 'l073.tab',
		rows,
		['pathFilename', *singleTargetFloats],
	)
	```

	References
	----------
	[1] `analyzeAudioFile`

	[2] `analyzeAudio.dataTabularTOpathFilenameDelimited`

	"""
	max_workers: int = defineConcurrencyLimit(limit=CPUlimit)

	with ProcessPoolExecutor(max_workers) as concurrencyManager:
		dictionaryConcurrency: dict[Future[tuple[str | float, ...]], str | PathLike[Any]] = {
			concurrencyManager.submit(analyzeAudioFile, pathFilename, listAspectNames): pathFilename
				for pathFilename in listPathFilenames}

		disabled: bool = True
		if (3 < len(listPathFilenames) and (5 < (max(len(listPathFilenames) / max_workers, 1) * len(listAspectNames)))):
			disabled = False

		rowsListFilenameAspectValues: list[list[str | float]] = [
			[PurePath(dictionaryConcurrency[claimTicket]).as_posix(), *claimTicket.result()] for claimTicket
				in tqdm(as_completed(dictionaryConcurrency), total=len(dictionaryConcurrency), unit='files', desc='Analyze audio file'
					, leave=False, disable=disabled)
		]

	return rowsListFilenameAspectValues
