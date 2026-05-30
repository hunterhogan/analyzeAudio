"""Analyze audio files with registered aspect analyzers.

(AI generated docstring)

You can use this module to compute named audio aspect values for one file or many files. The
module loads audio data, prepares common intermediate representations, and dispatches work to
analyzers registered in `analyzeAudio.audioAspectsRegistry.audioAspects` [1].

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

from analyzeAudio.audioAspectsRegistry import audioAspects
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Any, TYPE_CHECKING
import librosa
import numpy
import pathlib
import soundfile
import torch

if TYPE_CHECKING:
	from analyzeAudio import Audio, Spectrogram, SpectrogramMagnitude, SpectrogramPower
	from collections.abc import Sequence
	from concurrent.futures import Future
	from os import PathLike

def analyzeAudioFile(pathFilename: str | PathLike[Any], listAspectNames: Sequence[str]) -> list[str | float]:
	"""
	Compute requested aspect values for one audio file.

	You can use this function to evaluate each name in `listAspectNames` against
	`pathFilename`. The function reads `pathFilename`, prepares shared intermediate audio
	representations for analyzers registered in `analyzeAudio.audioAspectsRegistry.audioAspects`
	[1], and returns one value for each requested aspect name. If a name from
	`listAspectNames` is absent from `audioAspects` [1], the matching return entry is
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

	See Also
	--------
	`analyzeAudioListPathFilenames`
		Compute the same aspect values for many audio files.

	References
	----------
	[1] `analyzeAudio.audioAspectsRegistry.audioAspects`

	"""  # noqa: DOC501
	pathlib.Path(pathFilename).stat()  # raises FileNotFoundError if the file does not exist
	dictionaryAspectsAnalyzed: dict[str, str | float] = dict.fromkeys(listAspectNames, 'not found')
	"""Despite returning a list, use a dictionary to preserve the order of the listAspectNames.
	Similarly, 'not found' ensures the returned list length == len(listAspectNames)"""

	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate  # pyright: ignore[reportUnusedVariable]  # noqa: F841
		waveform: Audio = readSoundFile.read(dtype='float32').astype(numpy.float32)
		waveform = waveform.T

	# I need "lazy" loading
	tryAgain: bool = True
	while tryAgain:
		try:
			tensorAudio: torch.Tensor = torch.from_numpy(waveform)  # pyright: ignore[reportUnusedVariable, reportUnknownMemberType] # memory-sharing  # noqa: F841
			tryAgain = False
		except RuntimeError as ERRORmessage:  # noqa: PERF203
			if 'negative stride' in str(ERRORmessage):
				waveform = waveform.copy()  # not memory-sharing
				tryAgain = True
			else:
				raise ERRORmessage  # noqa: TRY201

	spectrogram: Spectrogram = librosa.stft(waveform)
	spectrogramMagnitude: SpectrogramMagnitude = numpy.absolute(spectrogram)
	spectrogramPower: SpectrogramPower = spectrogramMagnitude ** 2  # pyright: ignore[reportUnusedVariable] # noqa: F841

	pytorchOnCPU: bool = not torch.cuda.is_available()  # pyright: ignore[reportUnusedVariable] # False if GPU available, True if not  # noqa: F841

	for aspectName in listAspectNames:
		if aspectName in audioAspects:
			analyzer = audioAspects[aspectName]['analyzer']
			analyzerParameters: list[str] = audioAspects[aspectName]['analyzerParameters']
			dictionaryAspectsAnalyzed[aspectName] = analyzer(*map(vars().get, analyzerParameters))

	return [dictionaryAspectsAnalyzed[aspectName] for aspectName in listAspectNames]

def analyzeAudioListPathFilenames(listPathFilenames: Sequence[str] | Sequence[PathLike[Any]], listAspectNames: Sequence[str], CPUlimit: int | None = None) -> list[list[str | float]]:
	"""
	Compute requested aspect values for many audio files.

	You can use this function to evaluate the same `listAspectNames` against each path in
	`listPathFilenames`. The function calls `analyzeAudioFile` [1] in parallel workers and
	returns one row for each completed file. Each row begins with the file path normalized to
	POSIX text, followed by the aspect values aligned with `listAspectNames`. You can write
	the returned rows directly with `analyzeAudio.dataTabularTOpathFilenameDelimited` [2].

	Parameters
	----------
	listPathFilenames : Sequence[str] | Sequence[PathLike[Any]]
		Path sequence of audio files to analyze.
	listAspectNames : Sequence[str]
		Audio aspect name sequence to evaluate for each file.
	CPUlimit : int | None = None
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

	See Also
	--------
	`analyzeAudioFile`
		Compute requested aspect values for one audio file.
	`analyzeAudio.dataTabularTOpathFilenameDelimited`
		Write returned rows to a delimited text file.

	References
	----------
	[1] `analyzeAudioFile`

	[2] `analyzeAudio.dataTabularTOpathFilenameDelimited`

	"""
	rowsListFilenameAspectValues: list[list[str | float]] = []

	# TODO if hunterMakesPy is changed to python>=3.10
	# max_workers = defineConcurrencyLimit(limit=CPUlimit)  # noqa: ERA001

	with ProcessPoolExecutor(max_workers=CPUlimit) as concurrencyManager:
		dictionaryConcurrency: dict[Future[list[str | float]], str | PathLike[Any]] = {concurrencyManager.submit(analyzeAudioFile, pathFilename, listAspectNames)
									: pathFilename
									for pathFilename in listPathFilenames}

		for claimTicket in as_completed(dictionaryConcurrency):
			listAspectValues: list[str | float] = claimTicket.result()
			rowsListFilenameAspectValues.append(
				[str(pathlib.PurePath(dictionaryConcurrency[claimTicket]).as_posix())]  # noqa: RUF005
				+ listAspectValues)

	return rowsListFilenameAspectValues
