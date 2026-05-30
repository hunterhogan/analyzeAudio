# noqa: D100
from __future__ import annotations

from analyzeAudio.audioAspectsRegistry import audioAspects
from concurrent.futures import as_completed, ProcessPoolExecutor
from hunterMakesPy.parseParameters import defineConcurrencyLimit, oopsieKwargsie
from typing import Any, TYPE_CHECKING
from Z0Z_tools import stft
import numpy
import pathlib
import soundfile
import torch

if TYPE_CHECKING:
	from collections.abc import Sequence
	from numpy.typing import NDArray
	from os import PathLike

def analyzeAudioFile(pathFilename: str | PathLike[Any], listAspectNames: list[str]) -> list[str | float | NDArray[Any]]:
	"""
	Analyzes an audio file for specified aspects and returns the results.

	Parameters
	----------
	pathFilename : str or PathLike
		The path to the audio file to be analyzed.
	listAspectNames : list of str
		A list of aspect names to analyze in the audio file.

	Returns
	-------
	listAspectValues : list of (str or float or NDArray)
		A list of analyzed values in the same order as `listAspectNames`.

	"""  # noqa: DOC501
	pathlib.Path(pathFilename).stat()  # raises FileNotFoundError if the file does not exist
	dictionaryAspectsAnalyzed: dict[str, str | float | NDArray[Any]] = dict.fromkeys(listAspectNames, 'not found')
	"""Despite returning a list, use a dictionary to preserve the order of the listAspectNames.
	Similarly, 'not found' ensures the returned list length == len(listAspectNames)"""

	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform = readSoundFile.read(dtype='float32').astype(numpy.float32)
		waveform = waveform.T

	# I need "lazy" loading
	tryAgain = True
	while tryAgain:
		try:
			tensorAudio = torch.from_numpy(waveform)  # pyright: ignore[reportUnusedVariable, reportUnknownMemberType] # memory-sharing  # noqa: F841
			tryAgain = False
		except RuntimeError as ERRORmessage:
			if 'negative stride' in str(ERRORmessage):
				waveform = waveform.copy()  # not memory-sharing
				tryAgain = True
			else:
				raise ERRORmessage  # noqa: TRY201

	spectrogram = stft(waveform, sampleRate=sampleRate)
	spectrogramMagnitude = numpy.absolute(spectrogram)
	spectrogramPower = spectrogramMagnitude ** 2  # pyright: ignore[reportUnusedVariable] # noqa: F841

	pytorchOnCPU = not torch.cuda.is_available()  # pyright: ignore[reportUnusedVariable] # False if GPU available, True if not  # noqa: F841

	for aspectName in listAspectNames:
		if aspectName in audioAspects:
			analyzer = audioAspects[aspectName]['analyzer']
			analyzerParameters = audioAspects[aspectName]['analyzerParameters']
			dictionaryAspectsAnalyzed[aspectName] = analyzer(*map(vars().get, analyzerParameters))

	return [dictionaryAspectsAnalyzed[aspectName] for aspectName in listAspectNames]


def analyzeAudioListPathFilenames(listPathFilenames: Sequence[str] | Sequence[PathLike[Any]], listAspectNames: list[str], CPUlimit: int | float | bool | None = None) -> list[list[str | float | NDArray[Any]]]:  # noqa: FBT001
	"""
	Analyzes a list of audio files for specified aspects of the individual files and returns the results.

	Parameters
	----------
	listPathFilenames : Sequence of str or PathLike
		A list of paths to the audio files to be analyzed.
	listAspectNames : list of str
		A list of aspect names to analyze in each audio file.
	CPUlimit : int, float, bool, or None, default=None
		Whether and how to limit the CPU usage. See notes for details.

	Returns
	-------
	rowsListFilenameAspectValues : list of list of (str or float or NDArray)
		A list of lists, where each inner list contains the filename and analyzed values corresponding to the specified aspects, which are in the same order as `listAspectNames`.

	You can save the data with `Z0Z_tools.dataTabularTOpathFilenameDelimited()`.
	For example,

	```python
	dataTabularTOpathFilenameDelimited(
		pathFilename = pathFilename,
		tableRows = rowsListFilenameAspectValues, # The return of this function
		tableColumns = ['File'] + listAspectNames # A parameter of this function
	)
	```

	Nevertheless, I aspire to improve `analyzeAudioListPathFilenames` by radically improving the structure of the returned data.

	Limits on CPU usage CPUlimit:
		False, None, or 0: No limits on CPU usage; uses all available CPUs. All other values will potentially limit CPU usage.
		True: Yes, limit the CPU usage; limits to 1 CPU.
		Integer >= 1: Limits usage to the specified number of CPUs.
		Decimal value (float) between 0 and 1: Fraction of total CPUs to use.
		Decimal value (float) between -1 and 0: Fraction of CPUs to *not* use.
		Integer <= -1: Subtract the absolute value from total CPUs.

	"""
	rowsListFilenameAspectValues: list[list[str | float | NDArray[Any]]] = []

	if not (CPUlimit is None or isinstance(CPUlimit, (bool, int, float))):
		CPUlimit = oopsieKwargsie(CPUlimit)  # ty:ignore[invalid-assignment]
	max_workers = defineConcurrencyLimit(limit=CPUlimit)

	with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
		dictionaryConcurrency = {concurrencyManager.submit(analyzeAudioFile, pathFilename, listAspectNames)
									: pathFilename
									for pathFilename in listPathFilenames}

		for claimTicket in as_completed(dictionaryConcurrency):
			listAspectValues = claimTicket.result()
			rowsListFilenameAspectValues.append(
				[str(pathlib.PurePath(dictionaryConcurrency[claimTicket]).as_posix())]  # noqa: RUF005
				+ listAspectValues)

	return rowsListFilenameAspectValues
