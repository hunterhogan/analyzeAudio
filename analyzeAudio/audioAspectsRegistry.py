from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.typing import NDArray
from typing import Any, cast, ParamSpec, TypeAlias, TYPE_CHECKING, TypeVar
from Z0Z_tools import defineConcurrencyLimit, oopsieKwargsie, stft
import cachetools
import inspect
import librosa
import multiprocessing
import numpy
from os import PathLike
import pathlib
import soundfile
import torch
import warnings

if TYPE_CHECKING:
	from typing import TypedDict
else:
	TypedDict = dict

if __name__ == '__main__':
	multiprocessing.set_start_method('spawn')

warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics', message='.*fast=True.*')

parameterSpecifications = ParamSpec('parameterSpecifications')
typeReturned = TypeVar('typeReturned')

audioAspect: TypeAlias = str
class analyzersAudioAspects(TypedDict):
	analyzer: Callable[..., Any]
	analyzerParameters: list[str]

audioAspects: dict[audioAspect, analyzersAudioAspects] = {}
"""A register of 1) measurable aspects of audio data, 2) analyzer functions to measure audio aspects, 3) and parameters of analyzer functions."""

def registrationAudioAspect(aspectName: str) -> Callable[[Callable[parameterSpecifications, typeReturned]], Callable[parameterSpecifications, typeReturned]]:
	"""
	A function to "decorate" a registrant-analyzer function and the aspect of audio data it can analyze.

	Parameters:
		aspectName: The audio aspect that the registrar will enter into the register, `audioAspects`.
	"""

	def registrar(registrant: Callable[parameterSpecifications, typeReturned]) -> Callable[parameterSpecifications, typeReturned]:
		"""
		`registrar` updates the registry, `audioAspects`, with 1) the analyzer function, `registrant`, 2) the analyzer function's parameters, and 3) the aspect of audio data that the analyzer function measures.

		Parameters:
			registrant: The function that analyzes an aspect of audio data.

		Note:
			`registrar` does not change the behavior of `registrant`, the analyzer function.
		"""
		audioAspects[aspectName] = {
			'analyzer': registrant,
			'analyzerParameters': inspect.getfullargspec(registrant).args
		}

		# if registrant.__annotations__.get('return') is not None and issubclass(registrant.__annotations__['return'], subclassTarget): # maybe someday I will understand why this doesn't work
		# if registrant.__annotations__.get('return') is not None and issubclass(registrant.__annotations__.get('return', type(None)), subclassTarget): # maybe someday I will understand why this doesn't work
		if isinstance(registrant.__annotations__.get('return', type(None)), type) and issubclass(registrant.__annotations__.get('return', type(None)), numpy.ndarray): # maybe someday I will understand what all of this statement means
			def registrationAudioAspectMean(*arguments: parameterSpecifications.args, **keywordArguments: parameterSpecifications.kwargs) -> numpy.floating[Any]:
				"""
				`registrar` updates the registry with a new analyzer function that calculates the mean of the analyzer's numpy.ndarray result.

				Returns:
					mean: Mean value of the analyzer's numpy.ndarray result.
				"""
				aspectValue = registrant(*arguments, **keywordArguments)
				return numpy.mean(cast(NDArray[Any], aspectValue))
				# return aspectValue.mean()
			audioAspects[f"{aspectName} mean"] = {
				'analyzer': registrationAudioAspectMean,
				'analyzerParameters': inspect.getfullargspec(registrant).args
			}
		return registrant
	return registrar

def analyzeAudioFile(pathFilename: str | PathLike[Any], listAspectNames: list[str]) -> list[str | float | NDArray[Any]]:
	"""
	Analyzes an audio file for specified aspects and returns the results.

	Parameters:
		pathFilename: The path to the audio file to be analyzed.
		listAspectNames: A list of aspect names to analyze in the audio file.

	Returns:
		listAspectValues: A list of analyzed values in the same order as `listAspectNames`.
	"""
	pathlib.Path(pathFilename).stat() # raises FileNotFoundError if the file does not exist
	dictionaryAspectsAnalyzed: dict[str, str | float | NDArray[Any]] = {aspectName: 'not found' for aspectName in listAspectNames}
	"""Despite returning a list, use a dictionary to preserve the order of the listAspectNames.
	Similarly, 'not found' ensures the returned list length == len(listAspectNames)"""

	with soundfile.SoundFile(pathFilename) as readSoundFile:
		sampleRate: int = readSoundFile.samplerate
		waveform = readSoundFile.read(dtype='float32').astype(numpy.float32)
		waveform = waveform.T

	# I need "lazy" loading
	tryAgain = True
	while tryAgain: # `tenacity`?
		try:
			tensorAudio = torch.from_numpy(waveform)  # memory-sharing
			tryAgain = False
		except RuntimeError as ERRORmessage:
			if 'negative stride' in str(ERRORmessage):
				waveform = waveform.copy()  # not memory-sharing
				tryAgain = True
			else:
				raise ERRORmessage

	spectrogram = stft(waveform, sampleRate=sampleRate)
	spectrogramMagnitude = numpy.absolute(spectrogram)
	spectrogramPower = spectrogramMagnitude ** 2

	pytorchOnCPU = not torch.cuda.is_available()  # False if GPU available, True if not

	for aspectName in listAspectNames:
		if aspectName in audioAspects:
			analyzer = audioAspects[aspectName]['analyzer']
			analyzerParameters = audioAspects[aspectName]['analyzerParameters']
			dictionaryAspectsAnalyzed[aspectName] = analyzer(*map(vars().get, analyzerParameters))

	return [dictionaryAspectsAnalyzed[aspectName] for aspectName in listAspectNames]

def analyzeAudioListPathFilenames(listPathFilenames: Sequence[str] | Sequence[PathLike[Any]], listAspectNames: list[str], CPUlimit: int | float | bool | None = None) -> list[list[str | float | NDArray[Any]]]:
	"""
	Analyzes a list of audio files for specified aspects of the individual files and returns the results.

	Parameters:
		listPathFilenames: A list of paths to the audio files to be analyzed.
		listAspectNames: A list of aspect names to analyze in each audio file.
		CPUlimit (gluttonous resource usage): whether and how to limit the CPU usage. See notes for details.

	Returns:
		rowsListFilenameAspectValues: A list of lists, where each inner list contains the filename and
		analyzed values corresponding to the specified aspects, which are in the same order as `listAspectNames`.

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
		CPUlimit = oopsieKwargsie(CPUlimit)
	max_workers = defineConcurrencyLimit(CPUlimit)

	with ProcessPoolExecutor(max_workers=max_workers) as concurrencyManager:
		dictionaryConcurrency = {concurrencyManager.submit(analyzeAudioFile, pathFilename, listAspectNames)
									: pathFilename
									for pathFilename in listPathFilenames}

		for claimTicket in as_completed(dictionaryConcurrency):
			cacheAudioAnalyzers.pop(dictionaryConcurrency[claimTicket], None)
			listAspectValues = claimTicket.result()
			rowsListFilenameAspectValues.append(
				[str(pathlib.PurePath(dictionaryConcurrency[claimTicket]).as_posix())]
				+ listAspectValues)

	return rowsListFilenameAspectValues

def getListAvailableAudioAspects() -> list[str]:
	"""
	Returns a sorted list of audio aspect names. All valid values for the parameter `listAspectNames`, for example,
	are returned by this function.

	Returns:
		listAvailableAudioAspects: The list of aspect names registered in `audioAspects`.
	"""
	return sorted(audioAspects.keys())

cacheAudioAnalyzers: cachetools.LRUCache[str | PathLike[Any], NDArray[Any]] = cachetools.LRUCache(maxsize=256)
