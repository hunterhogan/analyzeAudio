"""
This module provides functionality to register and analyze various aspects of audio data.

Classes:
    analyzersAudioAspects: A TypedDict that defines the structure for storing analyzer functions and their parameters.

Functions:
    registrationAudioAspect(aspectName: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        A decorator function to register an analyzer function for a specific audio aspect.

    analyzeAudioFile(pathFilename: str, listAspectNames: List[str]) -> List[str | float]:
        Analyzes an audio file for specified aspects and returns the results.

    analyzeAudioListPathFilenames(listPathFilenames: List[str], listAspectNames: List[str]) -> List[List[str | float]]:
        Analyzes a list of audio files for specified aspects and returns the results.

    getListAvailableAudioAspects() -> List[str]:
        Returns a list of available audio aspects that can be analyzed.

Usage:
    Use the `registrationAudioAspect` decorator to register analyzer functions.
    Call `analyzeAudioFile` to analyze a single audio file.
    Call `analyzeAudioListPathFilenames` to analyze multiple audio files concurrently.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from numpy.typing import NDArray
from tqdm.auto import tqdm
from typing import Any, Callable, cast, Dict, List, Optional, ParamSpec, Sequence, TypeAlias, TypeVar, Union
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
import cachetools
import inspect
import librosa
import multiprocessing
import numpy
import os
import pathlib
import torch
import warnings

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics', message='.*fast=True.*')

parameterSpecifications = ParamSpec('parameterSpecifications')
typeReturned = TypeVar('typeReturned')

subclassTarget: TypeAlias = numpy.ndarray

class analyzersAudioAspects(TypedDict):
    analyzer: Callable[..., Any]
    analyzerParameters: List[str]

audioAspects: Dict[str, analyzersAudioAspects] = {}
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
        if isinstance(registrant.__annotations__.get('return', type(None)), type) and issubclass(registrant.__annotations__.get('return', type(None)), subclassTarget): # maybe someday I will understand what all of this statement means
            def registrationAudioAspectMean(*arguments: parameterSpecifications.args, **keywordArguments: parameterSpecifications.kwargs) -> numpy.floating[Any]:
                """
                `registrar` updates the registry with a new analyzer function that calculates the mean of the analyzer's numpy.ndarray result.

                Returns:
                    mean: Mean value of the analyzer's numpy.ndarray result.
                """
                aspectValue = registrant(*arguments, **keywordArguments)
                return numpy.mean(cast(subclassTarget, aspectValue))
                # return aspectValue.mean()
            audioAspects[f"{aspectName} mean"] = {
                'analyzer': registrationAudioAspectMean,
                'analyzerParameters': inspect.getfullargspec(registrant).args
            }
        return registrant
    return registrar

def analyzeAudioFile(pathFilename: Union[str, os.PathLike[Any]], listAspectNames: List[str]) -> List[Union[str, float, NDArray[Any]]]:
    """
    Analyzes an audio file for specified aspects and returns the results.

    Parameters:
        pathFilename: The path to the audio file to be analyzed.
        listAspectNames: A list of aspect names to analyze in the audio file.

    Returns:
        listAspectValues: A list of analyzed values in the same order as `listAspectNames`.
    """
    dictionaryAspectsAnalyzed: Dict[str, Union[str, float, NDArray[Any]]] = {aspectName: 'not found' for aspectName in listAspectNames}
    """Despite returning a list, use a dictionary to preserve the order of the listAspectNames.
    Similarly, 'not found' ensures the returned list length == len(listAspectNames)"""

    waveform, sampleRate = librosa.load(path=str(pathFilename), sr=None, mono=False)
    # need "lazy" loading
    tryAgain = True
    while tryAgain:
        try:
            tensorAudio = torch.from_numpy(waveform)  # memory-sharing
            tryAgain = False
        except RuntimeError as ERRORmessage:
            if 'negative stride' in str(ERRORmessage):
                waveform = waveform.copy()  # not memory-sharing
                tryAgain = True
            else:
                raise ERRORmessage

    spectrogram = librosa.stft(y=waveform)
    spectrogramMagnitude, DISCARDEDphase = librosa.magphase(D=spectrogram)
    spectrogramPower = numpy.absolute(spectrogram) ** 2

    pytorchOnCPU = not torch.cuda.is_available()  # False if GPU available, True if not

    for aspectName in listAspectNames:
        if aspectName in audioAspects:
            analyzer = audioAspects[aspectName]['analyzer']
            analyzerParameters = audioAspects[aspectName]['analyzerParameters']
            dictionaryAspectsAnalyzed[aspectName] = analyzer(*map(vars().get, analyzerParameters))

    return [dictionaryAspectsAnalyzed[aspectName] for aspectName in listAspectNames]

def analyzeAudioListPathFilenames(listPathFilenames: Union[Sequence[str], Sequence[os.PathLike[Any]]], listAspectNames: List[str], CPUlimit: Optional[Union[int, float, bool]] = None) -> List[List[Union[str, float, NDArray[Any]]]]:
    """
    Analyzes a list of audio files for specified aspects of the individual files and returns the results.

    Parameters:
        listPathFilenames: A list of paths to the audio files to be analyzed.
        listAspectNames: A list of aspect names to analyze in each audio file.
        Z0Z_concurrencyLevel (gluttonous resource usage): The maximum number of concurrent processes to use (default is None, which uses the number of CPUs).

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
    """
    rowsListFilenameAspectValues = []

    max_workers = None
    if CPUlimit is not None:
        if isinstance(CPUlimit, bool):
            if CPUlimit == False:
                max_workers = 1
        elif isinstance(CPUlimit, int):
            if CPUlimit > 0:
                max_workers = CPUlimit
            elif CPUlimit == 0:
                max_workers = None
            elif CPUlimit < 0:
                max_workers = max(multiprocessing.cpu_count() + CPUlimit, 1)
        elif isinstance(CPUlimit, float):
            max_workers = max(int(CPUlimit * multiprocessing.cpu_count()), 1)

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

def getListAvailableAudioAspects() -> List[str]:
    """
    Returns a sorted list of audio aspect names. All valid values for the parameter `listAspectNames`, for example,
    are returned by this function.

    Returns:
        listAvailableAudioAspects: The list of aspect names registered in `audioAspects`.
    """
    return sorted(audioAspects.keys())

cacheAudioAnalyzers = cachetools.LRUCache(maxsize=256)
