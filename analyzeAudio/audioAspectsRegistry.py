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
from pathlib import PurePath
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, TypedDict
import inspect
import librosa
import numpy
import torch
import warnings
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics', message='.*fast=True.*')
class analyzersAudioAspects(TypedDict):
    analyzer: Callable[..., Any]
    analyzerParameters: list[str]

audioAspects: Dict[str, analyzersAudioAspects] = {}
"""A register of 1) measurable aspects of audio data, 2) analyzer functions to measure audio aspects, 3) and parameters of analyzer functions."""

def registrationAudioAspect(aspectName: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    A function to "decorate" a registrant-analyzer function and the aspect of audio data it can analyze.

    Parameters:
        aspectName (str): The audio aspect that the registrar will enter into the register, `audioAspects`.
    """

    def registrar(registrant: Callable[..., Any]) -> Callable[..., Any]:
        """
        `registrar` updates the registry, `audioAspects`, with 1) the analyzer function, `registrant`, 2) the analyzer function's parameters, and 3) the aspect of audio data that the analyzer function measures.

        Parameters:
            registrant (Callable[..., Any]): The function that analyzes an aspect of audio data.

        Note:
            `registrar` does not change the behavior of `registrant`, the analyzer function.
        """
        audioAspects[aspectName] = {
            'analyzer': registrant,
            'analyzerParameters': inspect.getfullargspec(registrant).args
        }

        if isinstance(registrant.__annotations__.get('return', type(None)), type) and issubclass(registrant.__annotations__.get('return', type(None)), numpy.ndarray):
            def registrationAudioAspectMean(*arguments: Any, **keywordArguments: Any) -> float:
                """
                `registrar` updates the registry with a new analyzer function that calculates the mean of the analyzer's numpy.ndarray result.

                Returns:
                    float: Mean value of the analyzer's numpy.ndarray result.
                """
                return registrant(*arguments, **keywordArguments).mean()

            audioAspects[f"{aspectName} mean"] = {
                'analyzer': registrationAudioAspectMean,
                'analyzerParameters': inspect.getfullargspec(registrant).args
            }
        return registrant
    return registrar

def analyzeAudioFile(pathFilename: str, listAspectNames: List[str]) -> List[str | float | NDArray[Any]]:
    """
    Analyzes an audio file for specified aspects and returns the results.

    Parameters:
        pathFilename (str): The path to the audio file to be analyzed.
        listAspectNames (List[str]): A list of aspect names to analyze in the audio file.

    Returns:
        List[str | float | NDArray[Any]]: A list of analyzed values in the same order as `listAspectNames`.
    """
    dictionaryAspectsAnalyzed: Dict[str, str | float | NDArray[Any]] = {aspectName: 'not found' for aspectName in listAspectNames}
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
    print('analyzeAudioFile' , pathFilename, audioAspects['Abs_Peak_count']['analyzerParameters'])
    print(*map(locals().get, audioAspects['Abs_Peak_count']['analyzerParameters']))
    print(locals().get('pathFilename'))
    print(locals())
    dictionaryAspectsAnalyzed = {
        aspectName:
        audioAspects[aspectName]['analyzer'](*map(locals().get, audioAspects[aspectName]['analyzerParameters']))
        for aspectName in listAspectNames
    }

    return [dictionaryAspectsAnalyzed[aspectName] for aspectName in listAspectNames]

def analyzeAudioListPathFilenames(listPathFilenames: List[str], listAspectNames: List[str]) -> List[List[str | float | NDArray[Any]]]:
    """
    Analyzes a list of audio files for specified aspects of the individual files and returns the results.

    Parameters:
        listPathFilenames (List[str]): A list of paths to the audio files to be analyzed.
        listAspectNames (List[str]): A list of aspect names to analyze in each audio file.

    Returns:
        List[List[str | float]]: A list of lists, where each inner list contains the filename and analyzed values corresponding to the specified aspects.
    """
    rowsListFilenameAspectValues = []
    with ProcessPoolExecutor() as concurrencyManager:
        dictionaryConcurrency = {concurrencyManager.submit(analyzeAudioFile, pathFilename, listAspectNames)
                                 : pathFilename 
                                 for pathFilename in listPathFilenames}
        for claimTicket in tqdm(as_completed(dictionaryConcurrency), total=len(listPathFilenames)):
            listValuesExtracted = claimTicket.result()
            rowsListFilenameAspectValues.append(
                [str(PurePath(dictionaryConcurrency[claimTicket]).as_posix())] 
                + listValuesExtracted)
    return rowsListFilenameAspectValues

def getListAvailableAudioAspects() -> List[str]:
    """
    Returns a list of available audio aspects that can be analyzed.

    Returns:
        List[str]: A list of available audio aspects.
    """
    return sorted(list(audioAspects.keys()))
