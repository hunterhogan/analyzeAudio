"""
in a registry
the registrar
the registrant
a register; not to register
by/during registration
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Any, Callable, Dict, List, Tuple, Union
import inspect
import librosa
import numpy
import os
import torch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics', message='.*fast=True.*')

audioAspects: Dict[str, Dict[str, Union[Callable[..., Any], Tuple]]] = {}

def registrationAudioAspect(aspectTarget: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator function for registering audio aspects.
    Args:
        aspectTarget (str): The target aspect to register.
    Returns:
        Callable: The decorator function.
    """
    def registrar(registrant: Callable[..., Any]) -> Callable[..., Any]:
        """
        Register a function as an audio aspect analyzer.
        Parameters:
            registrant (Callable): The function to be registered.
        Returns:
            Callable: The registered function.
        """
        audioAspects[aspectTarget] = {
            'analyzer': registrant, 
            'analyzerArguments': inspect.getfullargspec(registrant).args 
        }
        
        # If the function returns NDArray, create a mean variant of the function
        if issubclass(registrant.__annotations__.get('return'), numpy.ndarray):
            def registrationAspectMean(*args, **kwargs) -> float:
                return registrant(*args, **kwargs).mean()
            
            audioAspects[f"{aspectTarget} mean"] = {
                'analyzer': registrationAspectMean,
                'analyzerArguments': inspect.getfullargspec(registrant).args
            }

        return registrant
    return registrar

def analyzeAudio(pathFilename: str, listAspectsTarget: List[str]) -> List[float]:
    dictionaryAspectsAnalyzed = {aspectTarget: 'not found' for aspectTarget in listAspectsTarget}
   
    waveform, sampleRate = librosa.load(path=pathFilename, sr=None, mono=False)
    tryAgain = True
    while tryAgain:
        try:
            tensorAudio = torch.from_numpy(waveform) # memory-sharing
            tryAgain = False
        except RuntimeError as error:
            if 'negative stride' in str(error):
                waveform = waveform.copy()
                tryAgain = True
            else:
                raise error
    # need "lazy" loading
    spectrogram = librosa.stft(y=waveform)
    spectrogramMagnitude, DISCARDEDphase = librosa.magphase(D=spectrogram)
    spectrogramPower = numpy.absolute(spectrogram)**2

    spectrogramMagnitude = spectrogramMagnitude

    pytorchOnCPU = not torch.cuda.is_available()  # False if GPU available, True if not

    dictionaryAspectsAnalyzed = {
        audioFeature: 
        audioAspects[audioFeature]['analyzer'](*map(vars().get, audioAspects[audioFeature]['analyzerArguments']))
        for audioFeature in listAspectsTarget
    }

    return [dictionaryAspectsAnalyzed[aspectTarget] for aspectTarget in listAspectsTarget]

def analyzeListPathFilenamesAudio(listPathFilenames, listAspectsTarget):
    rowsListFilenameAspectValues: List[List[str|float]] = []
    with ThreadPoolExecutor() as concurrencyManager:
        dictionaryConcurrency = {concurrencyManager.submit(analyzeAudio, pathFilename, listAspectsTarget): pathFilename for pathFilename in listPathFilenames}
        for claimTicket in tqdm(as_completed(dictionaryConcurrency), total=len(listPathFilenames)):
            listValuesExtracted: List[float] = claimTicket.result()
            rowsListFilenameAspectValues.append([os.path.basename(dictionaryConcurrency[claimTicket])] + listValuesExtracted)
    return rowsListFilenameAspectValues

