from analyzeAudio import registrationAudioAspect
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import Any
import numpy
import torch

# TODO: can numpy 2.x take the mean of tensors?
@registrationAudioAspect('SRMR mean')
def analyzeSRMR(tensorAudio: torch.Tensor, sampleRate: int, pytorchOnCPU: bool=False, **kwargs: Any) -> float: #-> torch.Tensor:
    return numpy.mean(torch.Tensor.numpy(speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, fast=pytorchOnCPU, **kwargs)))

# I would like pytorchOnCPU to be an optional parameter
# I didn't pass pytorchOnCPU, but got
# ValueError: Expected argument `fast` to be a bool value
