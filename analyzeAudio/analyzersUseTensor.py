from analyzeAudio import registrationAudioAspect
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import Any, Optional
import numpy
import torch

@registrationAudioAspect('SRMR')
def analyzeSRMR(tensorAudio: torch.Tensor, sampleRate: int, pytorchOnCPU: Optional[bool], **keywordArguments: Any) -> numpy.ndarray:
    keywordArguments['fast'] = keywordArguments.get('fast') or pytorchOnCPU or None
    return torch.Tensor.numpy(speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, **keywordArguments))
