# ruff: noqa: D103
"""Analyzers that use the tensor to analyze audio data."""
from __future__ import annotations

from analyzeAudio import registrationAudioAspect
from torchmetrics.functional.audio.srmr import speech_reverberation_modulation_energy_ratio
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
	import torch

def analyzeSRMR(tensorAudio: torch.Tensor, sampleRate: int, *, pytorchOnCPU: bool | None, **keywordArguments: Any) -> torch.Tensor:
	keywordArguments['fast'] = keywordArguments.get('fast') or pytorchOnCPU or None
	return speech_reverberation_modulation_energy_ratio(tensorAudio, sampleRate, **keywordArguments)

@registrationAudioAspect('SRMR mean')
def analyzeSRMRMean(tensorAudio: torch.Tensor, sampleRate: int, pytorchOnCPU: bool | None, **keywordArguments: Any) -> float:  # noqa: FBT001
	return float(analyzeSRMR(tensorAudio, sampleRate, pytorchOnCPU=pytorchOnCPU, **keywordArguments).mean().item())
