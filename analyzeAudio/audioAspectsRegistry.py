# noqa: D100
from __future__ import annotations

from multiprocessing import set_start_method as multiprocessing_set_start_method
from typing import TYPE_CHECKING
import contextlib
import inspect
import warnings

if TYPE_CHECKING:
	from analyzeAudio import analyzersAudioAspects, parameterSpecifications, typeReturned
	from collections.abc import Callable

with contextlib.suppress(RuntimeError):
	multiprocessing_set_start_method('spawn')

warnings.filterwarnings('ignore', category=UserWarning, module='torchmetrics', message='.*fast=True.*')

audioAspects: dict[str, analyzersAudioAspects] = {}
"""A register of 1) measurable aspects of audio data, 2) analyzer functions to measure audio aspects, 3) and parameters of analyzer functions."""

def registrationAudioAspect(aspectName: str) -> Callable[[Callable[parameterSpecifications, typeReturned]], Callable[parameterSpecifications, typeReturned]]:
	"""'Decorate' a registrant-analyzer function and the aspect of audio data it can analyze.

	Parameters
	----------
	aspectName : str
		The audio aspect that the registrar will enter into the register, `audioAspects`.

	"""  # noqa: DOC201

	def registrar(registrant: Callable[parameterSpecifications, typeReturned]) -> Callable[parameterSpecifications, typeReturned]:
		"""
		`registrar` updates the registry, `audioAspects`, with 1) the analyzer function, `registrant`, 2) the analyzer function's parameters, and 3) the aspect of audio data that the analyzer function measures.

		Parameters
		----------
		registrant : Callable
			The function that analyzes an aspect of audio data.

		Note
		----
		`registrar` does not change the behavior of `registrant`, the analyzer function.

		"""  # noqa: DOC201
		audioAspects[aspectName] = {'analyzer': registrant, 'analyzerParameters': inspect.getfullargspec(registrant).args}
		return registrant
	return registrar

def getListAvailableAudioAspects() -> list[str]:
	"""
	Return a sorted list of audio aspect names. All valid values for the parameter `listAspectNames`, for example, are returned by this function.

	Returns
	-------
	listAvailableAudioAspects : list of str
		The list of aspect names registered in `audioAspects`.

	"""
	return sorted(audioAspects.keys())
