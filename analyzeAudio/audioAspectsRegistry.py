"""Register audio analyzers by measurable aspect name.

(AI generated docstring)

You can use this module to collect analyzer functions under human-readable audio aspect names.
The module exposes the shared registry, the decorator factory that populates the registry, and
the listing function that reports the registered aspect names.

Contents
--------
Variables
	audioAspects
		Store analyzer metadata by registered audio aspect name.

Functions
	getListAvailableAudioAspects
		Return the registered audio aspect names in sorted order.
	registrationAudioAspect
		Register one analyzer function under one audio aspect name.
"""

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
"""Store analyzer metadata by registered audio aspect name.

You can inspect `audioAspects` to retrieve the analyzer function and the ordered list of
parameter names for each registered audio aspect.
"""

def registrationAudioAspect(aspectName: str) -> Callable[[Callable[parameterSpecifications, typeReturned]], Callable[parameterSpecifications, typeReturned]]:
	"""Register one analyzer function under one audio aspect name.

	You can use this function as a decorator factory when an analyzer function should become
	discoverable through `audioAspects` [1]. The returned decorator stores the original analyzer
	function and the ordered parameter names reported by `inspect.getfullargspec()` [2] without
	changing the analyzer function behavior.

	Parameters
	----------
	aspectName : str
		The audio aspect name that the returned decorator will use as the registry key.

	Returns
	-------
	registrar : Callable[[Callable[parameterSpecifications, typeReturned]], Callable[parameterSpecifications, typeReturned]]
		A decorator that records one analyzer function and then returns the same analyzer function.

	Registration
	------------
	If `aspectName` already exists in the registry, the later registration replaces the earlier
	registry entry.

	Examples
	--------
	The analyzer modules use `registrationAudioAspect` in declarations such as the following.

	```python
	aspectName = 'Peak Signal-to-Noise Ratio mean'

	@registrationAudioAspect(aspectName)
	def getPSNRmean(pathFilenameAlfa: str | PathLike[Any], pathFilenameBeta: str | PathLike[Any]) -> float | None:
		return ...
	```

	References
	----------
	[1] `audioAspects`

	[2] Python standard library documentation for `inspect.getfullargspec`
		https://docs.python.org/3/library/inspect.html#inspect.getfullargspec

	"""

	def registrar(registrant: Callable[parameterSpecifications, typeReturned]) -> Callable[parameterSpecifications, typeReturned]:
		"""I use this nested function to record one analyzer function in the module registry.

		This function receives `registrant`, stores `registrant` and the ordered parameter names of
		`registrant`, and then returns `registrant` unchanged so the registered callable identity is
		preserved.

		Parameters
		----------
		registrant : Callable[parameterSpecifications, typeReturned]
			The analyzer function to register under the enclosing `aspectName`.

		Returns
		-------
		registrant : Callable[parameterSpecifications, typeReturned]
			The same analyzer function after the registry entry has been written.

		"""
		audioAspects[aspectName] = {'analyzer': registrant, 'analyzerParameters': inspect.getfullargspec(registrant).args}
		return registrant
	return registrar

def getListAvailableAudioAspects() -> list[str]:
	"""Return the registered audio aspect names in sorted order.

	You can use this function to inspect which audio aspect names are currently available in the
	shared registry. This function is useful when another function expects one or more registered
	audio aspect names such as `listAspectNames`.

	Returns
	-------
	listAvailableAudioAspects : list[str]
		The sorted list of registered audio aspect names.

	"""
	return sorted(audioAspects.keys())
