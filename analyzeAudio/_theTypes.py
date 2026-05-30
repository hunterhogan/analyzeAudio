from __future__ import annotations

from typing import Any, ParamSpec, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar
import numpy

if TYPE_CHECKING:
	from collections.abc import Callable

parameterSpecifications = ParamSpec('parameterSpecifications')
typeReturned = TypeVar('typeReturned')

class analyzersAudioAspects(TypedDict):
	analyzer: Callable[..., Any]
	analyzerParameters: list[str]

Audio: TypeAlias = numpy.ndarray[tuple[int, ...], numpy.dtype[numpy.floating[Any]]]
libturd: TypeAlias = numpy.ndarray[tuple[int, ...], numpy.dtype[Any]]
Spectrogram: TypeAlias = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.complexfloating[Any, Any]]]
SpectrogramMagnitude: TypeAlias = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.floating[Any]]]
SpectrogramPower: TypeAlias = numpy.ndarray[tuple[int, int, int], numpy.dtype[numpy.floating[Any]]]
