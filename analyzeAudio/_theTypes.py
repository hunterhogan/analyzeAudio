from __future__ import annotations

from numpy import complexfloating, dtype, floating, ndarray
from typing import Any, Literal, NamedTuple, ParamSpec, TYPE_CHECKING, TypeAlias, TypedDict, TypeVar

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import ArrayLike, DTypeLike

parameterSpecifications = ParamSpec('parameterSpecifications')
typeReturned = TypeVar('typeReturned')

class analyzersAudioAspects(TypedDict):
	analyzer: Callable[..., Any]
	analyzerParameters: list[str]

class BleedFullArray(NamedTuple):
	arrayBleed: ndarray[tuple[int, int, int], dtype[floating[Any]]]
	arrayFull: ndarray[tuple[int, int, int], dtype[floating[Any]]]

class BleedFull(NamedTuple):
	bleed: float
	full: float

class ParametersMelSpectrogram(TypedDict, total=False):
	dtype: DTypeLike
	fmax: float | None
	fmin: float
	hop_length: int
	htk: bool
	n_fft: int
	n_mels: int
	norm: float | Literal['slaney'] | None
	power: float
	win_length: int
	window: str | tuple[Any, ...] | float | Callable[[int], ndarray] | ArrayLike

Audio: TypeAlias = ndarray[tuple[int, ...], dtype[floating[Any]]]
libturd: TypeAlias = ndarray[tuple[int, ...], dtype[Any]]
Spectrogram: TypeAlias = ndarray[tuple[int, int, int], dtype[complexfloating[Any, Any]]]
SpectrogramMagnitude: TypeAlias = ndarray[tuple[int, int, int], dtype[floating[Any]]]
SpectrogramPower: TypeAlias = ndarray[tuple[int, int, int], dtype[floating[Any]]]
