from __future__ import annotations

from numpy import dtype, float64, floating, ndarray
from typing import Any, Literal, NamedTuple, ParamSpec, Protocol, TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
	from collections.abc import Callable
	from numpy.typing import ArrayLike, DTypeLike
	from torch import device, Tensor
	from typing import TypeAlias

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
	sr: int
	top_db: float | None
	win_length: int
	window: str | tuple[Any, ...] | float | Callable[[int], ndarray] | ArrayLike

Audio: TypeAlias = ndarray[tuple[Any, ...], dtype[floating[Any]]]
ArrayAspect: TypeAlias = ndarray[tuple[Any, ...], dtype[floating[Any]]]
SpectrogramMagnitude: TypeAlias = ndarray[tuple[int, int, int], dtype[floating[Any]]]
SpectrogramPower: TypeAlias = ndarray[tuple[int, int, int], dtype[floating[Any]]]
ArrayAspectSpectrogramFramewise: TypeAlias = ndarray[tuple[int, Literal[1], int], dtype[floating[Any]]]
ArrayAspectWaveformFramewise: TypeAlias = ndarray[tuple[Literal[1], int], dtype[floating[Any]]]

ArrayChannelData: TypeAlias = ndarray[tuple[Any, ...], dtype[float64]]
ArrayOverallData: TypeAlias = ndarray[tuple[int], dtype[float64]]

class AuralossChromaSTFTLoss(Protocol):
	fft_size: int
	window: Tensor
	device: device | None
	scale: str
	n_bins: int
	fb: Tensor

	def __call__(self, tensorInput: Tensor, tensorTarget: Tensor) -> Tensor: ...
