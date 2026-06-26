from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
	from numpy import dtype, floating, ndarray
	from typing import Any

class BleedFull(NamedTuple):
	bleed: float
	full: float

class BleedFullArray(NamedTuple):
	arrayBleed: ndarray[tuple[int, int, int], dtype[floating[Any]]]
	arrayFull: ndarray[tuple[int, int, int], dtype[floating[Any]]]
