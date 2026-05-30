from __future__ import annotations

from typing import Any, ParamSpec, TYPE_CHECKING, TypeVar
from typing_extensions import TypedDict

if TYPE_CHECKING:
	from collections.abc import Callable

parameterSpecifications = ParamSpec('parameterSpecifications')
typeReturned = TypeVar('typeReturned')

class analyzersAudioAspects(TypedDict):
	analyzer: Callable[..., Any]
	analyzerParameters: list[str]
