from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from collections.abc import Sequence
	from torch import Tensor

# TODO generalize more, improve, move to `Z0Z_tools` or `torch_einops_kit`, as appropriate.
def truncateTensors(listTensors: Sequence[Tensor]) -> tuple[Tensor, ...]:
	"""I use this to truncate the last axis of each `Tensor` to the shortest length.

	Parameters
	----------
	listTensors : Sequence[Tensor]
		Sequence of waveform tensors whose last axis stores samples.

	Returns
	-------
	tupleTensors : tuple[Tensor, ...]
		Tuple of tensors truncated to the minimum trailing sample length.
	"""
	truncate: int = min(tensor.shape[-1] for tensor in listTensors)
	return tuple(tensor[..., 0:truncate] for tensor in listTensors)
