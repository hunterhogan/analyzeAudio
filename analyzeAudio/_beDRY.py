from __future__ import annotations

from operator import neg
import math

def KValue(unscaled: float, K: float = 10.0) -> float:
	unscaled = max(min(unscaled, K), neg(K) + 1e-6)
	return 100.0 * math.log1p(unscaled + K) / math.log1p(2 * K)

# TODO create DRY `return 20 * numpy.log10(arrayRMS, where=(arrayRMS != 0), out=None)`
# If possible, one function for in-place and copy variants.
