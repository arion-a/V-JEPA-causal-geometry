"""Array type shared by the modules that accept float32 or float64 data and keep the dtype given."""

import numpy as np
from numpy.typing import NDArray

type Float = NDArray[np.float32 | np.float64]
