import io

import numpy as np
import zstandard as zstd


def load_npy_to_tensor(path):
    with zstd.open(path, "rb") as fh:
        filedata = io.BytesIO(fh.readall())
        pool_npy = np.load(filedata)
        del filedata
    return pool_npy
