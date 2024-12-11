# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import gc
import pathlib
from typing import Dict, Optional

import hictkpy as htk
import pandas as pd

from .data import generate_chromosomes


def generate_singleres_test_file(
    path: pathlib.Path, resolution: int, chromosomes: Optional[Dict[str, int]] = None
) -> pathlib.Path:
    if chromosomes is None:
        chromosomes = generate_chromosomes()

    writer = htk.cooler.FileWriter(str(path), resolution=resolution, chromosomes=chromosomes)
    writer.add_pixels(pd.DataFrame({"bin1_id": [0], "bin2_id": [0], "count": 1}))
    writer.finalize()

    del writer
    gc.collect()

    return pathlib.Path(path)
