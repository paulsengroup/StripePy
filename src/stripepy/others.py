# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import os
from typing import Optional

import hictkpy
import structlog


def _raise_invalid_bin_type_except(f: hictkpy.File):
    raise RuntimeError(f"Only files with a uniform bin size are supported, found \"{f.attributes()['bin-type']}\".")


def open_matrix_file_checked(path: os.PathLike, resolution: int) -> hictkpy.File:
    logger = structlog.get_logger()
    logger.info('validating file "%s" (%dbp)...', path, resolution)

    try:
        if not isinstance(resolution, int):
            raise TypeError("resolution must be an integer.")

        if resolution <= 0:
            raise ValueError("resolution must be greater than zero.")

        if hictkpy.is_scool_file(path):
            raise RuntimeError(".scool files are not currently supported.")
        if hictkpy.is_cooler(path):
            f = hictkpy.File(path)
            if f.resolution() == 0:
                _raise_invalid_bin_type_except(f)
            if f.resolution() != resolution:
                raise RuntimeError(f"expected {resolution} resolution, found {f.resolution()}.")
        else:
            f = hictkpy.MultiResFile(path)[resolution]
    except RuntimeError as e:
        raise RuntimeError(f'error opening file "{path}"') from e

    if f.attributes().get("bin-type", "fixed") != "fixed":
        _raise_invalid_bin_type_except(f)
    logger.info('file "%s" successfully validated', path)

    return f


def define_RoI(location: Optional[str], chrom_size: int, resolution: int, window_size: int = 2_000_000):
    if location is None or window_size <= 0:
        return None

    assert chrom_size > 0
    assert resolution > 0

    if chrom_size > window_size:
        window_size = ((window_size + resolution - 1) // resolution) * resolution

    if location == "middle":
        e1 = max(0, ((chrom_size - window_size) // (2 * resolution)) * resolution)
        e2 = e1 + window_size
    elif location == "start":
        e1 = 0
        e2 = window_size
    else:
        raise NotImplementedError

    if e2 - e1 < window_size:
        e1 = 0
        e2 = window_size

    bounds = [e1, min(chrom_size, e2)]
    return {"genomic": bounds, "matrix": [x // resolution for x in bounds]}
