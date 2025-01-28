# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import os
from typing import Dict, Optional, Tuple

import hictkpy
import structlog


def _raise_invalid_bin_type_except(f: hictkpy.File):
    raise RuntimeError(f"Only files with a uniform bin size are supported, found \"{f.attributes()['bin-type']}\".")


def open_matrix_file_checked(path: os.PathLike, resolution: int, logger=None) -> hictkpy.File:
    """
    Open a file in one of the formats supported by hictkpy and check that it satisfies StripePy requirements.

    Parameters
    ----------
    path: os.PathLike
        path to the matrix file
    resolution: int
        resolution to be used to open the matrix file
    logger:
        logger

    Returns
    -------
    hictkpy.File
        the given file opened with hictkpy
    """
    if logger is None:
        logger = structlog.get_logger()

    logger.bind(step="IO")
    logger.info('validating file "%s" (%dbp)', path, resolution)

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


def define_region_of_interest(
    location: Optional[str],
    chrom_size: int,
    resolution: int,
    window_size: int = 2_000_000,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Define the region of interest with the desired properties.

    Parameters
    ----------
    location: Optional[str]
        location of the desired region.
        When provided, it should be either "start" or "middle".
    chrom_size: int
        the chromosome size in base-pairs.
    resolution: int
        resolution in base-pairs of the matrix to which the region refers to.
    window_size: int
        target width of the region of interest.

    Returns
    -------
    Optional[Dict[str, Tuple[int, int]]]
        return a dictionary with the coordinates of the region of interest.
        The dictionary has two keys:
        - genomic: two-element tuple with the genomic coordinates (bp) of the region of interest.
        - matrix: two-element tuple with the matrix coordinates of the region of interest.

        When location is None, return None.
    """
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

    bounds = (e1, min(chrom_size, e2))
    return {"genomic": bounds, "matrix": tuple(x // resolution for x in bounds)}
