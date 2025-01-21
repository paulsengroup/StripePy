# Copyright (C) 2025 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT


import pathlib
from typing import Sequence

from pandas.testing import assert_frame_equal

from stripepy.IO import ResultFile


def matplotlib_avail() -> bool:
    try:
        import matplotlib
    except ImportError:
        return False

    return True


def _compare_attributes(f1: ResultFile, f2: ResultFile):
    assert f1.assembly == f2.assembly
    assert f1.resolution == f2.resolution
    assert f1.format == f2.format
    assert f1.normalization == f2.normalization
    assert f1.chromosomes == f2.chromosomes

    metadata1 = f1.metadata
    metadata2 = f2.metadata

    metadata1.pop("min-chromosome-size")
    metadata2.pop("min-chromosome-size")

    assert metadata1 == metadata2


def _compare_field(f1: ResultFile, f2: ResultFile, chrom: str, field: str, location: str):
    assert chrom in f1.chromosomes
    df1 = f1.get(chrom, field, location)
    df2 = f2.get(chrom, field, location)

    assert_frame_equal(df1, df2, check_exact=False)


def _compare_result(f1: ResultFile, f2: ResultFile, chrom: str, location: str):
    fields = (
        "pseudodistribution",
        "all_minimum_points",
        "persistence_of_all_minimum_points",
        "all_maximum_points",
        "persistence_of_all_maximum_points",
        "stripes",
    )

    for field in fields:
        _compare_field(f1, f2, chrom, field, location)


def compare_result_files(reference: pathlib.Path, found: pathlib.Path, chroms: Sequence[str]):
    with ResultFile(reference) as f1, ResultFile(found) as f2:
        _compare_attributes(f1, f2)
        for chrom in chroms:
            _compare_result(f1, f2, chrom, "LT")
            _compare_result(f1, f2, chrom, "UT")
