# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import sys
from typing import Optional

import numpy as np
import pandas as pd

from stripepy.data_structures import ResultFile


def run(
    h5_file: pathlib.Path,
    relative_change_threshold: float,
    with_biodescriptors: bool,
    with_header: bool,
    transform: Optional[str] = None,
    main_logger=None,
) -> int:
    try:
        with ResultFile(h5_file) as f:
            for chrom, size in f.chromosomes.items():
                _dump_stripes(
                    f,
                    chrom,
                    size,
                    f.resolution,
                    relative_change_threshold,
                    transform,
                    with_biodescriptors,
                    with_header,
                )
                with_header = False
    except BrokenPipeError:
        if sys.stdout.isatty():
            raise

    return 0


def _read_stripes(f: ResultFile, chrom: str) -> Optional[pd.DataFrame]:
    try:
        bio_lt = f.get(chrom, "bio_descriptors", "LT")
        bio_ut = f.get(chrom, "bio_descriptors", "UT")
        geo_lt = f.get(chrom, "geo_descriptors", "LT")
        geo_ut = f.get(chrom, "geo_descriptors", "UT")

        geo_lt["type"] = "lt"
        geo_ut["type"] = "ut"

        df1 = pd.concat([geo_lt, bio_lt], axis="columns")
        df2 = pd.concat([geo_ut, bio_ut], axis="columns")

        return pd.concat([df1, df2]).set_index("seed").sort_index(kind="stable")
    except Exception as e:
        missing_chrom_err_msg = "Unable to synchronously open object (component not found)"
        if f.format_version > 1 or missing_chrom_err_msg not in str(e):
            raise RuntimeError(f'failed to read stripes for chromosome "{chrom}": {e}')
        return None


def _stripes_to_bedpe(
    df: pd.DataFrame,
    chrom: str,
    size: int,
    resolution: int,
    transpose_policy: Optional[str],
    with_biodescriptors: bool,
) -> pd.DataFrame:
    num_stripes = len(df)

    start1_pos = np.minimum(df["left_bound"] * resolution, size)
    end1_pos = np.minimum(df["right_bound"] * resolution, size)
    start2_pos = np.minimum(df["top_bound"] * resolution, size)
    end2_pos = np.minimum(df["bottom_bound"] * resolution, size)

    if transpose_policy is not None:
        if transpose_policy == "transpose_to_ut":
            swap = df["type"] == "lt"
        elif transpose_policy == "transpose_to_lt":
            swap = df["type"] == "ut"
        else:
            raise NotImplementedError
    else:
        swap = None

    if swap is not None:
        start1_pos[swap], start2_pos[swap] = start2_pos[swap], start1_pos[swap]
        end1_pos[swap], end2_pos[swap] = end2_pos[swap], end1_pos[swap]

    dff = pd.DataFrame(
        {
            "chrom1": [chrom] * num_stripes,
            "start1": start1_pos,
            "end1": end1_pos,
            "chrom2": [chrom] * num_stripes,
            "start2": start2_pos,
            "end2": end2_pos,
        }
    )

    if with_biodescriptors:
        cols = ["left_bound", "right_bound", "top_bound", "bottom_bound", "type"]
        dff = pd.concat([dff, df.drop(columns=cols)], axis="columns")

    return dff


def _dump_stripes(
    f: ResultFile,
    chrom: str,
    size: int,
    resolution: int,
    cutoff: float,
    transpose_policy: str,
    with_biodescriptors: bool,
    with_header: bool,
):
    df = _read_stripes(f, chrom)
    if df is None:
        return

    df = df[df["rel_change"] >= cutoff]
    df = _stripes_to_bedpe(df, chrom, size, resolution, transpose_policy, with_biodescriptors)
    df.to_csv(sys.stdout, sep="\t", index=False, header=with_header)
