# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import re
import sys
import warnings
from typing import Union

import numpy as np
import pandas as pd

from stripepy.IO import ResultFile


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for col in df.columns:
        new_cols.append(re.sub(r"\W+", "_", col.lower()))

    df.columns = new_cols
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.convert_dtypes()


def _read_stripes(f: ResultFile, chrom: str) -> pd.DataFrame:
    try:
        bio_lt = f.get(chrom, "bio_descriptors", "LT")[["rel_change"]]
        bio_ut = f.get(chrom, "bio_descriptors", "UT")[["rel_change"]]
        geo_lt = f.get(chrom, "geo_descriptors", "LT")
        geo_ut = f.get(chrom, "geo_descriptors", "UT")

        geo_lt["type"] = "lt"
        geo_ut["type"] = "ut"

        df1 = pd.concat([geo_lt, bio_lt], axis="columns")
        df2 = pd.concat([geo_ut, bio_ut], axis="columns")

        return pd.concat([df1, df2]).set_index("seed").sort_index()
    except Exception as e:
        raise RuntimeError(f'failed to read stripes for chromosome "{chrom}": {e}')


def _stripes_to_bedpe(
    df: pd.DataFrame, chrom: str, size: int, resolution: int, transpose_policy: Union[str, None]
) -> pd.DataFrame:
    num_stripes = len(df)

    start1_pos = df["left_bound"] * resolution
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

    return pd.DataFrame(
        {
            "chrom1": [chrom] * num_stripes,
            "start1": start1_pos,
            "end1": end1_pos,
            "chrom2": [chrom] * num_stripes,
            "start2": start2_pos,
            "end2": end2_pos,
        }
    )


def _dump_stripes(f: ResultFile, chrom: str, size: int, resolution: int, cutoff: float, transpose_policy: str):
    df = _read_stripes(f, chrom)
    df = df[df["rel_change"] >= cutoff]
    df = _stripes_to_bedpe(df, chrom, size, resolution, transpose_policy)

    df.to_csv(sys.stdout, sep="\t", index=False, header=False)


def run(
    h5_file: pathlib.Path,
    relative_change_threshold: float,
    transform: Union[str, None],
):
    with ResultFile(h5_file) as f:
        for chrom, size in f.chromosomes.items():
            _dump_stripes(f, chrom, size, f.resolution, relative_change_threshold, transform)
