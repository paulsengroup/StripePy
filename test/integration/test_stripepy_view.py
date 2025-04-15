# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import io
import pathlib
from typing import List

import pandas as pd
import pytest

from stripepy.main import main

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyView:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
            testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def _test_view(testfile: pathlib.Path, cols: List[str], num_rows: int):
        args = ["view", str(testfile), "--with-header", "--with-biodescriptors"]
        buff = io.StringIO()
        with contextlib.redirect_stdout(buff):
            main(args)

        buff.seek(0)

        df = pd.read_table(buff)

        null_mask = df[["chrom1", "start1", "end1", "chrom2", "start2", "end2"]].isnull().values

        assert df.columns.tolist() == cols
        assert len(df) == len(df[~null_mask])
        assert len(df) == num_rows

    @staticmethod
    def test_view_v1():
        cols = [
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            "top_persistence",
            "inner_mean",
            "outer_mean",
            "rel_change",
            "inner_std",
        ]
        TestStripePyView._test_view(
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            cols=cols,
            num_rows=18625,
        )

    @staticmethod
    def test_view_v2():
        cols = [
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            "top_persistence",
            "inner_mean",
            "inner_std",
            "outer_lmean",
            "outer_rmean",
            "outer_mean",
            "min",
            "q1",
            "q2",
            "q3",
            "max",
            "rel_change",
        ]
        TestStripePyView._test_view(
            testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
            cols=cols,
            num_rows=18625,
        )

    @staticmethod
    def test_view_v3():
        cols = [
            "chrom1",
            "start1",
            "end1",
            "chrom2",
            "start2",
            "end2",
            "top_persistence",
            "inner_mean",
            "inner_std",
            "outer_lsum",
            "outer_lsize",
            "outer_rsum",
            "outer_rsize",
            "min",
            "q1",
            "q2",
            "q3",
            "max",
            "outer_lmean",
            "outer_rmean",
            "outer_mean",
            "rel_change",
        ]
        TestStripePyView._test_view(
            testdir / "data" / "results_4DNFI9GMP2J8_v3.hdf5",
            cols=cols,
            num_rows=18379,
        )
