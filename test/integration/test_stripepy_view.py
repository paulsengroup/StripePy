# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import contextlib
import io
import pathlib

import pandas as pd
import pytest

from stripepy import main

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyView:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_view():
        versions = ["1", "2"]
        for version in versions:
            testfile = testdir / "data" / f"results_4DNFI9GMP2J8_v{version}.hdf5"

            args = ["view", str(testfile)]
            buff = io.StringIO()
            with contextlib.redirect_stdout(buff):
                main(args)

            buff.seek(0)
            cols = ["chrom1", "start1", "end1", "chrom2", "start2", "end2"]
            df = pd.read_table(buff, names=cols)

            assert len(df.columns) == len(cols)
            assert len(df) == len(df[~df.isnull().values])
            assert len(df) == 18625
