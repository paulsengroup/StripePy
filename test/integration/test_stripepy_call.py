# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import warnings

import hictkpy
import pytest

from stripepy import main

from .common import compare_result_files, matplotlib_avail

testdir = pathlib.Path(__file__).resolve().parent.parent


@pytest.mark.end2end
class TestStripePyCall:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "4DNFI9GMP2J8.mcool",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_stripepy_call(tmpdir):
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        result_file = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()
        chrom_size_cutoff = sum(chrom_sizes.values()) // len(chrom_sizes)

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--output-folder",
            str(tmpdir),
            "--min-chrom-size",
            str(chrom_size_cutoff),
        ]
        main(args)

        outfile = pathlib.Path(tmpdir) / testfile.stem / str(resolution) / "results.hdf5"

        assert outfile.is_file()
        compare_result_files(
            result_file, outfile, [chrom for chrom, size in chrom_sizes.items() if size >= chrom_size_cutoff]
        )

    @staticmethod
    def test_stripepy_call_with_roi(tmpdir):
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        result_file = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()
        chrom_size_cutoff = max(chrom_sizes.values()) - 1

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--output-folder",
            str(tmpdir),
            "--min-chrom-size",
            str(chrom_size_cutoff),
            "--roi",
            "middle",
        ]
        if not matplotlib_avail():
            with pytest.raises(ImportError):
                main(args)
            pytest.skip("matplotlib not available")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            main(args)

        outfile = pathlib.Path(tmpdir) / testfile.stem / str(resolution) / "results.hdf5"

        assert outfile.is_file()
        compare_result_files(result_file, outfile, [tuple(chrom_sizes.keys())[0]])
