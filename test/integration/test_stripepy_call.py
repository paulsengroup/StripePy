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
            testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_stripepy_call(tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        result_file = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()
        chrom_size_cutoff = chrom_sizes["chr7"] - 1

        output_file = tmpdir / f"{testfile.stem}.hdf5"
        log_file = tmpdir / f"{testfile.stem}.log"

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--glob-pers-min",
            "0.05",
            "--loc-pers-min",
            "0.33",
            "--loc-trend-min",
            "0.25",
            "--output-file",
            str(output_file),
            "--log-file",
            str(log_file),
            "--min-chrom-size",
            str(chrom_size_cutoff),
        ]
        main(args)

        assert output_file.is_file()
        assert log_file.is_file()
        compare_result_files(
            result_file, output_file, [chrom for chrom, size in chrom_sizes.items() if size >= chrom_size_cutoff]
        )

    @staticmethod
    def test_stripepy_call_with_roi(tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        result_file = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()
        chrom_size_cutoff = chrom_sizes["chr1"] - 1

        output_file = tmpdir / f"{testfile.stem}.hdf5"
        log_file = tmpdir / f"{testfile.stem}.log"
        plot_dir = tmpdir / "plots"

        args = [
            "call",
            str(testfile),
            str(resolution),
            "--glob-pers-min",
            "0.05",
            "--loc-pers-min",
            "0.33",
            "--loc-trend-min",
            "0.25",
            "--output-file",
            str(output_file),
            "--log-file",
            str(log_file),
            "--plot-dir",
            str(plot_dir),
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

        assert output_file.is_file()
        assert log_file.is_file()
        assert plot_dir.is_dir()

        chroms = [chrom for chrom, size in chrom_sizes.items() if size >= chrom_size_cutoff]

        for chrom in chroms:
            assert (plot_dir / chrom).is_dir()

        compare_result_files(result_file, output_file, chroms)
