# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import warnings
from typing import Optional

import hictkpy
import pytest

from stripepy.main import main

from .common import (
    check_results_are_empty,
    compare_result_files,
    get_avail_cpu_cores,
    matplotlib_avail,
)

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
    def _run_stripepy_call(tmpdir, nproc: int = 1, min_chrom_size: Optional[int] = None, with_roi: bool = False):
        assert nproc > 0
        tmpdir = pathlib.Path(tmpdir)
        testfile = testdir / "data" / "4DNFI9GMP2J8.mcool"
        result_file = testdir / "data" / "results_4DNFI9GMP2J8_v2.hdf5"
        resolution = 10_000

        chrom_sizes = hictkpy.MultiResFile(testfile).chromosomes()

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
            "--nproc",
            str(nproc),
        ]

        if min_chrom_size is not None:
            assert min_chrom_size >= 0
            args.extend(("--min-chrom-size", str(min_chrom_size)))

        if with_roi:
            args.extend(("--roi", "middle", "--plot-dir", str(plot_dir)))

        if with_roi:
            if not matplotlib_avail():
                with pytest.raises(ImportError):
                    main(args)
                pytest.skip("matplotlib not available")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                main(args)
        else:
            main(args)

        assert output_file.is_file()
        assert log_file.is_file()

        chroms_processed = []
        chroms_skipped = []

        for chrom, size in chrom_sizes.items():
            if size > min_chrom_size:
                chroms_processed.append(chrom)
            else:
                chroms_skipped.append(chrom)

        compare_result_files(result_file, output_file, chroms_processed)
        check_results_are_empty(output_file, chroms_skipped)

        if with_roi:
            assert plot_dir.is_dir()

            for chrom in chroms_processed:
                assert (plot_dir / chrom).is_dir()

            for chrom in chroms_skipped:
                assert not (plot_dir / chrom).exists()

    @staticmethod
    @pytest.mark.skipif(
        get_avail_cpu_cores() < 2, reason="host does not support multiprocessing or has a single CPU core"
    )
    def test_stripepy_call_parallel(tmpdir):
        TestStripePyCall._run_stripepy_call(
            tmpdir,
            min_chrom_size=int(150e6),
            nproc=min(8, get_avail_cpu_cores()),
            with_roi=False,
        )

    @staticmethod
    @pytest.mark.skipif(
        get_avail_cpu_cores() < 2, reason="host does not support multiprocessing or has a single CPU core"
    )
    def test_stripepy_call_with_roi_parallel(tmpdir):
        TestStripePyCall._run_stripepy_call(
            tmpdir,
            min_chrom_size=int(150e6),
            nproc=min(8, get_avail_cpu_cores()),
            with_roi=True,
        )

    @staticmethod
    def test_stripepy_call(tmpdir):
        TestStripePyCall._run_stripepy_call(
            tmpdir,
            min_chrom_size=int(200e6),
            nproc=1,
            with_roi=False,
        )

    @staticmethod
    def test_stripepy_call_with_roi(tmpdir):
        TestStripePyCall._run_stripepy_call(
            tmpdir,
            min_chrom_size=int(200e6),
            nproc=1,
            with_roi=True,
        )
