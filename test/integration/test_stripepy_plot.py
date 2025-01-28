# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import shutil
import tarfile

import pytest

from stripepy.main import main

from .common import matplotlib_avail

testdir = pathlib.Path(__file__).resolve().parent.parent


def compare_images(expected: pathlib.Path, actual: pathlib.Path, tol: float = 0.1):
    import matplotlib.testing.compare

    res = matplotlib.testing.compare.compare_images(str(expected), str(actual), tol)
    if res is not None:
        pytest.fail(res)


def extract_image(
    name: str, dest_dir: pathlib.Path, prefix: pathlib.Path = pathlib.Path("stripepy-plot-test-images")
) -> pathlib.Path:
    dest = dest_dir / name
    with tarfile.TarFile.open(testdir / "data" / "stripepy-plot-test-images.tar.xz") as tar:
        with tar.extractfile((prefix / name).as_posix()) as fin, dest.open("wb") as fout:
            shutil.copyfileobj(fin, fout)  # noqa

    return dest_dir / name


def run_main(args):
    if matplotlib_avail():
        main(args)
        return

    with pytest.raises(ImportError):
        main(args)

    pytest.skip("matplotlib not available")


@pytest.mark.end2end
class TestStripePyPlot:
    @staticmethod
    def setup_class():
        test_files = [
            testdir / "data" / "4DNFI9GMP2J8.mcool",
            testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5",
            testdir / "data" / "stripepy-plot-test-images.tar.xz",
        ]

        for f in test_files:
            if not f.exists():
                raise RuntimeError(
                    f'unable to find file "{f}". Did you download the test files prior to running pytest?'
                )

    @staticmethod
    def test_contact_map(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        expected = extract_image("contact_map.png", tmpdir)

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "contact-map", str(matrix_file), str(resolution), str(outfile), "--region", region]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_seeds(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("contact_map_with_seeds.png", tmpdir)

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-seeds",
            "--region",
            region,
        ]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_stripes(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("contact_map_with_stripes.png", tmpdir)

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-stripes",
            "--region",
            region,
        ]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_contact_map_with_stripes_no_heights(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        matrix_file = testdir / "data" / "4DNFI9GMP2J8.mcool"
        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("contact_map_with_stripes_no_heights.png", tmpdir)

        resolution = 10_000
        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = [
            "plot",
            "contact-map",
            str(matrix_file),
            str(resolution),
            str(outfile),
            "--stripepy-hdf5",
            str(stripe_file),
            "--highlight-stripes",
            "--ignore-stripe-heights",
            "--region",
            region,
        ]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_pseudodistribution(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("pseudodistribution.png", tmpdir)

        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "pseudodistribution", str(stripe_file), str(outfile), "--region", region]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_stripe_hist(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("stripe_hist.png", tmpdir)

        region = "chr2:120100000-122100000"

        outfile = tmpdir / "img.png"
        args = ["plot", "stripe-hist", str(stripe_file), str(outfile), "--region", region]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)

    @staticmethod
    def test_stripe_hist_gw(tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        stripe_file = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        expected = extract_image("stripe_hist_gw.png", tmpdir)

        outfile = tmpdir / "img.png"
        args = ["plot", "stripe-hist", str(stripe_file), str(outfile)]
        run_main(args)

        assert outfile.is_file()
        compare_images(expected, outfile)
