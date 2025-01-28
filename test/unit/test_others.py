# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import pathlib
import tempfile

import pytest

from stripepy.others import define_region_of_interest, open_matrix_file_checked

from .common.cool import generate_singleres_test_file


@pytest.mark.unit
class TestOpenMatrixFileChecked:
    def test_invalid_paths(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        with pytest.raises(RuntimeError):
            open_matrix_file_checked(tmpdir / "foo", 5000)

    def test_invalid_formats(self, tmpdir):
        invalid_file = pathlib.Path(__file__).resolve()
        folder = pathlib.Path(tmpdir)
        with pytest.raises(RuntimeError):
            open_matrix_file_checked(invalid_file, 5000)

        with pytest.raises(RuntimeError):
            open_matrix_file_checked(folder, 5000)

    def test_invalid_resolutions(self, tmpdir):
        with tempfile.NamedTemporaryFile(dir=tmpdir) as clr:
            clr.close()
            path_to_clr = generate_singleres_test_file(pathlib.Path(clr.name), 1000)

            with pytest.raises(RuntimeError):
                open_matrix_file_checked(path_to_clr, 5000)
            with pytest.raises(TypeError, match="must be an integer"):
                open_matrix_file_checked(path_to_clr, 1000.0)
            with pytest.raises(ValueError, match="must be greater than zero"):
                open_matrix_file_checked(path_to_clr, -1)
            with pytest.raises(ValueError, match="must be greater than zero"):
                open_matrix_file_checked(path_to_clr, 0)

    def test_valid_files(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)
        chromosomes = {"A": 100_000, "B": 50_000, "C": 10_000}
        resolution = 1000
        path_to_clr = generate_singleres_test_file(tmpdir / "test.cool", resolution, chromosomes)

        f = open_matrix_file_checked(path_to_clr, resolution)
        assert f.resolution() == 1000

        with pytest.raises(Exception):
            open_matrix_file_checked(path_to_clr, resolution + 1)


@pytest.mark.unit
class TestDefineRegionOfInterest:
    def test_middle_large_chromosome(self):
        roi = define_region_of_interest(location="middle", chrom_size=1000, resolution=10, window_size=500)
        assert roi == {"genomic": (250, 750), "matrix": (25, 75)}

    def test_middle_small_chromosome(self):
        roi = define_region_of_interest(location="middle", chrom_size=123, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 123), "matrix": (0, 12)}

    def test_start_large_chromosome(self):
        roi = define_region_of_interest(location="start", chrom_size=1000, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 500), "matrix": (0, 50)}

    def test_start_small_chromosome(self):
        roi = define_region_of_interest(location="start", chrom_size=123, resolution=10, window_size=500)
        assert roi == {"genomic": (0, 123), "matrix": (0, 12)}

    def test_noop(self):
        assert define_region_of_interest(location=None, chrom_size=0, resolution=0) is None
        assert define_region_of_interest(location="middle", chrom_size=1000, resolution=10, window_size=0) is None

    def test_invalid_params(self):
        with pytest.raises(Exception):
            define_region_of_interest(location="middle", chrom_size=0, resolution=10, window_size=500)
            define_region_of_interest(location="middle", chrom_size=1000, resolution=0, window_size=500)

            define_region_of_interest(location="foo", chrom_size=1000, resolution=10, window_size=500)
