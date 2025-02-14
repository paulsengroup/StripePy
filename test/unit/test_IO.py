# Copyright (C) 2024 Roberto Rossini <roberroso@uio.no>
#
# SPDX-License-Identifier: MIT

import itertools
import json
import math
import pathlib
from typing import Dict, Sequence

import hictkpy as htk
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from stripepy.IO import Result, ResultFile
from stripepy.utils.stripe import Stripe

from .common.cool import generate_singleres_test_file

testdir = pathlib.Path(__file__).resolve().parent.parent


def _directory_is_empty(path) -> bool:
    path = pathlib.Path(path)
    assert path.is_dir()
    return next(path.iterdir(), None) is None  # noqa


@pytest.mark.unit
class TestResult:
    def test_ctor(self):
        res = Result("chr1", 123)

        assert res.empty
        assert res.chrom[0] == "chr1"
        assert res.chrom[1] == 123
        assert res.roi is None

    def test_setters(self):
        res = Result("chr1", 123)

        res.set_min_persistence(1.0)
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_min_persistence(1.0)

        res.set_roi({"foobar": [1, 2, 3]})
        with pytest.raises(RuntimeError, match="has already been set"):
            res.set_roi({})

        assert res.roi is not None
        assert "foobar" in res.roi
        assert res.roi["foobar"] == [1, 2, 3]

        with pytest.raises(AttributeError, match='No attribute named "invalid name"'):
            res.set("invalid name", [], "LT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.set("all_minimum_points", [], "invalid location")

        res.set("all_minimum_points", [1, 2, 3], "LT")
        res.set("all_minimum_points", [4, 5, 6], "UT")

        with pytest.raises(RuntimeError, match="has already been set"):
            res.set("all_minimum_points", [], "LT")

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()
        assert (res.get("all_minimum_points", "UT") == [4, 5, 6]).all()

    def test_getters(self):
        res = Result("chr1", 123)

        res.set("all_minimum_points", [1, 2, 3], "LT")

        with pytest.raises(RuntimeError, match="Attribute .* is not set"):
            res.min_persistence  # noqa

        res.set_min_persistence(1.23)
        assert res.min_persistence == 1.23

        with pytest.raises(AttributeError, match='No attribute named "invalid name"'):
            res.get("invalid name", "LT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.get("all_minimum_points", "invalid location")

        with pytest.raises(RuntimeError, match='Attribute "all_minimum_points" for "UT" is not set'):
            res.get("all_minimum_points", "UT")

        with pytest.raises(RuntimeError, match='Attribute "all_maximum_points" for "LT" is not set'):
            res.get("all_maximum_points", "LT")

        assert len(res.get("stripes", "LT")) == 0

        assert (res.get("all_minimum_points", "LT") == [1, 2, 3]).all()

    def test_stripe_getters(self):
        res = Result("chr1", 123)

        stripes = [Stripe(10, 1.23, where="upper_triangular")]
        stripes[0].set_vertical_bounds(5, 10)
        stripes[0].set_horizontal_bounds(8, 12)
        stripes[0].compute_biodescriptors(csr_matrix((15, 15), dtype=float))

        with pytest.raises(AttributeError, match='does not have an attribute named "foobar"'):
            res.get_stripes_descriptor("foobar", "UT")

        with pytest.raises(ValueError, match="Location should be UT or LT"):
            res.get_stripes_descriptor("seed", "foobar")

        res.set("stripes", stripes, "LT")

        assert len(res.get_stripe_geo_descriptors("UT")) == 0
        assert len(res.get_stripe_bio_descriptors("UT")) == 0

        df = res.get_stripe_geo_descriptors("LT")
        assert df.columns.tolist() == [
            "seed",
            "top_persistence",
            "left_bound",
            "right_bound",
            "top_bound",
            "bottom_bound",
        ]
        assert len(df) == 1

        assert df["seed"].iloc[0] == 10
        assert np.isclose(df["top_persistence"].iloc[0], 1.23)
        assert df["left_bound"].iloc[0] == 8
        assert df["right_bound"].iloc[0] == 12
        assert df["top_bound"].iloc[0] == 5
        assert df["bottom_bound"].iloc[0] == 10

        df = res.get_stripe_bio_descriptors("LT")
        assert df.columns.tolist() == ["inner_mean", "outer_mean", "rel_change", "inner_std"]
        assert len(df) == 1

        assert df["inner_mean"].iloc[0] == 0
        assert df["outer_mean"].iloc[0] == 0
        assert np.isnan(df["rel_change"].iloc[0])
        assert df["inner_std"].iloc[0] == 0


@pytest.mark.unit
class TestResultFile:
    def test_ctor(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        path = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        with ResultFile(path) as f:
            assert f.path == path

        with pytest.raises(OSError):
            ResultFile(tmpdir / "asdf123.hdf5")

        path = pathlib.Path(tmpdir) / "test.hdf5"
        chroms = {"chr1": 10}
        with ResultFile.create(path, "w", chroms=chroms, resolution=1) as f:
            assert f.path == path

        with pytest.raises(OSError):
            ResultFile.create(path, "w", chroms=chroms, resolution=1)

        with pytest.raises(RuntimeError, match="cannot append to a file that has already been finalized!"):
            ResultFile.create(path, "a", chroms=chroms, resolution=1)

        with pytest.raises(RuntimeError, match="Please use .* to open a file in write mode"):
            ResultFile(path, "w")

    def test_properties(self):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        with ResultFile(path) as f:
            assert f.path == path
            assert f.assembly == "unknown"
            assert f.resolution == 10_000

            try:
                f.creation_date  # noqa
            except ValueError:
                pytest.fail("creation-date attribute is not a valid date")

            assert f.format == "HDF5::StripePy"
            assert f.format_url == "https://github.com/paulsengroup/StripePy"
            assert f.format_version == 1
            assert f.generated_by.startswith("StripePy")

            try:
                f.metadata  # noqa
            except json.decoder.JSONDecodeError:
                pytest.fail("metadata attribute is not a valid JSON")

            assert f.normalization is None

            assert len(f.chromosomes) == 24
            assert "chr1" in f.chromosomes

    def test_getters(self):
        path = testdir / "data" / "results_4DNFI9GMP2J8_v1.hdf5"
        with ResultFile(path) as f:
            assert math.isclose(f.get_min_persistence("chr1"), 0.05)
            with pytest.raises(KeyError):
                f.get_min_persistence("foobar")

            df = f.get("chr1", "pseudodistribution", "LT")

            assert "pseudodistribution" in df
            assert len(df) == 24896

            df = f.get("chr1", "geo_descriptors", "LT")

            geo_descriptor_cols = ["seed", "top_persistence", "left_bound", "right_bound", "top_bound", "bottom_bound"]
            assert len(df.columns) == len(geo_descriptor_cols)
            assert (df.columns == geo_descriptor_cols).all()
            assert len(df) == 1304

            df = f.get("chr1", "bio_descriptors", "LT")

            bio_descriptor_cols = ["inner_mean", "outer_mean", "rel_change", "inner_std"]
            assert len(df.columns) == len(bio_descriptor_cols)
            assert (df.columns == bio_descriptor_cols).all()
            assert len(df) == 1304

            with pytest.raises(KeyError):
                f.get("foobar", "pseudodistribution", "LT")

            with pytest.raises(KeyError):
                f.get("chr1", "foobar", "LT")

            with pytest.raises(ValueError):
                f.get("chr1", "pseudodistribution", "foobar")

    @staticmethod
    def _write_mock_result_to_file(
        f: ResultFile,
        chrom: str,
        resolution: int,
        points: Sequence[int],
        pseudodistribution: Sequence[float],
        persistence: Sequence[float],
    ):
        res = Result(chrom, resolution * len(points))
        res.set_min_persistence(1.23)

        for location in ("UT", "LT"):
            res.set("pseudodistribution", pseudodistribution, location)

            for key in (
                "all_minimum_points",
                "all_maximum_points",
                "persistent_minimum_points",
                "persistent_maximum_points",
            ):
                res.set(key, points, location)

            for key in (
                "persistence_of_all_minimum_points",
                "persistence_of_all_maximum_points",
                "persistence_of_minimum_points",
                "persistence_of_maximum_points",
            ):
                res.set(key, persistence, location)

        f.write_descriptors(res)

    @staticmethod
    def _compare_results(
        path: pathlib.Path,
        chroms: Dict[str, int],
        points: Sequence[int],
        pseudodistribution: Sequence[float],
        persistence: Sequence[float],
    ):

        with ResultFile(path) as f:
            assert f.chromosomes == chroms
            assert f.normalization == "weight"
            assert f.metadata.get("key", "missing") == "value"

            for chrom, location in itertools.product(chroms, ("UT", "LT")):
                assert np.isclose(f.get_min_persistence(chrom), 1.23)
                assert np.allclose(
                    f.get(chrom, "pseudodistribution", location)["pseudodistribution"], pseudodistribution
                )

                for key in ("all_minimum_points", "all_maximum_points"):
                    assert (f.get(chrom, key, location)[key] == points).all()

                for key in ("persistence_of_all_minimum_points", "persistence_of_all_maximum_points"):
                    assert np.allclose(f.get(chrom, key, location)[key], persistence)

    def test_file_creation(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        resolution = 10_000
        clr_file = generate_singleres_test_file(tmpdir / "test.cool", resolution)
        chroms = htk.File(clr_file).chromosomes()

        points = [1, 2, 3]
        persistence = [4.0, 5.0, 6.0]
        pseudodistribution = [7.0, 8.0, 9.0]

        # Create a mock ResultFile
        path = tmpdir / "results.hdf5"
        with ResultFile.create_from_file(
            path,
            "w",
            htk.File(clr_file),
            normalization="weight",
            metadata={"key": "value"},
        ) as f:
            for chrom in chroms:
                TestResultFile._write_mock_result_to_file(f, chrom, resolution, points, pseudodistribution, persistence)

        TestResultFile._compare_results(path, chroms, points, pseudodistribution, persistence)

    def test_progressive_file_creation(self, tmpdir):
        tmpdir = pathlib.Path(tmpdir)

        points = [1, 2, 3]
        persistence = [4.0, 5.0, 6.0]
        pseudodistribution = [7.0, 8.0, 9.0]

        resolution = 10_000
        clr_file = generate_singleres_test_file(
            tmpdir / "test.cool", resolution, {"chr1": len(points) * resolution, "chr2": len(points) * resolution}
        )
        chroms = htk.File(clr_file).chromosomes()

        path = tmpdir / "results.hdf5"
        with ResultFile.create(path, "a", chroms, resolution, normalization="weight", metadata={"key": "value"}) as f:
            pass

        for chrom in chroms:
            with ResultFile.append(path) as f:
                TestResultFile._write_mock_result_to_file(f, chrom, resolution, points, pseudodistribution, persistence)

        with ResultFile.append(path) as f:
            f.finalize()

        TestResultFile._compare_results(path, chroms, points, pseudodistribution, persistence)
