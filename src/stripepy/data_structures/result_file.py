# Copyright (C) 2024 Andrea Raffo <andrea.raffo@ibv.uio.no>
#
# SPDX-License-Identifier: MIT

import datetime
import functools
import itertools
import json
import pathlib
import re
from importlib.metadata import version
from typing import Any, Dict, Optional, Sequence, Tuple

import h5py
import hictkpy
import numpy as np
import numpy.typing as npt
import pandas as pd

from stripepy.data_structures import Result, Stripe


class ResultFile(object):
    """
    A class used to read and write StripePy results to a HDF5 file.

    There are 3 main use cases:
        - Open the file in read mode:
            with ResultFile("results.hdf5") as h5:
                ...

        - Open file in write mode:
            - if all data will be written to the file before the file is closed:
                with ResultFile.create("results.hdf5", mode="w", ...) as h5:
                    h5.write_descriptors(res1)
                    h5.write_descriptors(res2)
                    ...

            - if the data will be added progressively:
                with ResultFile.create("results.hdf5", mode="a", ...) as h5:
                    h5.write_descriptors(res1)  # not mandatory, it is also possible to create the
                                                # file and close it immediately
                ...
                with ResultFile.append("results.hdf5") as h5:
                    h5.write_descriptors(res2)
                    h5.write_descriptors(res3)
                ...
                with ResultFile.append("results.hdf5") as h5:
                    h5.write_descriptors(res4)
                    h5.finalize()  # IMPORTANT!
                                   # Without the above line you'll get an error when trying to open
                                   # the file in read mode

    When opening or creating a ResultFile write or append mode, a context manager (e.g. with:) must be used

    Attributes
    ----------
    path: pathlib.Path
        the path to the opened file
    assembly: str
        the name of the reference genome assembly used to generate the file
    resolution: int
        the resolution of the Hi-C matrix used to generate the file
    creation_date: datetime.datetime
        the file creation date
    format: str
        the file format string
    format_url: str
        the URL where the file format is documented
    format_version: int
        the format version of the file currently opened
    generated_by: str
        the name of the tool used to generate the opened file
    normalization: Optional[str]
        the name of the normalization used to generate the data stored in the given file
    chromosomes: Dict[str, int]
        the chromosomes associated with the opened file

    """

    # This is just a private member used to ensure that files are initialized correctly when mode != "r"
    __create_key = object()

    def __init__(self, path: pathlib.Path, mode: str = "r", _create_key: Optional[object] = None):
        if mode != "r" and _create_key != ResultFile.__create_key:
            raise RuntimeError(
                "Please use ResultFile.create(), ResultFile.create_from_file(), or ResultFile.append() to open a file in write mode"
            )

        self._path = path

        if mode == "r":
            open = ResultFile._open_in_read_mode  # noqa
        elif mode == "a" or mode == "r+":
            open = functools.partial(ResultFile._open_in_append_mode, mode=mode)  # noqa
        elif mode == "w":
            open = ResultFile._open_in_write_mode  # noqa
        else:
            raise ValueError('mode should be "r", "w", or "a"')

        self._h5, self._mode, self._version = open(path)
        self._chroms = self._read_chroms()
        self._chroms_idx = {chrom: i for i, chrom in enumerate(self._chroms)}
        self._attrs = dict(self._h5.attrs)  # noqa

    @staticmethod
    def create(
        path: pathlib.Path,
        mode: str,
        chroms: Dict[str, int],
        resolution: int,
        normalization: Optional[str] = None,
        assembly: str = "unknown",
        metadata: Optional[Dict[str, Any]] = None,
        compression_lvl: int = 9,
    ):
        """
        Create a ResultFile using the provided information.
        """

        if normalization is None:
            normalization = "NONE"

        if metadata is None:
            metadata = {}

        f = ResultFile(path, mode, _create_key=ResultFile.__create_key)
        f._init_attributes(resolution, assembly, normalization, metadata)
        f._init_chromosomes(chroms)
        f._init_index()
        f._init_min_persistence()
        f._init_points(compression_lvl)
        f._init_pseudodistribution(compression_lvl)
        f._init_stripes(compression_lvl)
        f._chroms_idx = {chrom: i for i, chrom in enumerate(f._chroms)}

        return f

    @staticmethod
    def create_from_file(
        path: pathlib.Path,
        mode: str,
        matrix_file: hictkpy.File,
        normalization: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compression_lvl: int = 9,
    ):
        """
        Create a ResultFile using information from the given matrix file.
        """
        return ResultFile.create(
            path,
            mode,
            matrix_file.chromosomes(include_ALL=False),
            matrix_file.resolution(),
            normalization,
            matrix_file.attributes().get("assembly", "unknown"),
            metadata,
            compression_lvl,
        )

    @staticmethod
    def append(path: pathlib.Path):
        """
        Append to an existing ResultFile.
        IMPORTANT: the file must have been created with create() or create_from_file() with mode="a"
        """
        return ResultFile(path, mode="a", _create_key=ResultFile.__create_key)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._mode != "r":
            self._finalize()
        self._h5.close()

    def __getitem__(self, chrom: str) -> Result:
        if chrom not in self._chroms:
            raise KeyError(f'chromosome "{chrom}" not found')

        res = Result(chrom, self._chroms[chrom])
        res.set_min_persistence(self.get_min_persistence(chrom))
        for location in ("LT", "UT"):
            res.set(
                "all_minimum_points", self.get(chrom, "all_minimum_points", location)["all_minimum_points"], location
            )
            res.set(
                "all_maximum_points", self.get(chrom, "all_maximum_points", location)["all_maximum_points"], location
            )
            res.set(
                "persistence_of_all_minimum_points",
                self.get(chrom, "persistence_of_all_minimum_points", location)["persistence_of_all_minimum_points"],
                location,
            )
            res.set(
                "persistence_of_all_maximum_points",
                self.get(chrom, "persistence_of_all_maximum_points", location)["persistence_of_all_maximum_points"],
                location,
            )
            res.set(
                "pseudodistribution", self.get(chrom, "pseudodistribution", location)["pseudodistribution"], location
            )

            df = self.get(chrom, "stripes", location)

            if len(df) == 0:
                continue

            stripes = []

            cols = [
                "seed",
                "left_bound",
                "right_bound",
                "top_bound",
                "bottom_bound",
                "top_persistence",
                "inner_mean",
                "inner_std",
                "outer_lsum",
                "outer_lsize",
                "outer_rsum",
                "outer_rsize",
                "outer_lmean",
                "outer_rmean",
                "outer_mean",
                "quartile",
            ]

            # Default-initialize missing columns
            for col in cols:
                if col not in df:
                    if col == "quartile":
                        df[col] = pd.Series(list(np.full(5, np.nan)))
                    elif re.match(r"^outer_[lr]size", col):
                        df[col] = 0
                    else:
                        df[col] = np.nan

            # Ensure column order is deterministic
            df = df[cols]

            if location == "LT":
                location_ = "lower_triangular"
            else:
                location_ = "upper_triangular"

            for (
                seed,
                left_bound,
                right_bound,
                top_bound,
                bottom_bound,
                top_persistence,
                inner_mean,
                inner_std,
                outer_lsum,
                outer_lsize,
                outer_rsum,
                outer_rsize,
                outer_lmean,
                outer_rmean,
                outer_mean,
                quartile,
            ) in df[cols].itertuples(index=False):
                s = Stripe(
                    seed=seed,
                    top_pers=top_persistence,
                    horizontal_bounds=(left_bound, right_bound),
                    vertical_bounds=(top_bound, bottom_bound),
                    where=location_,
                )

                if np.isnan(outer_lmean):
                    # Unfortunately in v1 files we only have the aggregated outer_mean,
                    # so this is the best we can do
                    assert np.isnan(outer_rmean)
                    assert not np.isnan(outer_mean)
                    outer_lmean = outer_mean
                    outer_rmean = outer_mean

                if np.isnan(outer_lsum):
                    # Estimate outer_lsum and outer_lsize
                    assert not np.isnan(outer_lmean)
                    outer_lsize = (bottom_bound - top_bound + 1) * 3
                    outer_lsum = outer_lsize * outer_lmean

                if np.isnan(outer_rsum):
                    # Estimate outer_rsum and outer_rsize
                    assert not np.isnan(outer_rmean)
                    outer_rsize = (bottom_bound - top_bound + 1) * 3
                    outer_rsum = outer_rsize * outer_rmean

                s.set_biodescriptors(
                    inner_mean=inner_mean,
                    inner_std=inner_std,
                    outer_lsum=outer_lsum,
                    outer_lsize=outer_lsize,
                    outer_rsum=outer_rsum,
                    outer_rsize=outer_rsize,
                    five_number=quartile,
                )
                stripes.append(s)

            res.set("stripes", stripes, location)

        return res

    def finalize(self):
        """
        Finalize a file opened in append mode.
        """
        self._finalize(abs(self._version))

    @property
    def path(self) -> pathlib.Path:
        return self._path

    @functools.cached_property
    def assembly(self) -> str:
        return self._h5.attrs["assembly"]

    @functools.cached_property
    def resolution(self) -> int:
        return int(self._h5.attrs["bin-size"])

    @functools.cached_property
    def creation_date(self) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self._h5.attrs["creation-date"])

    @functools.cached_property
    def format(self) -> str:
        return self._h5.attrs["format"]

    @functools.cached_property
    def format_url(self) -> str:
        return self._h5.attrs["format-url"]

    @functools.cached_property
    def format_version(self) -> int:
        return abs(self._h5.attrs["format-version"])

    @functools.cached_property
    def generated_by(self) -> str:
        return self._h5.attrs["generated-by"]

    @functools.cached_property
    def metadata(self) -> Dict[str, Any]:
        return json.loads(self._h5.attrs["metadata"])

    @functools.cached_property
    def normalization(self) -> Optional[str]:
        norm = self._h5.attrs["normalization"]
        if norm == "NONE":
            return None
        return norm

    @property
    def chromosomes(self) -> Dict[str, int]:
        return self._chroms

    def get_min_persistence(self, chrom: str) -> float:
        """
        Get the minimum persistence associated with the given chromosome.

        Parameters
        ----------
        chrom: str
            chromosome name

        Returns
        -------
            the minimum persistence
        """
        if chrom not in self._chroms:
            raise KeyError(f'File "{self.path}" does not have data for chromosome "{chrom}"')

        if self._version == 1:
            return self._get_min_persistence_v1(chrom)
        return self._get_min_persistence_v2(chrom)

    def get(self, chrom: Optional[str], field: str, location: str) -> pd.DataFrame:
        """
        Get the data associated with the given chromosome, field, and location.

        Parameters
        ----------
        chrom: Optional[str]
            chromosome name.
            when not provided, return data for the entire genome.
        field: str
            name of the field to be fetched.
            Supported names:
                * pseudodistribution
                * all_minimum_points
                * persistence_of_all_minimum_points
                * all_maximum_points
                * persistence_of_all_maximum_points
                * geo_descriptors
                * bio_descriptors
        location: str
            location of the attribute to be registered. Should be "LT" or "UT"

        Returns
        -------
            the data associated with the given chromosome, field, and location
        """
        if chrom is None:
            dfs = []
            for chrom in self._chroms:
                df = self.get(chrom, field, location)
                df["chrom"] = chrom
                dfs.append(df)

            df = pd.concat(dfs).reset_index(drop=True)
            df["chrom"] = df["chrom"].astype("category")
            cols = df.columns.tolist()
            cols.insert(0, cols.pop(cols.index("chrom")))

            return df[cols]

        if chrom not in self._chroms:
            raise KeyError(f'File "{self.path}" does not have data for chromosome "{chrom}"')

        if location not in {"LT", "UT"}:
            raise ValueError("Location should be UT or LT")

        if self._version == 1:
            return self._get_v1(chrom, field, location)
        if self._version == 2:
            return self._get_v2(chrom, field, location)
        return self._get_v3(chrom, field, location)

    def write_descriptors(self, result: Result):
        """
        Read the descriptors from the given Result object and write them to the opened file.

        Parameters
        ----------
        result: Result
            results to be added to the opened file
        """
        chrom = result.chrom[0]
        self._write_min_persistence(chrom, result.min_persistence)

        for location in ("LT", "UT"):
            self._append_pseudodistribution(chrom, result.get("pseudodistribution", location), location)
            self._append_persistence(
                chrom,
                result.get("all_minimum_points", location),
                result.get("persistence_of_all_minimum_points", location),
                "min",
                location,
            )
            self._append_persistence(
                chrom,
                result.get("all_maximum_points", location),
                result.get("persistence_of_all_maximum_points", location),
                "max",
                location,
            )
            self._append_stripes(chrom, result.get("stripes", location), location)

    def _get_v1(self, chrom: str, field: str, location: str) -> pd.DataFrame:
        mappings = {
            "pseudodistribution": f"/{chrom}/global-pseudo-distributions/{location}/pseudo-distribution",
            "all_minimum_points": f"/{chrom}/global-pseudo-distributions/{location}/minima_pts_and_persistence",
            "persistence_of_all_minimum_points": f"/{chrom}/global-pseudo-distributions/{location}/minima_pts_and_persistence",
            "all_maximum_points": f"/{chrom}/global-pseudo-distributions/{location}/maxima_pts_and_persistence",
            "persistence_of_all_maximum_points": f"/{chrom}/global-pseudo-distributions/{location}/maxima_pts_and_persistence",
            "geo_descriptors": f"/{chrom}/stripes/{location}/geo-descriptors",
            "bio_descriptors": f"/{chrom}/stripes/{location}/bio-descriptors",
            "stripes": None,
        }

        if field not in mappings:
            raise KeyError(f"Unknown field \"{field}\". Valid fields are {', '.join(mappings.keys())}")

        if field == "stripes":
            df1 = self._get_v1(chrom, "geo_descriptors", location)
            df2 = self._get_v1(chrom, "bio_descriptors", location)
            return pd.concat((df1, df2), axis="columns")

        path = mappings[field]

        if field not in {"geo_descriptors", "bio_descriptors"}:
            data = self._h5[path][:]
            if field.startswith("persistence"):
                data = data[1, :]
            elif field.endswith("points"):
                data = data[0, :].astype(int)
            return pd.DataFrame({field: data})

        df = pd.DataFrame(data=self._h5[path], columns=self._h5[path].attrs["col_names"])

        if field == "geo_descriptors":
            df = df.rename(
                columns={
                    "seed persistence": "top_persistence",
                    "L-boundary": "left_bound",
                    "R_boundary": "right_bound",
                    "U-boundary": "top_bound",
                    "D-boundary": "bottom_bound",
                }
            )
            for col in ("seed", "left_bound", "right_bound", "top_bound", "bottom_bound"):
                df[col] = df[col].astype(int)
            return df

        return df.rename(
            columns={
                "inner mean": "inner_mean",
                "outer mean": "outer_mean",
                "relative change": "rel_change",
                "standard deviation": "inner_std",
            }
        )

    @staticmethod
    def _open_in_read_mode(path: pathlib.Path) -> Tuple[h5py.File, str, int]:
        h5 = h5py.File(path, "r")
        ResultFile._validate(h5)

        return h5, "r", h5.attrs["format-version"]

    @staticmethod
    def _open_in_append_mode(path: pathlib.Path, mode: str) -> Tuple[h5py.File, str, int]:
        new_file = not path.is_file()
        h5 = h5py.File(path, mode)
        if not new_file:
            ResultFile._validate(h5, skip_index_validation=True, skip_version_check=True)

        # Negative version numbers mark files that have not yet been finalized.
        # A version value of -3 marks a v3 file that is being constructed
        version = h5.attrs.get("format-version", -3)  # noqa
        if version >= 0:
            raise RuntimeError("cannot append to a file that has already been finalized!")
        if version != -3:
            raise RuntimeError("can only append to v3 files")

        return h5, mode, version

    @staticmethod
    def _open_in_write_mode(path: pathlib.Path) -> Tuple[h5py.File, str, int]:
        return h5py.File(path, "w-"), "w", 3

    @staticmethod
    def _validate_index(h5: h5py.File):
        num_chroms = len(h5["/chroms/name"])
        for prefix, suffix in itertools.product(("/index/lt", "/index/ut"), h5["/index/lt"]):
            dset_name = f"{prefix}/{suffix}"
            dset_length = len(h5[dset_name])
            if dset_length != num_chroms + 1:
                raise RuntimeError(
                    f'malformed index "{dset_name}": expected {num_chroms + 1} entries, found {dset_length}'
                )

            data = h5[dset_name][:]
            data = data[np.isfinite(data)]
            if len(data) > 1 and not np.all(data[:-1] <= data[1:]):
                raise RuntimeError(f'malformed index "{dset_name}": index entries are not sorted in ascending order"')

    @staticmethod
    def _validate(h5: h5py.File, skip_version_check: bool = False, skip_index_validation: bool = False):
        """
        Perform a basic sanity check on the metadata of the current file
        """
        format = h5.attrs.get("format")  # noqa
        format_version = h5.attrs.get("format-version")
        try:
            if format is None:
                raise RuntimeError('attribute "format" is missing')

            if format_version is None:
                raise RuntimeError('attribute "format-version" is missing')

            if format != "HDF5::StripePy":
                raise RuntimeError(f'unrecognized file format: expected "HDF5::StripePy", found "{format}"')

            known_versions = {1, 2, 3}
            if not skip_version_check and format_version not in known_versions:
                known_versions_ = ", ".join(str(v) for v in known_versions)
                raise RuntimeError(
                    f'unsupported file format version "{format_version}". At present only versions {known_versions_} are supported'
                )

            if not skip_index_validation and abs(format_version) > 1:
                ResultFile._validate_index(h5)

        except RuntimeError as e:
            raise RuntimeError(
                f'failed to validate input file "{h5.filename}": file is corrupt or was not generated by StripePy.'
            ) from e

    def _read_chroms(self) -> Dict[str, int]:
        if not "/chroms/name" in self._h5:
            return {}

        return {
            chrom.decode("utf-8"): size for chrom, size in zip(self._h5["/chroms/name"], self._h5["/chroms/length"])
        }

    @functools.cache
    def _get_min_persistence_v1(self, chrom: str) -> float:
        return float(self._h5[f"/{chrom}/global-pseudo-distributions"].attrs["min_persistence_used"])

    @functools.cache
    def _read_min_persistence_values(self) -> Dict[str, float]:
        values = self._h5["/min_persistence"][:]
        if len(values) != len(self._chroms):
            raise RuntimeError(
                f"min_persistence vector has an unexpected length: expected {len(self._chroms)}, found {len(values)}"
            )
        return {chrom: float(pers) for chrom, pers in zip(self._chroms, values)}

    def _get_min_persistence_v2(self, chrom: str) -> float:
        return self._read_min_persistence_values()[chrom]

    @functools.cache
    def _read_index(self, name: str, location: str) -> npt.NDArray[int]:
        return self._h5[f"/index/{location.lower()}/{name}"][:].astype(int)

    @functools.cache
    def _index_chromosomes(self) -> Dict[str, int]:
        return {chrom: i for i, chrom in enumerate(self._chroms)}

    def _get_v2(self, chrom: str, field: str, location: str) -> pd.DataFrame:
        known_fields = {
            "pseudodistribution",
            "all_minimum_points",
            "persistence_of_all_minimum_points",
            "all_maximum_points",
            "persistence_of_all_maximum_points",
            "geo_descriptors",
            "bio_descriptors",
            "stripes",
        }

        if field not in known_fields:
            raise KeyError(f"Unknown field \"{field}\". Valid fields are {', '.join(known_fields)}")

        if field == "stripes":
            df1 = self._get_v2(chrom, "geo_descriptors", location)
            df2 = self._get_v2(chrom, "bio_descriptors", location)
            return pd.concat((df1, df2), axis="columns")

        chrom_id = self._index_chromosomes()[chrom]

        if field == "pseudodistribution":
            i1, i2 = self._read_index("chrom_offsets_pd", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data=self._h5[f"/pseudodistribution/{location.lower()}"][i1:i2].astype(float),
                columns=["pseudodistribution"],
            )

        if field == "all_minimum_points":
            i1, i2 = self._read_index("chrom_offsets_min_points", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data=self._h5[f"/min_points/{location.lower()}/points"][i1:i2].astype(int),
                columns=["all_minimum_points"],
            )

        if field == "persistence_of_all_minimum_points":
            i1, i2 = self._read_index("chrom_offsets_min_points", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data=self._h5[f"/min_points/{location.lower()}/persistence"][i1:i2].astype(float),
                columns=["persistence_of_all_minimum_points"],
            )

        if field == "all_maximum_points":
            i1, i2 = self._read_index("chrom_offsets_max_points", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data=self._h5[f"/max_points/{location.lower()}/points"][i1:i2].astype(int),
                columns=["all_maximum_points"],
            )

        if field == "persistence_of_all_maximum_points":
            i1, i2 = self._read_index("chrom_offsets_max_points", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data=self._h5[f"/max_points/{location.lower()}/persistence"][i1:i2].astype(float),
                columns=["persistence_of_all_maximum_points"],
            )

        if field == "geo_descriptors":
            i1, i2 = self._read_index("chrom_offsets_stripes", location)[chrom_id : chrom_id + 2]
            return pd.DataFrame(
                data={
                    "seed": self._h5[f"/stripes/{location.lower()}/seed"][i1:i2].astype(int),
                    "top_persistence": self._h5[f"/stripes/{location.lower()}/top_persistence"][i1:i2].astype(float),
                    "left_bound": self._h5[f"/stripes/{location.lower()}/left_bound"][i1:i2].astype(int),
                    "right_bound": self._h5[f"/stripes/{location.lower()}/right_bound"][i1:i2].astype(int),
                    "top_bound": self._h5[f"/stripes/{location.lower()}/top_bound"][i1:i2].astype(int),
                    "bottom_bound": self._h5[f"/stripes/{location.lower()}/bottom_bound"][i1:i2].astype(int),
                }
            )

        if field == "bio_descriptors":
            i1, i2 = self._read_index("chrom_offsets_stripes", location)[chrom_id : chrom_id + 2]
            outer_lmean = self._h5[f"/stripes/{location.lower()}/outer_lmean"][i1:i2].astype(float)
            outer_rmean = self._h5[f"/stripes/{location.lower()}/outer_rmean"][i1:i2].astype(float)
            quartile = self._h5[f"/stripes/{location.lower()}/quartile"][i1:i2, :].astype(float)

            assert quartile.shape[1] == 5

            df = pd.DataFrame(
                data={
                    "inner_mean": self._h5[f"/stripes/{location.lower()}/inner_mean"][i1:i2].astype(float),
                    "inner_std": self._h5[f"/stripes/{location.lower()}/inner_std"][i1:i2].astype(float),
                    "outer_lmean": outer_lmean,
                    "outer_rmean": outer_rmean,
                    "outer_mean": (outer_lmean + outer_rmean) / 2,
                    "min": quartile[:, 0],
                    "q1": quartile[:, 1],
                    "q2": quartile[:, 2],
                    "q3": quartile[:, 3],
                    "max": quartile[:, 4],
                }
            )

            df.loc[outer_lmean == -1, "outer_mean"] = -1

            df["rel_change"] = np.abs(df["inner_mean"] - df["outer_mean"]) / df["outer_mean"] * 100
            df.loc[df["outer_mean"] <= 0, "rel_change"] = -1.0

            return df

        raise NotImplementedError

    def _get_v3(self, chrom: str, field: str, location: str) -> pd.DataFrame:
        if field not in {"bio_descriptors", "stripes"}:
            # v3 and v2 layouts are different only for bio_descriptors
            return self._get_v2(chrom, field, location)

        if field == "stripes":
            df1 = self._get_v3(chrom, "geo_descriptors", location)
            df2 = self._get_v3(chrom, "bio_descriptors", location)
            return pd.concat((df1, df2), axis="columns")

        chrom_id = self._index_chromosomes()[chrom]

        assert field == "bio_descriptors"
        i1, i2 = self._read_index("chrom_offsets_stripes", location)[chrom_id : chrom_id + 2]
        inner_mean = self._h5[f"/stripes/{location.lower()}/inner_mean"][i1:i2].astype(float)
        inner_std = self._h5[f"/stripes/{location.lower()}/inner_std"][i1:i2].astype(float)
        outer_lsum = self._h5[f"/stripes/{location.lower()}/outer_lsum"][i1:i2].astype(float)
        outer_lsize = self._h5[f"/stripes/{location.lower()}/outer_lsize"][i1:i2].astype(int)
        outer_rsum = self._h5[f"/stripes/{location.lower()}/outer_rsum"][i1:i2].astype(float)
        outer_rsize = self._h5[f"/stripes/{location.lower()}/outer_rsize"][i1:i2].astype(int)
        quartile = self._h5[f"/stripes/{location.lower()}/quartile"][i1:i2, :].astype(float)

        assert quartile.shape[1] == 5

        df = pd.DataFrame(
            data={
                "inner_mean": inner_mean,
                "inner_std": inner_std,
                "outer_lsum": outer_lsum,
                "outer_lsize": outer_lsize,
                "outer_rsum": outer_rsum,
                "outer_rsize": outer_rsize,
                "min": quartile[:, 0],
                "q1": quartile[:, 1],
                "q2": quartile[:, 2],
                "q3": quartile[:, 3],
                "max": quartile[:, 4],
            }
        )

        df["outer_lmean"] = df["outer_lsum"] / df["outer_lsize"]
        df["outer_rmean"] = df["outer_rsum"] / df["outer_rsize"]
        df["outer_mean"] = (outer_lsum + outer_rsum) / (outer_lsize + outer_rsize)
        df["rel_change"] = np.abs(df["inner_mean"] - df["outer_mean"]) / df["outer_mean"] * 100

        return df

    def _init_attributes(self, resolution: int, assembly: str, normalization: Optional[str], metadata: Dict[str, Any]):
        if normalization is None:
            normalization = "NONE"

        assert resolution > 0

        self._h5.attrs["assembly"] = assembly
        self._h5.attrs["bin-size"] = resolution
        self._h5.attrs["creation-date"] = datetime.datetime.now().isoformat()
        self._h5.attrs["format"] = "HDF5::StripePy"
        self._h5.attrs["format-url"] = "https://github.com/paulsengroup/StripePy"
        self._h5.attrs["format-version"] = self._version
        self._h5.attrs["generated-by"] = f"StripePy v{version('stripepy-hic')}"
        self._h5.attrs["metadata"] = json.dumps(metadata, sort_keys=True, indent=2)
        self._h5.attrs["normalization"] = normalization

    def _init_chromosomes(self, chroms: Dict[str, int]):
        assert len(chroms) > 0

        self._chroms = chroms
        self._h5.create_group("/chroms")
        self._h5.create_dataset("/chroms/name", data=list(self._chroms.keys()))
        self._h5.create_dataset("/chroms/length", data=list(self._chroms.values()))

    def _init_index(self):
        templates = [
            "/index/{{location}}/chrom_offsets_min_points",
            "/index/{{location}}/chrom_offsets_max_points",
            "/index/{{location}}/chrom_offsets_stripes",
        ]

        data = np.zeros(len(self._chroms) + 1, dtype=int)
        for pattern, location in itertools.product(templates, ("lt", "ut")):
            self._h5.create_dataset(
                name=pattern.replace("{{location}}", location),
                data=data,
            )

        self._h5.create_dataset(
            name="/index/lt/chrom_offsets_pd",
            data=data,
        )

        self._h5["/index/ut/chrom_offsets_pd"] = h5py.SoftLink("/index/lt/chrom_offsets_pd")

    def _init_min_persistence(self):
        self._h5.create_dataset(name="/min_persistence", data=np.full(len(self._chroms), np.nan, dtype=float))

    def _init_pseudodistribution(self, compression_lvl: int):
        resolution = self.resolution
        maxshape = sum((length + resolution - 1) // resolution for length in self._chroms.values())
        self._h5.create_dataset(
            name="/pseudodistribution/lt",
            dtype=float,
            shape=(0,),
            maxshape=(maxshape,),
            compression="gzip",
            compression_opts=compression_lvl,
            shuffle=True,
        )
        self._h5.create_dataset(
            name="/pseudodistribution/ut",
            dtype=float,
            shape=(0,),
            maxshape=(maxshape,),
            compression="gzip",
            compression_opts=compression_lvl,
            shuffle=True,
        )

    def _init_points(self, compression_lvl: int):
        params = {
            "compression": "gzip",
            "compression_opts": compression_lvl,
            "shuffle": True,
            "chunks": True,
            "maxshape": (None,),
            "shape": (0,),
        }

        templates = {
            "/min_points/{{location}}/points": params | {"dtype": int},
            "/min_points/{{location}}/persistence": params | {"dtype": float},
            "/max_points/{{location}}/points": params | {"dtype": int},
            "/max_points/{{location}}/persistence": params | {"dtype": float},
        }

        for pattern, params in templates.items():
            dsets = {pattern.replace("{{location}}", "lt"), pattern.replace("{{location}}", "ut")}
            for name in dsets:
                self._h5.create_dataset(name=name, **params)

    def _init_stripes(self, compression_lvl: int):
        params = {
            "compression": "gzip",
            "compression_opts": compression_lvl,
            "shuffle": True,
            "chunks": True,
            "maxshape": (None,),
            "shape": (0,),
        }

        templates = {
            "/stripes/{{location}}/seed": params | {"dtype": int},
            "/stripes/{{location}}/left_bound": params | {"dtype": int},
            "/stripes/{{location}}/right_bound": params | {"dtype": int},
            "/stripes/{{location}}/top_bound": params | {"dtype": int},
            "/stripes/{{location}}/bottom_bound": params | {"dtype": int},
            "/stripes/{{location}}/top_persistence": params | {"dtype": float},
            "/stripes/{{location}}/inner_mean": params | {"dtype": float},
            "/stripes/{{location}}/inner_std": params | {"dtype": float},
            "/stripes/{{location}}/outer_lsum": params | {"dtype": float},
            "/stripes/{{location}}/outer_lsize": params | {"dtype": int},
            "/stripes/{{location}}/outer_rsum": params | {"dtype": float},
            "/stripes/{{location}}/outer_rsize": params | {"dtype": int},
            "/stripes/{{location}}/quartile": params | {"dtype": float, "shape": (0, 0), "maxshape": (None, 5)},
        }

        for pattern, params in templates.items():
            dsets = {pattern.replace("{{location}}", "lt"), pattern.replace("{{location}}", "ut")}
            for name in dsets:
                self._h5.create_dataset(name=name, **params)

    def _append_to_dset(self, path: str, data):
        if len(data) == 0:
            return

        dset = self._h5[path]

        shape = list(dset.maxshape)
        shape[0] = dset.shape[0] + len(data)

        dset.resize(shape)
        dset[-len(data) :] = data

    def _update_index(self, chrom: str, name: str, offset: int, location: str):
        idx = self._chroms_idx[chrom] + 1
        dset_name = f"/index/{location}/{name}"
        self._h5[dset_name][idx:] = offset

    def _write_min_persistence(self, chrom: str, pers: float):
        idx = self._chroms_idx[chrom]
        self._h5["/min_persistence"][idx] = pers

    def _append_pseudodistribution(self, chrom: str, data: Sequence[float], location: str):
        location = location.lower()
        assert location in {"lt", "ut"}

        self._append_to_dset(f"/pseudodistribution/{location}", data)
        self._update_index(
            chrom,
            name="chrom_offsets_pd",
            offset=len(self._h5[f"/pseudodistribution/{location}"]),
            location=location,
        )

    def _append_persistence(
        self, chrom: str, points: Sequence[int], persistence: Sequence[float], kind: str, location: str
    ):
        location = location.lower()
        assert len(points) == len(persistence)
        assert kind in {"min", "max"}
        assert location in {"lt", "ut"}

        self._append_to_dset(f"{kind}_points/{location}/points", points)
        self._append_to_dset(f"{kind}_points/{location}/persistence", persistence)
        self._update_index(
            chrom,
            name=f"chrom_offsets_{kind}_points",
            offset=len(self._h5[f"{kind}_points/{location}/points"]),
            location=location,
        )

    def _append_stripes(self, chrom: str, stripes: Sequence[Stripe], location: str):
        location = location.lower()
        assert location in {"lt", "ut"}
        seeds = np.empty_like(stripes, dtype=int)
        left_bounds = np.empty_like(stripes, dtype=int)
        right_bounds = np.empty_like(stripes, dtype=int)
        top_bounds = np.empty_like(stripes, dtype=int)
        bottom_bounds = np.empty_like(stripes, dtype=int)
        top_persistence_values = np.empty_like(stripes, dtype=float)
        inner_means = np.empty_like(stripes, dtype=float)
        inner_stds = np.empty_like(stripes, dtype=float)
        outer_lsums = np.empty_like(stripes, dtype=float)
        outer_lsizes = np.empty_like(stripes, dtype=int)
        outer_rsums = np.empty_like(stripes, dtype=float)
        outer_rsizes = np.empty_like(stripes, dtype=int)
        quartiles = np.empty((len(stripes), 5), dtype=float)

        for i, s in enumerate(stripes):
            seeds[i] = s.seed
            left_bounds[i] = s.left_bound
            right_bounds[i] = s.right_bound
            top_bounds[i] = s.top_bound
            bottom_bounds[i] = s.bottom_bound
            top_persistence_values[i] = s.top_persistence
            inner_means[i] = s.inner_mean
            inner_stds[i] = s.inner_std
            outer_lsums[i] = s.outer_lsum
            outer_lsizes[i] = s.outer_lsize
            outer_rsums[i] = s.outer_rsum
            outer_rsizes[i] = s.outer_rsize
            quartiles[i] = s.five_number

        self._append_to_dset(f"/stripes/{location}/seed", seeds)
        self._append_to_dset(f"/stripes/{location}/left_bound", left_bounds)
        self._append_to_dset(f"/stripes/{location}/right_bound", right_bounds)
        self._append_to_dset(f"/stripes/{location}/top_bound", top_bounds)
        self._append_to_dset(f"/stripes/{location}/bottom_bound", bottom_bounds)
        self._append_to_dset(f"/stripes/{location}/top_persistence", top_persistence_values)
        self._append_to_dset(f"/stripes/{location}/inner_mean", inner_means)
        self._append_to_dset(f"/stripes/{location}/inner_std", inner_stds)
        self._append_to_dset(f"/stripes/{location}/outer_lsum", outer_lsums)
        self._append_to_dset(f"/stripes/{location}/outer_lsize", outer_lsizes)
        self._append_to_dset(f"/stripes/{location}/outer_rsum", outer_rsums)
        self._append_to_dset(f"/stripes/{location}/outer_rsize", outer_rsizes)
        self._append_to_dset(f"/stripes/{location}/quartile", quartiles)

        self._update_index(
            chrom,
            name="chrom_offsets_stripes",
            offset=len(self._h5[f"/stripes/{location}/seed"]),
            location=location,
        )

    def _finalize(self, format_version: Optional[int] = None):
        if self._version > 0 and self._mode != "w":
            raise RuntimeError("file cannot be finalized: file was not opened in append mode")

        if format_version is not None:
            self._version = format_version
        self._h5.attrs["format"] = "HDF5::StripePy"
        self._h5.attrs["format-url"] = "https://github.com/paulsengroup/StripePy"
        self._h5.attrs["format-version"] = self._version
        self._h5.attrs["generated-by"] = f"StripePy v{version('stripepy-hic')}"

        if "/chroms/name" not in self._h5:
            raise RuntimeError("file has not been yet initialized")

        skip_version_check = self._version < 0
        self._validate(self._h5, skip_version_check=skip_version_check)
        self._mode = "r"
        self._h5.close()
        self._h5 = h5py.File(self._path, "r")
