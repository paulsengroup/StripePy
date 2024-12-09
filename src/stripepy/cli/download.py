# Copyright (C) 2024 Roberto Rossini <roberros@uio.no>
#
# SPDX-License-Identifier: MIT

import functools
import hashlib
import json
import math
import pathlib
import random
import sys
import tempfile
import time
import urllib.request
from typing import Any, Dict, Tuple, Union

import structlog


@functools.cache
def _get_datasets(max_size: float) -> Dict[str, Dict[str, str]]:
    assert not math.isnan(max_size)

    datasets = {
        "4DNFI3RFZLZ5": {
            "url": "https://zenodo.org/records/14283922/files/4DNFI3RFZLZ5.stripepy.mcool?download=1",
            "md5": "f6e060211c95dd5fbf6e708c637d1c1c",
            "assembly": "mm10",
            "format": "mcool",
            "size_mb": 83.85,
        },
        "4DNFIC1CLPK7": {
            "url": "https://zenodo.org/records/14283922/files/4DNFI6HDY7WZ.stripepy.mcool?download=1",
            "md5": "745df902a842c17e535222fb7f9748ca",
            "assembly": "hg38",
            "format": "mcool",
            "size_mb": 104.73,
        },
        "4DNFI9GMP2J8": {
            "url": "https://zenodo.org/records/14283922/files/4DNFI9GMP2J8.stripepy.mcool?download=1",
            "md5": "a17d08460c03cf6c926e2ca5743e4888",
            "assembly": "hg38",
            "format": "mcool",
            "size_mb": 106.84,
        },
        "ENCFF993FGR": {
            "url": "https://zenodo.org/records/14283922/files/ENCFF993FGR.stripepy.hic?download=1",
            "md5": "3bcb8c8c5aac237f26f994e0f5e983d7",
            "assembly": "hg38",
            "format": "hic",
            "size_mb": 185.29,
        },
        "__results_v1": {
            "url": "https://zenodo.org/records/14283922/files/results_4DNFI9GMP2J8_v1.hdf5?download=1",
            "md5": "632b2a7a6e5c1a24dc3635710ed68a80",
            "assembly": "hg38",
            "format": "stripepy",
            "size_mb": 8.75,
        },
    }

    valid_dsets = {k: v for k, v in datasets.items() if v.get("size_mb", math.inf) < max_size}

    if len(valid_dsets) > 0:
        return valid_dsets

    raise RuntimeError(f"unable to find any dataset smaller than {max_size:.2f} MB")


def _list_datasets():
    dsets = {k: v for k, v in _get_datasets(math.inf).items() if not k.startswith("__")}
    json.dump(dsets, fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


def _get_random_dataset(max_size: float) -> Tuple[str, Dict[str, str]]:
    dsets = _get_datasets(max_size)
    assert len(dsets) > 0

    key = random.sample(list(dsets.keys()), 1)[0]
    return key, dsets[key]


def _lookup_dataset(name: Union[str, None], assembly: Union[str, None], max_size: float) -> Tuple[str, Dict[str, str]]:
    if name is not None:
        max_size = math.inf
        try:
            return name, _get_datasets(max_size)[name]
        except KeyError as e:
            raise RuntimeError(
                f'unable to find dataset "{name}". Please make sure the provided dataset is present in the list produced by stripepy download --list-only.'
            ) from e

    assert assembly is not None
    assert max_size >= 0

    dsets = {k: v for k, v in _get_datasets(max_size).items() if v["assembly"] == assembly}
    if len(dsets) == 0:
        raise RuntimeError(
            f'unable to find a dataset using "{assembly}" as reference genome. Please make sure such dataset exists in the list produced by stripepy download --list-only.'
        )

    key = random.sample(list(dsets.keys()), 1)[0]
    return key, dsets[key]


def _hash_file(path: pathlib.Path, chunk_size=16 << 20) -> str:
    logger = structlog.get_logger()
    logger.info('computing MD5 digest for file "%s"...', path)
    with path.open("rb") as f:
        hasher = hashlib.md5()
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return hasher.hexdigest()
            hasher.update(chunk)


def _download_progress_reporter(chunk_no, max_chunk_size, download_size):
    if download_size == -1:
        if not _download_progress_reporter.skip_progress_report:
            _download_progress_reporter.skip_progress_report = True
            logger = structlog.get_logger()
            logger.warning("unable to report download progress: remote file size is not known!")
        return

    timepoint = _download_progress_reporter.timepoint

    if time.time() - timepoint >= 15:
        mb_downloaded = (chunk_no * max_chunk_size) / (1024 << 10)
        download_size_mb = download_size / (1024 << 10)
        progress_pct = (mb_downloaded / download_size_mb) * 100
        logger = structlog.get_logger()
        logger.info("downloaded %.2f/%.2f MB (%.2f%%)", mb_downloaded, download_size_mb, progress_pct)
        _download_progress_reporter.timepoint = time.time()


# this is Python's way of defining static variables inside functions
_download_progress_reporter.skip_progress_report = False
_download_progress_reporter.timepoint = 0.0


def _download_and_checksum(name: str, dset: Dict[str, Any], dest: pathlib.Path):
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f"{dest.stem}.") as tmpfile:
        logger = structlog.get_logger()
        tmpfile.close()
        tmpfile = pathlib.Path(tmpfile.name)

        url = dset["url"]
        md5sum = dset["md5"]
        assembly = dset.get("assembly", "unknown")

        logger.info('downloading dataset "%s" (assembly=%s)...', name, assembly)
        t0 = time.time()
        urllib.request.urlretrieve(url, tmpfile, reporthook=_download_progress_reporter)
        t1 = time.time()
        logger.info('DONE! Downloading dataset "%s" took %.2fs.', name, t1 - t0)

        digest = _hash_file(tmpfile)
        if digest == md5sum:
            logger.info("MD5 checksum match!")
            return tmpfile.rename(dest)

        raise RuntimeError(
            f'MD5 checksum for file downloaded from "{url}" does not match: expected {md5sum}, found {digest}.'
        )


def run(
    name: Union[str, None],
    output_path: Union[pathlib.Path, None],
    assembly: Union[str, None],
    max_size: float,
    list_only: bool,
    force: bool,
):
    t0 = time.time()
    if list_only:
        _list_datasets()
        return

    do_random_sample = name is None and assembly is None

    if do_random_sample:
        dset_name, config = _get_random_dataset(max_size)
    else:
        dset_name, config = _lookup_dataset(name, assembly, max_size)

    if output_path is None:
        output_path = pathlib.Path(f"{dset_name}." + config["format"])

    if output_path.exists() and not force:
        raise RuntimeError(f"refusing to overwrite file {output_path}. Pass --force to overwrite.")
    output_path.unlink(missing_ok=True)

    dest = _download_and_checksum(dset_name, config, output_path)
    t1 = time.time()

    logger = structlog.get_logger()
    logger.info('successfully downloaded dataset "%s" to file "%s"', config["url"], dest)
    logger.info(f"file size: %.2fMB. Elapsed time: %.2fs", dest.stat().st_size / (1024 << 10), t1 - t0)
