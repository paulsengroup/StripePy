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
from typing import Any, Dict, Optional, Sequence, Tuple

import structlog

from stripepy.utils.common import pretty_format_elapsed_time
from stripepy.utils.progress_bar import initialize_progress_bar


@functools.cache
def _get_datasets(max_size: float, include_private: bool) -> Dict[str, Dict[str, str]]:
    assert not math.isnan(max_size)

    record_id = "14762415"

    datasets = {
        "4DNFI3RFZLZ5": {
            "url": f"https://zenodo.org/records/{record_id}/files/4DNFI3RFZLZ5.stripepy.mcool?download=1",
            "md5": "f6e060211c95dd5fbf6e708c637d1c1c",
            "assembly": "mm10",
            "format": "mcool",
            "size_mb": 83.85,
        },
        "4DNFIC1CLPK7": {
            "url": f"https://zenodo.org/records/{record_id}/files/4DNFI6HDY7WZ.stripepy.mcool?download=1",
            "md5": "745df902a842c17e535222fb7f9748ca",
            "assembly": "hg38",
            "format": "mcool",
            "size_mb": 104.73,
        },
        "4DNFI9GMP2J8": {
            "url": f"https://zenodo.org/records/{record_id}/files/4DNFI9GMP2J8.stripepy.mcool?download=1",
            "md5": "a17d08460c03cf6c926e2ca5743e4888",
            "assembly": "hg38",
            "format": "mcool",
            "size_mb": 106.84,
        },
        "ENCFF993FGR": {
            "url": f"https://zenodo.org/records/{record_id}/files/ENCFF993FGR.stripepy.hic?download=1",
            "md5": "3bcb8c8c5aac237f26f994e0f5e983d7",
            "assembly": "hg38",
            "format": "hic",
            "size_mb": 185.29,
        },
    }

    private_datasets = {
        "__results_v1": {
            "url": f"https://zenodo.org/records/{record_id}/files/results_4DNFI9GMP2J8_v1.hdf5?download=1",
            "md5": "03bca8d430191aaf3c90a4bc22a8c579",
            "filename": "results_4DNFI9GMP2J8_v1.hdf5",
            "assembly": "hg38",
            "format": "stripepy",
            "size_mb": 8.76,
        },
        "__results_v2": {
            "url": f"https://zenodo.org/records/{record_id}/files/results_4DNFI9GMP2J8_v2.hdf5?download=1",
            "md5": "dd14a2f69b337c40727d414d85e2f0a4",
            "filename": "results_4DNFI9GMP2J8_v2.hdf5",
            "assembly": "hg38",
            "format": "stripepy",
            "size_mb": 9.26,
        },
        "__stripepy_plot_images": {
            "url": f"https://zenodo.org/records/{record_id}/files/stripepy-plot-test-images.tar.xz?download=1",
            "md5": "adf60f386521f70b24936e53a6d11eab",
            "filename": "stripepy-plot-test-images.tar.xz",
            "assembly": "hg38",
            "format": "tar",
            "size_mb": 1.5,
        },
    }

    if include_private:
        datasets |= private_datasets

    valid_dsets = {k: v for k, v in datasets.items() if v.get("size_mb", math.inf) < max_size}

    if len(valid_dsets) > 0:
        return valid_dsets

    raise RuntimeError(f"unable to find any dataset smaller than {max_size:.2f} MB")


def _list_datasets():
    json.dump(_get_datasets(math.inf, include_private=False), fp=sys.stdout, indent=2)
    sys.stdout.write("\n")


def _get_random_dataset(max_size: float, include_private: bool) -> Tuple[str, Dict[str, str]]:
    dsets = _get_datasets(max_size, include_private)
    assert len(dsets) > 0

    key = random.sample(list(dsets.keys()), 1)[0]
    return key, dsets[key]


def _lookup_dataset(
    name: Optional[str], assembly: Optional[str], max_size: float, include_private: bool
) -> Tuple[str, Dict[str, str]]:
    if name is not None:
        max_size = math.inf
        try:
            return name, _get_datasets(max_size, include_private)[name]
        except KeyError as e:
            raise RuntimeError(
                f'unable to find dataset "{name}". Please make sure the provided dataset is present in the list produced by stripepy download --list-only.'
            ) from e

    assert assembly is not None
    assert max_size >= 0

    dsets = {k: v for k, v in _get_datasets(max_size, include_private).items() if v["assembly"] == assembly}
    if len(dsets) == 0:
        raise RuntimeError(
            f'unable to find a dataset using "{assembly}" as reference genome. Please make sure such dataset exists in the list produced by stripepy download --list-only.'
        )

    key = random.sample(list(dsets.keys()), 1)[0]
    return key, dsets[key]


def _hash_file(path: pathlib.Path, chunk_size=16 << 20) -> str:
    file_size = path.stat().st_size
    disable_bar = not sys.stderr.isatty() or file_size < (256 << 20)

    with initialize_progress_bar(
        total=file_size,
        disable=disable_bar,
        enrich_print=False,
        file=sys.stderr,
        receipt=False,
        monitor="{percent:.2%}",
        unit="B",
        scale="SI2",
    ) as bar:
        logger = structlog.get_logger()
        logger.info('computing MD5 digest for file "%s"...', path)
        with path.open("rb") as f:
            hasher = hashlib.md5()
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    return hasher.hexdigest()
                hasher.update(chunk)
                bar(len(chunk))


def _fetch_remote_file_size(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url)
        return int(urllib.request.urlopen(req).headers.get("Content-Length"))
    except Exception:  # noqa
        return None


def _download_progress_reporter(chunk_no, max_chunk_size, download_size):
    if _download_progress_reporter.bar is not None:
        _download_progress_reporter.bar(
            min(chunk_no * max_chunk_size, download_size) / _download_progress_reporter.total
        )


# this is Python's way of defining static variables inside functions
_download_progress_reporter.bar = None
_download_progress_reporter.total = None


def _download_and_checksum(name: str, dset: Dict[str, Any], dest: pathlib.Path):
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f"{dest.stem}.") as tmpfile:
        logger = structlog.get_logger()
        tmpfile.close()
        tmpfile = pathlib.Path(tmpfile.name)

        url = dset["url"]
        md5sum = dset["md5"]
        assembly = dset.get("assembly", "unknown")
        size = _fetch_remote_file_size(url)

        disable_bar = not sys.stderr.isatty() or size is None

        with initialize_progress_bar(
            total=size,
            manual=True,
            disable=disable_bar,
            enrich_print=False,
            file=sys.stderr,
            receipt=False,
            refresh_secs=0.05,
            monitor="{percent:.2%}",
            unit="B",
            scale="SI2",
        ) as bar:
            _download_progress_reporter.bar = bar
            _download_progress_reporter.total = size

            logger.info('downloading dataset "%s" (assembly=%s)...', name, assembly)
            t0 = time.time()
            urllib.request.urlretrieve(url, tmpfile, reporthook=_download_progress_reporter)
            logger.info('DONE! Downloading dataset "%s" took %s.', name, pretty_format_elapsed_time(t0))

        digest = _hash_file(tmpfile)
        if digest == md5sum:
            logger.info("MD5 checksum match!")
            return tmpfile.rename(dest)

        raise RuntimeError(
            f'MD5 checksum for file downloaded from "{url}" does not match: expected {md5sum}, found {digest}.'
        )


def _download_multiple(names: Sequence[str], output_paths: Sequence[pathlib.Path]):
    assert len(names) == len(output_paths)

    logger = structlog.get_logger()

    for name, output_path in zip(names, output_paths):
        t0 = time.time()
        dset_name, config = _lookup_dataset(name, None, math.inf, include_private=True)

        if output_path.exists():
            logger.info('found existing file "%s"', output_path)
            digest = _hash_file(output_path)
            if digest == config["md5"]:
                logger.info('dataset "%s" has already been downloaded: SKIPPING!', name)
                continue
        output_path.unlink(missing_ok=True)

        dest = _download_and_checksum(dset_name, config, output_path)
        t1 = time.time()
        logger.info('successfully downloaded dataset "%s" to file "%s"', config["url"], dest)
        logger.info(f"file size: %.2fMB. Elapsed time: %.2fs", dest.stat().st_size / (1024 << 10), t1 - t0)


def _download_data_for_unit_tests():
    output_dir = pathlib.Path("test/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    names = {
        "4DNFI9GMP2J8": pathlib.Path("test/data/4DNFI9GMP2J8.mcool"),
        "__results_v1": pathlib.Path("test/data/results_4DNFI9GMP2J8_v1.hdf5"),
        "__results_v2": pathlib.Path("test/data/results_4DNFI9GMP2J8_v2.hdf5"),
        "__stripepy_plot_images": pathlib.Path("test/data/stripepy-plot-test-images.tar.xz"),
    }

    _download_multiple(list(names.keys()), list(names.values()))


def _download_data_for_end2end_tests():
    _download_data_for_unit_tests()


def run(
    max_size: float,
    list_only: bool,
    unit_test: bool,
    end2end_test: bool,
    include_private: bool,
    force: bool,
    name: Optional[str] = None,
    output_path: Optional[pathlib.Path] = None,
    assembly: Optional[str] = None,
) -> int:
    t0 = time.time()
    if list_only:
        _list_datasets()
        return 0

    if unit_test:
        _download_data_for_unit_tests()
        return 0

    if end2end_test:
        _download_data_for_end2end_tests()
        return 0

    do_random_sample = name is None and assembly is None

    if do_random_sample:
        dset_name, config = _get_random_dataset(max_size, include_private)
    else:
        dset_name, config = _lookup_dataset(name, assembly, max_size, include_private)

    if output_path is None:
        if "filename" in config:
            output_path = pathlib.Path(config["filename"])
        else:
            output_path = pathlib.Path(f"{dset_name}.{config['format']}")

    if output_path.exists() and not force:
        raise RuntimeError(f"refusing to overwrite file {output_path}. Pass --force to overwrite.")
    output_path.unlink(missing_ok=True)

    dest = _download_and_checksum(dset_name, config, output_path)

    logger = structlog.get_logger()
    logger.info('successfully downloaded dataset "%s" to file "%s"', config["url"], dest)
    logger.info(
        f"file size: %.2fMB. Elapsed time: %s", dest.stat().st_size / (1024 << 10), pretty_format_elapsed_time(t0)
    )

    return 0
