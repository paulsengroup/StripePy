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

from stripepy.utils import pretty_format_elapsed_time


@functools.cache
def _get_datasets(max_size: float, include_private: bool) -> Dict[str, Dict[str, str]]:
    assert not math.isnan(max_size)

    record_id = "15301784"

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
        "__results_tables": {
            "url": f"https://zenodo.org/records/{record_id}/files/stripepy-call-result-tables.tar.xz?download=1",
            "md5": "04ef7694cbb68739f205c5030681c199",
            "filename": "stripepy-call-result-tables.tar.xz",
            "assembly": "hg38",
            "format": "tar",
            "size_mb": 19.58,
        },
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
        "__results_v3": {
            "url": f"https://zenodo.org/records/{record_id}/files/results_4DNFI9GMP2J8_v3.hdf5?download=1",
            "md5": "47c6b3ec62b53397d44cd1813caf678b",
            "filename": "results_4DNFI9GMP2J8_v3.hdf5",
            "assembly": "hg38",
            "format": "stripepy",
            "size_mb": 10.50,
        },
        "__stripepy_plot_images": {
            "url": f"https://zenodo.org/records/{record_id}/files/stripepy-plot-test-images.tar.xz?download=1",
            "md5": "e88d5a6ff33fb7cb0a15e27c5bac7644",
            "filename": "stripepy-plot-test-images.tar.xz",
            "assembly": "hg38",
            "format": "tar",
            "size_mb": 1.54,
        },
    }

    if include_private:
        datasets |= private_datasets

    valid_dsets = {k: v for k, v in datasets.items() if v.get("size_mb", math.inf) < max_size}

    if len(valid_dsets) > 0:
        return valid_dsets

    raise RuntimeError(f"unable to find any dataset smaller than {max_size:.2f} MB")


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
    main_logger=None,
) -> int:
    t0 = time.time()
    if list_only:
        _list_datasets()
        return 0

    if unit_test:
        _download_data_for_unit_tests(progress_bar=main_logger.progress_bar)
        return 0

    if end2end_test:
        _download_data_for_end2end_tests(progress_bar=main_logger.progress_bar)
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
        raise FileExistsError(f"refusing to overwrite file {output_path}. Pass --force to overwrite.")
    output_path.unlink(missing_ok=True)

    dest = _download_and_checksum(
        dset_name,
        config,
        output_path,
        progress_bar=main_logger.progress_bar,
    )

    logger = structlog.get_logger()
    logger.info('successfully downloaded dataset "%s" to file "%s"', config["url"], dest)
    logger.info(
        f"file size: %.2f MiB. Elapsed time: %s", dest.stat().st_size / (1024 << 10), pretty_format_elapsed_time(t0)
    )

    return 0


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


def _hash_file(path: pathlib.Path, progress_bar, chunk_size: int = 16 << 20) -> str:
    file_size = path.stat().st_size
    disable_progress_bar = file_size < (256 << 20)
    progress_bar_task_id = str(path.stem)

    logger = structlog.get_logger()
    logger.info('computing MD5 digest for file "%s"...', path)
    progress_bar.add_task(
        task_id=progress_bar_task_id,
        name=f"{path.stem}: checksumming",
        description="",
        start=True,
        total=file_size,
        visible=not disable_progress_bar,
    )
    with path.open("rb") as f:
        hasher = hashlib.md5()
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return hasher.hexdigest()
            hasher.update(chunk)
            progress_bar.update(task_id=progress_bar_task_id, advance=len(chunk))


def _fetch_remote_file_size(url: str) -> Optional[int]:
    try:
        req = urllib.request.Request(url)
        return int(urllib.request.urlopen(req).headers.get("Content-Length"))
    except Exception:  # noqa
        return None


def _download_progress_reporter(
    chunk_no,
    max_chunk_size,
    download_size,
    progress_bar,
    progress_bar_task_id,
    total,
):
    try:
        bytes_downloaded = min(chunk_no * max_chunk_size, total)
        progress_bar.update(
            task_id=progress_bar_task_id,
            advance=min(max_chunk_size, total - bytes_downloaded),
        )
    except Exception:  # noqa
        # There are rare scenarios where some of the params can be None, leading to all kinds of problems.
        # When this is the case, simply ignore any errors related to updating the progress bar and keep downloading.
        pass


def _download_and_checksum(
    name: str,
    dset: Dict[str, Any],
    dest: pathlib.Path,
    progress_bar,
):
    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=f"{dest.stem}.") as tmpfile:
        logger = structlog.get_logger()
        tmpfile.close()
        tmpfile = pathlib.Path(tmpfile.name)

        url = dset["url"]
        md5sum = dset["md5"]
        assembly = dset.get("assembly", "unknown")
        size = _fetch_remote_file_size(url)
        disable_progress_bar = size is None

        logger.info('downloading dataset "%s" (assembly=%s)...', name, assembly)
        t0 = time.time()
        progress_bar_task_id = name
        progress_bar.add_task(
            task_id=progress_bar_task_id,
            name=f"{name}: downloading",
            description="",
            start=True,
            total=size,
            visible=not disable_progress_bar,
        )

        urllib.request.urlretrieve(
            url,
            tmpfile,
            reporthook=functools.partial(
                _download_progress_reporter,
                progress_bar=progress_bar,
                progress_bar_task_id=progress_bar_task_id,
                total=size,
            ),
        )

        logger.info('DONE! Downloading dataset "%s" took %s.', name, pretty_format_elapsed_time(t0))

        digest = _hash_file(
            tmpfile,
            progress_bar,
        )
        if digest == md5sum:
            logger.info("MD5 checksum match!")
            return tmpfile.rename(dest)

        raise RuntimeError(
            f'MD5 checksum for file downloaded from "{url}" does not match: expected {md5sum}, found {digest}.'
        )


def _download_multiple(
    names: Sequence[str],
    output_paths: Sequence[pathlib.Path],
    progress_bar,
):
    assert len(names) == len(output_paths)

    logger = structlog.get_logger()

    for name, output_path in zip(names, output_paths):
        t0 = time.time()
        dset_name, config = _lookup_dataset(name, None, math.inf, include_private=True)

        if output_path.exists():
            logger.info('found existing file "%s"', output_path)
            digest = _hash_file(output_path, progress_bar)
            if digest == config["md5"]:
                logger.info('dataset "%s" has already been downloaded: SKIPPING!', name)
                continue
        output_path.unlink(missing_ok=True)

        dest = _download_and_checksum(dset_name, config, output_path, progress_bar)
        t1 = time.time()
        logger.info('successfully downloaded dataset "%s" to file "%s"', config["url"], dest)
        logger.info(f"file size: %.2f MiB. Elapsed time: %.2fs", dest.stat().st_size / (1024 << 10), t1 - t0)


def _download_data_for_unit_tests(progress_bar):
    output_dir = pathlib.Path("test/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    names = {
        "__results_tables": pathlib.Path("test/data/stripepy-call-result-tables.tar.xz"),
        "__results_v1": pathlib.Path("test/data/results_4DNFI9GMP2J8_v1.hdf5"),
        "__results_v2": pathlib.Path("test/data/results_4DNFI9GMP2J8_v2.hdf5"),
        "__results_v3": pathlib.Path("test/data/results_4DNFI9GMP2J8_v3.hdf5"),
    }

    _download_multiple(list(names.keys()), list(names.values()), progress_bar)


def _download_data_for_end2end_tests(progress_bar):
    output_dir = pathlib.Path("test/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    names = {
        "4DNFI9GMP2J8": pathlib.Path("test/data/4DNFI9GMP2J8.mcool"),
        "__results_v1": pathlib.Path("test/data/results_4DNFI9GMP2J8_v1.hdf5"),
        "__results_v2": pathlib.Path("test/data/results_4DNFI9GMP2J8_v2.hdf5"),
        "__results_v3": pathlib.Path("test/data/results_4DNFI9GMP2J8_v3.hdf5"),
        "__stripepy_plot_images": pathlib.Path("test/data/stripepy-plot-test-images.tar.xz"),
    }

    _download_multiple(list(names.keys()), list(names.values()), progress_bar)
